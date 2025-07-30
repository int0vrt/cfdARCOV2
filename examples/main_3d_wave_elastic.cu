/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2025 cfdARCO team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <chrono>
#include <argparse/argparse.hpp>
#include <cmath>

#include "operators.hpp"
#include "equation.hpp"
#include "utils3d.hpp"

Eigen::Matrix<float, -1, 1> get_damp_matrix(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, float dt) {
    const float damping_width = 0.1f; // percentage of domain
    const float alpha_max = 2.0f;
    const int power = 4; // polynomial order
    Eigen::Matrix<float, -1, 1> damped = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    damped.setOnes();

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float dx = std::min(1.0f, std::min(node.x(), mesh->_lx - node.x()) / (damping_width * mesh->_lx));
        float dy = std::min(1.0f, std::min(node.y(), mesh->_ly - node.y()) / (damping_width * mesh->_ly));
        float dz = std::min(1.0f, std::min(node.z(), mesh->_lz - node.z()) / (damping_width * mesh->_lz));
        float d = std::min({dx, dy, dz, 1.0f});
        if (d < 1.0f) {
            d = dx * dy * dz;
            float alpha = alpha_max * std::pow(1.0f - d, power);
            damped(i) *= std::exp(-alpha * dt);
        }

    }
    return damped;
}

// Mineral heterogeneity setup
struct Mineral {
    float rho;
    float vp;
    float vs;
    float center_x;
    float center_y;
    float center_z;
    float radius;
};

auto assign_minerals(Mesh3D *mesh) {
    std::vector<Mineral> regions = {
            {2700.0f, 6000.0f, 3464.0f, 0.5f, 0.5f, 0.8f, 0.3f}, // granite (deep)
            {2500.0f, 4500.0f, 2600.0f, 0.5f, 0.5f, 0.5f, 0.3f}, // sandstone (middle)
            {2200.0f, 3200.0f, 1800.0f, 0.5f, 0.5f, 0.2f, 0.3f}  // soil (top)
    };

    Eigen::Matrix<float, -1, 1> rho(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> lambda(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> mu(mesh->_num_nodes);

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float x = node.x() / mesh->_lx;
        float y = node.y() / mesh->_ly;
        float z = node.z() / mesh->_lz;

        bool assigned = false;
        for (const auto &m: regions) {
            float dx = x - m.center_x;
            float dy = y - m.center_y;
            float dz = z - m.center_z;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 < m.radius * m.radius) {
                rho(i) = m.rho;
                mu(i) = m.rho * m.vs * m.vs;
                lambda(i) = m.rho * (m.vp * m.vp - 2 * m.vs * m.vs);
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            rho(i) = 2500.0f;
            mu(i) = 2500.0f * 3000.0f * 3000.0f;
            lambda(i) = 2500.0f * (5000.0f * 5000.0f - 2 * 3000.0f * 3000.0f);
        }
    }
    return std::tuple{
            rho,
            lambda,
            mu
    };
}


int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    Eigen::Matrix<float, -1, 1> zero_field = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    auto [rho_v, lambda_v, mu_v] = assign_minerals(mesh.get());
    auto damp_matrix = get_damp_matrix(mesh.get(), zero_field, initializer.dt);

    auto rho =      Variable(mesh.get(), rho_v,     boundary_none, boundary_none_cu, "rho");
    auto lambda =   Variable(mesh.get(), lambda_v,  boundary_none, boundary_none_cu, "lambda");
    auto mu =       Variable(mesh.get(), mu_v,      boundary_none, boundary_none_cu, "mu");
    auto damp = Variable(mesh.get(), damp_matrix, boundary_none, boundary_none_cu, "damp");

    auto vel = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vel");

    auto vx = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vx");
    auto vy = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vy");
    auto vz = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vz");

    auto vx_interp_y = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vx_interp_y");
    auto vx_interp_z = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vx_interp_z");
    auto vy_interp_x = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vy_interp_x");
    auto vy_interp_z = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vy_interp_z");
    auto vz_interp_x = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vz_interp_x");
    auto vz_interp_y = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vz_interp_y");

    auto sxx = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxx");
    auto syy = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "syy");
    auto szz = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "szz");
    auto sxy = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxy");
    auto sxz = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxz");
    auto syz = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "syz");

    auto sxy_interp_x = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxy_interp_x");
    auto sxz_interp_x = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxz_interp_x");
    auto sxy_interp_y = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxy_interp_y");
    auto syz_interp_y = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "syz_interp_y");
    auto sxz_interp_z = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "sxz_interp_z");
    auto syz_interp_z = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "syz_interp_z");

    std::vector<Variable *> space_vars{&vel};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    auto ricker_fill = [](float current_time){
        float f0 = 20.f; // dominant freq in Hz
        auto pi_f0_t = static_cast<float>(M_PI) * f0 * current_time;
        auto source_val = (1.0f - 2.0f * pi_f0_t * pi_f0_t) * std::exp(-pi_f0_t * pi_f0_t);
        return source_val;
    };

    auto source1 = PointSource(mesh.get(), &dt, timesteps,
                               mesh->_lx * 0.5, mesh->_ly * 0.5, mesh->_lz * 0.0,
                               mesh->_lx * 0.05, ricker_fill, "ricker1");

    EquationTemplate equation_system = {
            NEntryVar{
                    {&vel, '=', d1dx(vx) + d1dy(vy) + d1dz(vz), true},
            }.restruct(),

            NEntryVar{
                    {&vx,  '=', damp * vx,  true},
                    {&vy,  '=', damp * vy,  true},
                    {&vz,  '=', damp * vz,  true},
                    {&sxx, '=', damp * sxx + source1, true},
                    {&syy, '=', damp * syy + source1, true},
                    {&szz, '=', damp * szz + source1, true},
                    {&sxy, '=', damp * sxy + source1, true},
                    {&sxz, '=', damp * sxz + source1, true},
                    {&syz, '=', damp * syz + source1, true},
            }.restruct(),

            NEntryVar{
                {&sxy_interp_x, '=', 1.0 * interp_x(sxy), true},
                {&syz_interp_y, '=', 1.0 * interp_y(syz), true},
                {&sxz_interp_z, '=', 1.0 * interp_z(sxz), true},
                {&sxz_interp_x, '=', 1.0 * interp_x(sxz), true},
                {&sxy_interp_y, '=', 1.0 * interp_y(sxy), true},
                {&syz_interp_z, '=', 1.0 * interp_z(syz), true}
            }.restruct(),

            NEntryVar{{&vx, '=', vx + dt / rho * (d1dx(sxx)             + d1dy(sxy_interp_y)    + d1dz(sxz_interp_z)), true},
                      {&vy, '=', vy + dt / rho * (d1dx(sxy_interp_x)    + d1dy(syy)             + d1dz(syz_interp_z)), true},
                      {&vz, '=', vz + dt / rho * (d1dx(sxz_interp_x)    + d1dy(syz_interp_y)    + d1dz(szz)         ), true}}.restruct(),

            NEntryVar{
                {&vx_interp_y, '=', 1.0 * interp_y(vx, true), true},
                {&vz_interp_x, '=', 1.0 * interp_x(vz, true), true},
                {&vy_interp_z, '=', 1.0 * interp_z(vy, true), true},
                {&vy_interp_x, '=', 1.0 * interp_x(vy, true), true},
                {&vx_interp_z, '=', 1.0 * interp_z(vx, true), true},
                {&vz_interp_y, '=', 1.0 * interp_y(vz, true), true},
            }.restruct(),

            NEntryVar{{&sxx, '=', sxx + dt * ((lambda + 2 * mu) * d1dx(vx) + lambda * (d1dy(vy) + d1dz(vz))), true},
                      {&syy, '=', syy + dt * ((lambda + 2 * mu) * d1dy(vy) + lambda * (d1dx(vx) + d1dz(vz))), true},
                      {&szz, '=', szz + dt * ((lambda + 2 * mu) * d1dz(vz) + lambda * (d1dx(vx) + d1dy(vy))), true},
                      {&sxy, '=', sxy + dt * mu * (d1dy(vx_interp_y) + d1dx(vy_interp_x))                   , true},
                      {&sxz, '=', sxz + dt * mu * (d1dz(vx_interp_z) + d1dx(vz_interp_x))                   , true},
                      {&syz, '=', syz + dt * mu * (d1dz(vy_interp_z) + d1dy(vz_interp_y))                   , true}}.restruct(),
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{
            &vx, &vy, &vz,
            &sxx, &syy, &szz, &sxy, &sxz, &syz,
            &vel,
            &sxy_interp_x, &sxz_interp_x, &sxy_interp_y, &syz_interp_y, &sxz_interp_z, &syz_interp_z,
            &vx_interp_y, &vy_interp_x, &vx_interp_z, &vz_interp_x, &vy_interp_z, &vz_interp_y
    };

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Elastic Wave Simulation Completed!" << std::endl;
        std::cout << "Heterogeneous mineral distribution: granite, sandstone, soil" << std::endl;
        std::cout << "Boundary condition: Absorbing boundaries with damping" << std::endl;
        std::cout << "Source: Ricker wavelet (f0 = 25 Hz)" << std::endl;
        std::cout << "Expected features: P-waves, S-waves, reflections, wave propagation" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
}