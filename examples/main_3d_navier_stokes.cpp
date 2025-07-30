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
#include "operators.hpp"
#include "equation.hpp"
#include "utils3d.hpp"


Eigen::Matrix<float, -1, 1> get_cavity_source(Mesh3D *mesh) {
    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float z = node.z() / mesh->_lx;
        if (node.is_boundary_z() && z < 0.5) {
            res(i) = 1.0;
        }
    }
    return res;
}


int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto cavity_source_ = get_cavity_source(mesh.get());
    auto cavity_source = Variable(mesh.get(), cavity_source_, boundary_none, boundary_none_cu, "cavity_source");

    // Physical parameters for lid-driven cavity
    const float rho = 1.0f;  // Density
    const float mu = 0.001f; // Dynamic viscosity (Re = 1000)
    const float nu = mu / rho; // Kinematic viscosity
    const float lid_velocity = 1.0f; // Lid velocity

    // Initialize velocity fields
    Eigen::Matrix<float, -1, 1> u_init = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> v_init = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> w_init = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> p_init = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    
    // Create variables with appropriate boundary conditions
    auto u = Variable(mesh.get(), u_init, zero_gradient_bc, zero_gradient_bc_cu, "u");
    auto v = Variable(mesh.get(), v_init, zero_gradient_bc, zero_gradient_bc_cu, "v");
    auto w = Variable(mesh.get(), w_init, zero_gradient_bc, zero_gradient_bc_cu, "w");
    auto p = Variable(mesh.get(), p_init, boundary_none, boundary_none_cu, "p");
    
    // Intermediate variables for pressure correction
    auto u_star = Variable(mesh.get(), u_init, boundary_none, boundary_none_cu, "u_star");
    auto v_star = Variable(mesh.get(), v_init, boundary_none, boundary_none_cu, "v_star");
    auto w_star = Variable(mesh.get(), w_init, boundary_none, boundary_none_cu, "w_star");

    // Time step variables
    std::vector<Variable *> space_vars{&u, &v, &w};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    // Navier-Stokes equations with pressure-velocity coupling using kernel building
    EquationTemplate equation_system = {
        NEntryVar{
                {&u, '=', u + cavity_source, true},
        }.restruct(),

        // Step 1: Predictor step (momentum equations without pressure) - FUSED KERNEL
        NEntryVar{
            {&u_star, '=', u + dt * (-u * d1dx(u) - v * d1dy(u) - w * d1dz(u) + nu * (d2dx(u) + d2dy(u) + d2dz(u))), true},
            {&v_star, '=', v + dt * (-u * d1dx(v) - v * d1dy(v) - w * d1dz(v) + nu * (d2dx(v) + d2dy(v) + d2dz(v))), true},
            {&w_star, '=', w + dt * (-u * d1dx(w) - v * d1dy(w) - w * d1dz(w) + nu * (d2dx(w) + d2dy(w) + d2dz(w))), true}
        }.restruct(),

        // Step 2: Pressure Poisson equation - FUSED KERNEL
        NEntryVar{
            {&p, '=', p + dt * (d2dx(p) + d2dy(p) + d2dz(p) - rho / dt * (d1dx(u_star) + d1dy(v_star) + d1dz(w_star))), true}
        }.restruct(),

        // Step 3: Corrector step (velocity update with pressure gradient) - FUSED KERNEL
        NEntryVar{
            {&u, '=', u_star - dt / rho * d1dx(p), true},
            {&v, '=', v_star - dt / rho * d1dy(p), true},
            {&w, '=', w_star - dt / rho * d1dz(p), true}
        }.restruct(),

    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{&u, &v, &w, &p, &u_star, &v_star, &w_star};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\nLid-Driven Cavity Flow Simulation Completed!" << std::endl;
        std::cout << "Reynolds number: " << lid_velocity * mesh->_lx / nu << std::endl;
        std::cout << "Expected features: Primary vortex, secondary corner vortices" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
} 