/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2024 cfdARCHO

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
// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <iostream>
#include <matplot/matplot.h>
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "utils3d.hpp"


Eigen::Matrix<float, -1, 1> initial_rho(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.4 * mesh->_lx;
    float x_limit_upper = 0.6 * mesh->_lx;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x() && node->x() < x_limit_upper) {
            ret(i) = 2;
        } else {
            ret(i) = 1;
        }
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_v(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.4 * mesh->_lx;
    float x_limit_upper = 0.6 * mesh->_lx;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x() && node->x() < x_limit_upper) {
            ret(node->_id) = -1;
        } else {
            ret(node->_id) = 1;
        }
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_u(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.4 * mesh->_lx;
    float x_limit_upper = 0.6 * mesh->_lx;

    for (auto& node : mesh->_nodes) {
//        ret(node->_id) = 0.3 * std::sin(4 * M_PI * node->x() / mesh->_x);
        ret(node->_id) = 0;
//        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr) {
    return arr;
}

Eigen::Matrix<float, -1, 1> _boundary_copy_2d_via_3d(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& copy_var) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = mesh->_dx + 0.001;
    float x_limit_upper = mesh->_lx - mesh->_dx - 0.001;
    float z_limit_lower = mesh->_dz + 0.001;
    float z_limit_upper = mesh->_lz - mesh->_dz - 0.001;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x() && node->x() < x_limit_upper && z_limit_lower < node->z() && node->z() < z_limit_upper) {
            ret(i) = arr(i);
        } else {
            ret(i) = copy_var(i);
        }
        ++i;
    }

    return ret;
}

auto boundary_copy(const Eigen::Matrix<float, -1, 1>& copy_var) {
    return [copy_var] (Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr) { return _boundary_copy_2d_via_3d(mesh, arr, copy_var); };
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto rho_initial = initial_rho(mesh.get());
    auto rho = Variable(mesh.get(), rho_initial, boundary_copy(rho_initial), "rho");

    auto u_initial = initial_u(mesh.get());
    auto u = Variable(mesh.get(), u_initial, boundary_copy(u_initial), "u");

    auto v_initial = initial_v(mesh.get());
    auto v = Variable(mesh.get(), v_initial, boundary_copy(v_initial), "v");

    Eigen::Matrix<float, -1, 1> p_initial = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    p_initial.setConstant(2.5);
    auto p = Variable(mesh.get(), p_initial, boundary_copy(p_initial), "p");

    Eigen::Matrix<float, -1, 1> mass_initial = rho.current.array() * mesh->_volumes.array();
    auto mass = Variable(mesh.get(), mass_initial, boundary_copy(mass_initial), "mass");

    Eigen::Matrix<float, -1, 1> rho_u_initial = rho.current.array() * u.current.array() * mesh->_volumes.array();
    auto rho_u = Variable(mesh.get(), rho_u_initial, boundary_copy(rho_u_initial), "rho_u");

    Eigen::Matrix<float, -1, 1> rho_v_initial = rho.current.array() * v.current.array() * mesh->_volumes.array();
    auto rho_v = Variable(mesh.get(), rho_v_initial, boundary_copy(rho_v_initial), "rho_v");

    float gamma = 5. / 3.;

    auto E = p / (gamma - 1) + 0.5 * rho * ((u * u) + (v * v));
    Eigen::Matrix<float, -1, 1> E_initial = (p.current.array() / (gamma - 1) + 0.5 * rho.current.array() * (u.current.array() * u.current.array() + v.current.array() * v.current.array())) * mesh->_volumes.array();
    auto rho_e = Variable(mesh.get(), E_initial, boundary_copy(E_initial), "rho_e");

    std::vector<Variable*> space_vars {&u, &v, &p, &rho};
    auto dt = DT(mesh.get(), UpdatePolicies::CourantFriedrichsLewy, initializer.dt, space_vars);

    EquationTemplate equation_system = {
            {&rho,        '=', mass / mesh->_volumes},
            {&u,          '=', rho_u / rho / mesh->_volumes},
            {&v,          '=', rho_v / rho / mesh->_volumes},
            {&p,          '=', (rho_e / mesh->_volumes - 0.5 * rho * (u * u + v * v)) * (gamma - 1)},

            {&rho,    '=', rho - 0.5 * dt * (u * d1dx(rho) + rho * d1dx(u) + v * d1dz(rho) + rho * d1dz(v))},
            {&u,      '=', u - 0.5 * dt * (u * d1dx(u) + v * d1dz(u) + (1 / rho) * d1dx(p))},
            {&v,      '=', v - 0.5 * dt * (u * d1dx(v) + v * d1dz(v) + (1 / rho) * d1dz(p))},
            {&p,      '=', p - 0.5 * dt * (gamma * p * (d1dx(u) + d1dz(v)) + u * d1dx(p) + v * d1dz(p))},

            {d1t(mass),  '=', -((d1dx(rho * u) + d1dz(rho * v)) + stab_tot(rho) * 2)},
            {d1t(rho_u), '=', -((d1dx(rho * u * u + p) + d1dz(rho * v * u)) + stab_tot(rho * u) * 2)},
            {d1t(rho_v), '=', -((d1dx(rho * v * u) + d1dz(rho * v * v + p)) + stab_tot(rho * v) * 2)},
            {d1t(rho_e), '=', -((d1dx((E + p) * u) + d1dz((E + p) * v)) + stab_tot(E) * 2)},


    };


    auto equation = Equation(timesteps);
    initializer.init_store({&rho});

    std::vector<Variable*> all_vars {&rho, &u, &v, &p, &mass, &rho_u, &rho_v, &rho_e};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&rho});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}