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
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "utils3d.hpp"


Eigen::Matrix<float, -1, 1> initial_rho(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.5 * mesh->_lx;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x()) {
            ret(i) = 1;
        } else {
            ret(i) = 0.125;
        }
        ++i;
    }
    return ret;
}


Eigen::Matrix<float, -1, 1> initial_p(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.5 * mesh->_lx;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x()) {
            ret(i) = 1;
        } else {
            ret(i) = 0.1;
        }
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) {
    return arr;
}

Eigen::Matrix<float, -1, 1> _boundary_copy_2d_via_3d(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& copy_var) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = mesh->_dx + 0.001;
    float x_limit_upper = mesh->_lx - mesh->_dx - 0.001;

//    for (auto& node : mesh->_nodes) {
//        if (node->_id == 0 || node->_id == mesh->_num_nodes - 1) {
////        if (x_limit_lower < node->x() && node->x() < x_limit_upper) {
////        if (node->is_boundary_x()) {
//            ret(i) = copy_var(i);
//        } else {
//            ret(i) = arr(i);
//        }
//        ++i;
//    }

    for (auto& node : mesh->_nodes) {
        if (node->_id == 0) {
            ret(node->_id) = arr(node->_id + 1);
        } else if (node->_id == mesh->_num_nodes - 1) {
            ret(node->_id) = arr(node->_id - 1);
        } else {
            ret(node->_id) = arr(node->_id);
        }
        ++i;
    }

    return ret;
}

auto boundary_copy(const Eigen::Matrix<float, -1, 1>& copy_var) {
    return [copy_var] (Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) { return _boundary_copy_2d_via_3d(mesh, arr, copy_var); };
}
int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto rho_initial = initial_rho(mesh.get());
    auto rho = Variable(mesh.get(), rho_initial, boundary_copy(rho_initial), "rho");
//    auto rho = Variable(mesh.get(), rho_initial, boundary_none, "rho");

    Eigen::Matrix<float, -1, 1> u_initial = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    u_initial.setConstant(0);
    auto u = Variable(mesh.get(), u_initial, boundary_copy(u_initial), "u");
//    auto u = Variable(mesh.get(), u_initial, boundary_none, "u");

    auto p_initial = initial_p(mesh.get());
    auto p = Variable(mesh.get(), p_initial, boundary_copy(p_initial), "p");
//    auto p = Variable(mesh.get(), p_initial, boundary_none, "p");

    Eigen::Matrix<float, -1, 1> mass_initial = rho.current.array() * mesh->_volumes.array();
    auto mass = Variable(mesh.get(), mass_initial, boundary_copy(mass_initial), "mass");
//    auto mass = Variable(mesh.get(), mass_initial, boundary_none, "mass");

    Eigen::Matrix<float, -1, 1> rho_u_initial = rho.current.array() * u.current.array() * mesh->_volumes.array();
    auto rho_u = Variable(mesh.get(), rho_u_initial, boundary_copy(rho_u_initial), "rho_u");
//    auto rho_u = Variable(mesh.get(), rho_u_initial, boundary_none, "rho_u");

//    float gamma = 5. / 3.;
    float gamma = 1.4;

    auto E = p / (gamma - 1) + 0.5 * rho * ((u * u));
    Eigen::Matrix<float, -1, 1> E_initial = (p.current.array() / (gamma - 1) + 0.5 * rho.current.array() * (u.current.array() * u.current.array())) * mesh->_volumes.array();
    auto rho_e = Variable(mesh.get(), E_initial, boundary_copy(E_initial), "rho_e");
//    auto rho_e = Variable(mesh.get(), E_initial, boundary_none, "rho_e");

    auto mass_tmp = Variable(mesh.get(), mass_initial,   boundary_none, "mass_tmp");
    auto rho_u_tmp = Variable(mesh.get(), rho_u_initial, boundary_none, "rho_u_tmp");
    auto rho_e_tmp = Variable(mesh.get(), E_initial,     boundary_none, "rho_e_tmp");

    std::vector<Variable*> space_vars {&u, &p, &rho};
    auto dt = DT(mesh.get(), UpdatePolicies::CourantFriedrichsLewy1D, UpdatePolicies::CourantFriedrichsLewy1DCu,  initializer.dt, space_vars);
//    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu,  initializer.dt, space_vars);

    auto volumes_var = Variable(mesh.get(), mesh->_volumes, boundary_none, "volumes_var");

    float dissip = 2;
    EquationTemplate equation_system = {
            {&rho,        '=', mass / volumes_var, true},
            {&u,          '=', rho_u / rho / volumes_var, true},
            {&p,          '=', (rho_e / volumes_var - 0.5 * rho * (u * u)) * (gamma - 1), true},

            {&rho,    '=', rho - 0.5 * dt * (u * d1dx(rho) + rho * d1dx(u)), true},
            {&u,      '=', u   - 0.5 * dt * (u * d1dx(u) + (1 / rho) * d1dx(p)), true},
            {&p,      '=', p   - 0.5 * dt * (gamma * p * (d1dx(u)) + u * d1dx(p)), true},

            {&mass_tmp,  '=', 0 * mass_tmp -  (d1dx(rho * u)), true},
            {&rho_u_tmp, '=', 0 * rho_u_tmp - (d1dx(rho * u * u + p)), true},
            {&rho_e_tmp, '=', 0 * rho_e_tmp - (d1dx((E + p) * u)), true},

            {d1t(mass),  '=', mass_tmp -  stabx(rho) * dissip, false},
            {d1t(rho_u), '=', rho_u_tmp - stabx(rho * u) * dissip, false},
            {d1t(rho_e), '=', rho_e_tmp - stabx(E) * dissip, false},

    };

    auto equation = Equation(timesteps);
    initializer.init_store({&rho, &u, &p});

    std::vector<Variable*> all_vars {&rho, &u, &p, &mass, &rho_u, &rho_e};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&rho, &u, &p});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}