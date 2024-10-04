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

#include "mesh2d.hpp"
#include "fvm.hpp"
#include "cfdarcho_main.hpp"
#include "utils.hpp"
#include "val_utils.hpp"


Eigen::Matrix<float, -1, 1> initial_rho(Mesh2D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (.4 < node->y() && node->y() < .6) {
            ret(i) = 2;
        } else {
            ret(i) = 1;
        }
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_u(Mesh2D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (.4 < node->y() && node->y() < .6) {
            ret(i) = 0.5;
        } else {
            ret(i) = -0.5;
        }
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_v(Mesh2D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (.4 < node->x() && node->x() < .6) {
            ret(i) = -0.3;
        } else {
            ret(i) = -0.5;
        }
        ++i;
    }
    return ret;
}


int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    float CFL = 0.5;
    float gamma = 5. / 3.;

    auto rho_initial = initial_rho(mesh.get());
    auto rho = Variable(mesh.get(), rho_initial, boundary_copy(rho_initial), boundary_copy_cu(rho_initial), "rho");

    auto u_initial = initial_u(mesh.get());
    auto u = Variable(mesh.get(), u_initial, boundary_copy(u_initial), boundary_copy_cu(u_initial), "u");

    auto v_initial = initial_with_val(mesh.get(), 0);
    auto v = Variable(mesh.get(), v_initial, boundary_copy(v_initial), boundary_copy_cu(v_initial), "v");

    Eigen::Matrix<float, -1, 1> p_initial = initial_with_val(mesh.get(), 2.5);
    auto p = Variable(mesh.get(), p_initial, boundary_copy(p_initial), boundary_copy_cu(p_initial), "p");

    Eigen::Matrix<float, -1, 1> mass_initial = rho.current.array() * mesh->_volumes.array();
    auto mass = Variable(mesh.get(), mass_initial, boundary_copy(mass_initial), boundary_copy_cu(mass_initial), "mass");

    Eigen::Matrix<float, -1, 1> rho_u_initial = rho.current.array() * u.current.array() * mesh->_volumes.array();
    auto rho_u = Variable(mesh.get(), rho_u_initial, boundary_copy(rho_u_initial), boundary_copy_cu(rho_u_initial), "rho_u");

    Eigen::Matrix<float, -1, 1> rho_v_initial = rho.current.array() * v.current.array() * mesh->_volumes.array();
    auto rho_v = Variable(mesh.get(), rho_v_initial, boundary_copy(rho_v_initial), boundary_copy_cu(rho_v_initial), "rho_v");

    auto E = p / (gamma - 1) + 0.5 * rho * ((u * u) + (v * v));
    Eigen::Matrix<float, -1, 1> E_initial = (p.current.array() / (gamma - 1) + 0.5 * rho.current.array() * (u.current.array() * u.current.array() + v.current.array() * v.current.array())) * mesh->_volumes.array();
    auto rho_e = Variable(mesh.get(), E_initial, boundary_copy(E_initial), boundary_copy_cu(E_initial), "rho_e");

    std::vector<Variable*> space_vars {&u, &v, &p, &rho};
    auto dt = DT(mesh.get(), UpdatePolicies::CourantFriedrichsLewy, UpdatePolicies::CourantFriedrichsLewyCu, CFL, space_vars);


    EquationTemplate equation_system = {
            {&rho,        '=', mass / mesh->_volumes},
            {&u,          '=', rho_u / rho / mesh->_volumes},
            {&v,          '=', rho_v / rho / mesh->_volumes},
            {&p,          '=', (rho_e / mesh->_volumes - 0.5 * rho * (u * u + v * v)) * (gamma - 1)},

            {&rho,    '=', rho - 0.5 * dt * (u * rho.dx() + rho * u.dx() + v * rho.dy() + rho * v.dy())},
            {&u,      '=', u - 0.5 * dt * (u * u.dx() + v * u.dy() + (1 / rho) * p.dx())},
            {&v,      '=', v - 0.5 * dt * (u * v.dx() + v * v.dy() + (1 / rho) * p.dy())},
            {&p,      '=', p - 0.5 * dt * (gamma * p * (u.dx() + v.dy()) + u * p.dx() + v * p.dy())},

            {d1t(mass),  '=', -((d1dx(rho * u) + d1dy(rho * v)) - stab_tot(rho) * 2)},
            {d1t(rho_u), '=', -((d1dx(rho * u * u + p) + d1dy(rho * v * u)) - stab_tot(rho * u) * 2)},
            {d1t(rho_v), '=', -((d1dx(rho * v * u) + d1dy(rho * v * v + p)) - stab_tot(rho * v) * 2)},
            {d1t(rho_e), '=', -((d1dx((E + p) * u) + d1dy((E + p) * v)) - stab_tot(E) * 2)},

    };

    auto equation = Equation(timesteps);

    std::vector<Variable*> all_vars {&rho, &u, &v, &p, &mass, &rho_u, &rho_v, &rho_e};

    initializer.init_store({&rho});

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&rho});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}