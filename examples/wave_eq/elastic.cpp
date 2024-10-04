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

int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto rho = Eigen::Matrix<float, -1, 1> {mesh->_num_nodes};
    rho.setConstant(2800);

    auto v_p_ = Eigen::Matrix<float, -1, 1> {mesh->_num_nodes};
    auto v_s_ = Eigen::Matrix<float, -1, 1> {mesh->_num_nodes};
    for (int x = 0; x < mesh->_x; ++x) {
        for (int y = 0; y < mesh->_y; ++y) {
            if (y < mesh->_y / 2) {
                v_p_[x * mesh->_y + y] = 3300;
                v_s_[x * mesh->_y + y] = 1905.31;
            } else {
                v_p_[x * mesh->_y + y] = 2600;
                v_s_[x * mesh->_y + y] = 1501.15;
            }
        }
    }

    auto v_p = Variable(mesh.get(), 3300);
    v_p.current = v_p_;
    auto v_s = Variable(mesh.get(), 3300);
    v_s.current = v_s_;

    auto A = 10e7;
    auto f_0 = 200;
    auto mu = (v_s * v_s) * rho;
    auto lambda = (v_p * v_p) * rho - 2*mu;

    auto initial_zero = initial_with_val(mesh.get(), 0);
    auto initial_vx = initial_with_val(mesh.get(), 0);
    auto initial_vy = initial_with_val(mesh.get(), 0);

    auto v_x = Variable(mesh.get(), initial_vx, boundary_copy(initial_vx), boundary_copy_cu(initial_zero), "v_x");
    auto v_y = Variable(mesh.get(), initial_vy, boundary_copy(initial_vy), boundary_copy_cu(initial_zero), "v_y");
    auto tau_xx = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_xx");
    auto tau_xy = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_xy");
    auto tau_yy = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "tau_yy");

    std::vector<Variable*> all_vars {&v_x, &v_y, &tau_xx, &tau_xy, &tau_yy};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.001, all_vars);


    auto central_point = Variable(mesh.get(), 0.0);
    sett(central_point.current, mesh->_x, mesh->_y, mesh->_y * 0.5, mesh->_x * 0.4, 1.0);
    auto curr_time = PointerVariable(mesh.get(), &dt._current_time_dbl);

    auto a = 3.14 * 3.14 * f_0 * f_0;
    auto time_shift = curr_time - (1.2 / f_0);
    auto s_base = -2 * A * a * time_shift * exp(-a * time_shift * time_shift);
    auto s = s_base * central_point;

    EquationTemplate equation_system = {
            {d1t(v_x),    '=', (d1dx(tau_xx) + d1dy(tau_xy) + s) / rho},
            {d1t(v_y),    '=', (d1dx(tau_xy) + d1dy(tau_yy) + s) / rho},
            {d1t(tau_xx), '=', (lambda + 2 * mu) * d1dx(v_x) + lambda * d1dy(v_y)},
            {d1t(tau_xy), '=', mu * (d1dx(v_y) + d1dy(v_x))},
            {d1t(tau_yy), '=', (lambda + 2 * mu) * d1dy(v_y) + lambda * d1dx(v_x)},
    };

    auto equation = Equation(timesteps);

    auto store_vars = {&v_x, &v_y};
    initializer.init_store(store_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, store_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();

    return 0;
}
