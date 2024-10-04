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
#include "io_operators.hpp"
#include "utils.hpp"
#include "val_utils.hpp"

Eigen::Matrix<float, -1, 1> boundary_sine(Mesh2D* mesh, Eigen::Matrix<float, -1, 1>& arr) {
    static int ii = 0;

    ++ii;

    Eigen::Matrix<float, -1, 1> ret{arr};
    sett(ret, mesh->_x, mesh->_y, mesh->_x * 0.25, mesh->_y * 0.25, std::sin(static_cast<float>(ii) * 0.1));
    sett(ret, mesh->_x, mesh->_y, mesh->_x * 0.75, mesh->_y * 0.75, std::sin(static_cast<float>(ii) * 0.1));

    return ret;
}

CudaDataMatrix boundary_sine_cu(Mesh2D* mesh, CudaDataMatrix& arr) {
    static int ii = 0;
    ++ii;

    CudaDataMatrix arr_n{arr};
    arr_n.set(mesh->_x, mesh->_y, mesh->_x * 0.25, mesh->_y * 0.25, std::sin(static_cast<float>(ii) * 0.1));
    arr_n.set(mesh->_x, mesh->_y, mesh->_x * 0.75, mesh->_y * 0.75, std::sin(static_cast<float>(ii) * 0.1));

    return arr_n;
}

int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto initial_zero = initial_with_val(mesh.get(), 0);
    auto p_x = Variable(mesh.get(), initial_zero, boundary_sine, boundary_sine_cu, "p_x");
    auto p_y = Variable(mesh.get(), initial_zero, boundary_sine, boundary_sine_cu, "p_y");
    auto v_x = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "v_x");
    auto v_y = Variable(mesh.get(), initial_zero, boundary_copy(initial_zero), boundary_copy_cu(initial_zero), "v_y");

    std::vector<Variable*> all_vars {&p_x, &p_y, &v_x, &v_y};
    std::vector<Variable*> store_vars {&p_x, &p_y};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.1, all_vars);

    float kappa = 2.2 * 10e2;
    float rho = 1;

    EquationTemplate equation_system = {
            {d1t(v_x), '=', -1 / rho * (d1dx(p_x) + d1dx(p_y))},
            {d1t(v_y), '=', -1 / rho * (d1dy(p_x) + d1dy(p_y))},
            {d1t(p_x), '=', -kappa * d1dx(v_x)},
            {d1t(p_y), '=', -kappa * d1dy(v_y)},
    };

    auto equation = Equation(timesteps);
    initializer.init_store(store_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, store_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}