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

Eigen::Matrix<float, -1, 1> initial_T(Mesh2D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    ret.setConstant(0);
    sett(ret, mesh->_x, mesh->_y, mesh->_y * 0.25, mesh->_x * 0.25, 100.0);
    sett(ret, mesh->_x, mesh->_y, mesh->_y * 0.75, mesh->_x * 0.75, 100.0);
    return ret;
}

int main(int argc, char **argv) {
    SingleLibInitializer initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    Eigen::Matrix<float, -1, 1> T_initial = initial_T(mesh.get());
    auto T = Variable(mesh.get(), T_initial, boundary_copy(T_initial), boundary_copy_cu(T_initial), "T");

    std::vector<Variable*> space_vars {&T};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.005, space_vars);

    EquationTemplate equation_system = {
            {d1t(T), '=', 5.0 * lapl(T)},
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(space_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}