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
// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <iostream>
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "operators.hpp"
#include "equation.hpp"
#include "utils3d.hpp"


Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
    return arr;
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    Eigen::Matrix<float, -1, 1> zero_init = Eigen::Matrix<float, -1, 1>::Ones(mesh->_num_nodes);
    auto test_var = Variable(mesh.get(), zero_init, boundary_none, "test_var");

    std::vector<Variable *> space_vars{&test_var};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    EquationTemplate equation_system = {
            {&test_var, '=', test_var + dt * (d1dx(test_var) + d1dz(test_var) + d1dz(test_var)), true},
    };


    auto equation = Equation(timesteps);
    initializer.init_store({&test_var});

    std::vector<Variable *> all_vars{&test_var};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&test_var});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0)
        std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(
                end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}