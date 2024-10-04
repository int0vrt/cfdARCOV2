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



int main(int argc, char **argv) {
//  Step 1: INITIALIZATION
    SingleLibInitializer initializer{argc, argv};

    auto mesh = read_mesh("mesh_file.json");
    mesh->compute();

    CFDArcoGlobalInit::make_node_distribution(mesh.get(), DistributionStrategy::Linear);
    CFDArcoGlobalInit::enable_cuda(mesh.get());

//    Step 2: EQUATION CREATION
    auto initial_T = initial_with_val(mesh.get(), 0);
    auto T = Variable(mesh.get(), initial_T, boundary_zero, boundary_zero_cu, "T");
    std::vector<Variable*> all_variables = {&T};

    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {d1t(T), '=', 5.0 * (d2dx(T) + d2dy(T))},
    };
    auto equation = Equation(initializer.timesteps);

//    Step 3: EQUATION EVALUATION
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.005);
    equation.evaluate(all_variables, equation_system, &dt, initializer.visualize, all_variables);

    initializer.finalize();
    return 0;
}