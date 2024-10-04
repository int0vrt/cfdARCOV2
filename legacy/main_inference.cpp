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

#include "mesh2d.hpp"
#include "fvm.hpp"
#include "cfdarcho_main.hpp"
#include "io_operators.hpp"


int main(int argc, char **argv) {
    CFDArcoGlobalInit::initialize(argc, argv, false);
    auto mesh = read_mesh();
    std::cout << "Mesh read" << std::endl;
    auto [vars, history_count] = init_read_history_stepping(mesh.get());
    std::cout << "Vars read" << std::endl;

    auto& rho = vars.at(0);

    auto fig = matplot::figure(true);
    for (int i = 0; i < history_count - 1; ++i) {
//        if (i % 50 != 0) continue;
        std::cout << "Reading step " << i << std::endl;
        read_history_stepping(mesh.get(), {&rho}, i);
        std::cout << "Reading step " << i << " done" << std::endl;
        auto grid_hist = to_grid_local(mesh.get(), rho.current);
        auto vect = from_eigen_matrix<double>(grid_hist);
        fig->current_axes()->image(vect);
        fig->draw();
        std::this_thread::sleep_for(std::chrono::milliseconds {50});
    }

    CFDArcoGlobalInit::finalize();
    return 0;
}