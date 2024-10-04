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

Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr) {
    return arr;
}

Eigen::Matrix<float, -1, 1> initial_range(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    for (int x = 0; x < mesh->_x; ++x) {
        for (int y = 0; y < mesh->_y; ++y) {
            for (int z = 0; z < mesh->_z; ++z) {
                auto node_id = mesh->square_node_coord_to_idx(x, y, z);
                ret(node_id) = x + 1;
            }
        }
    }
    return ret;
}


int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto rho_initial = initial_range(mesh.get());
    auto rho = Variable(mesh.get(), rho_initial, boundary_none, "rho");
    auto der = Grad2Var(&rho, true, false, false );
//    auto stb = StabVar(&rho, false, false, true);
    auto calc = der.evaluate();
//    auto calc_stb = stb.evaluate();

    auto calc_cu = der.evaluate_cu().to_eigen();
//    auto calc_stb_cu = stb.evaluate_cu().to_eigen();

    std::cout << "var:" << std::endl;

    for (int x = 0; x < mesh->_x; ++x) {
        for (int y = 0; y < mesh->_y; ++y) {
            for (int z = 0; z < mesh->_z; ++z) {
                auto node_id = mesh->square_node_coord_to_idx(x, y, z);
                std::cout << rho_initial(node_id) << " ";
            }
            std::cout << std::endl;
        }
    }


    std::cout << "\n\ndz:" << std::endl;

    for (int x = 0; x < mesh->_x; ++x) {
        for (int y = 0; y < mesh->_y; ++y) {
            for (int z = 0; z < mesh->_z; ++z) {
                auto node_id = mesh->square_node_coord_to_idx(x, y, z);
                std::cout << calc(node_id) << " ";
            }
            std::cout << std::endl;
        }
    }

//    std::cout << "\n\nstab:" << std::endl;
//
//    for (int x = 0; x < mesh->_x; ++x) {
//        for (int y = 0; y < mesh->_y; ++y) {
//            for (int z = 0; z < mesh->_z; ++z) {
//                auto node_id = mesh->square_node_coord_to_idx(x, y, z);
//                std::cout << calc_stb(node_id) << " ";
//            }
//            std::cout << std::endl;
//        }
//    }

    std::cout << "\n\ndz cu:" << std::endl;

    for (int x = 0; x < mesh->_x; ++x) {
        for (int y = 0; y < mesh->_y; ++y) {
            for (int z = 0; z < mesh->_z; ++z) {
                auto node_id = mesh->square_node_coord_to_idx(x, y, z);
                std::cout << calc_cu(node_id) << " ";
            }
            std::cout << std::endl;
        }
    }

//    std::cout << "\n\nstab cu:" << std::endl;
//
//    for (int x = 0; x < mesh->_x; ++x) {
//        for (int y = 0; y < mesh->_y; ++y) {
//            for (int z = 0; z < mesh->_z; ++z) {
//                auto node_id = mesh->square_node_coord_to_idx(x, y, z);
//                std::cout << calc_stb_cu(node_id) << " ";
//            }
//            std::cout << std::endl;
//        }
//    }


    initializer.finalize();
    return 0;
}