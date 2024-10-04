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

Eigen::Matrix<float, -1, 1> initial_T(Mesh2D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
//        if (mesh->_edges.at(node->_edges_id.at(2))->is_boundary()) {
//            ret(i) = 100.0;
//        } else {
//            ret(i) = 0;
//        }
        ret(i) = 0;
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> boundary_T(Mesh2D* mesh, Eigen::Matrix<float, -1, 1>& arr) {
    static int ii = 0;

    ++ii;

    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
//        if (mesh->_edges.at(node->_edges_id.at(2))->is_boundary()) {
        if (node->x() < 0.01 && ((node->y() > 0.24 && node->y() < 0.26) || (node->y() > 0.74 && node->y() < 0.76))) {
            ret(i) = std::sin(static_cast<float>(ii) * 0.01);
        } else {
            ret(i) = arr(i);
        }
        ++i;
    }
    return ret;
}

CudaDataMatrix boundary_T_cu(Mesh2D* mesh, CudaDataMatrix& arr) {
    static int ii = 0;
    ++ii;

    CudaDataMatrix arr_n{arr};
    arr_n.set(mesh->_x, mesh->_y, 0, mesh->_y * 0.25, std::sin(static_cast<float>(ii) * 0.01) * 10);
    arr_n.set(mesh->_x, mesh->_y, 0, mesh->_y * 0.75, std::sin(static_cast<float>(ii) * 0.01) * 5);

    return arr_n;
}


Eigen::Matrix<float, -1, 1> _boundary_copy(Mesh2D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& copy_var) {
    auto arr1 = arr.cwiseProduct(mesh->_node_is_boundary_reverce);
    auto copy_var1 = copy_var.cwiseProduct(mesh->_node_is_boundary);
    return arr1 + copy_var1;
}

CudaDataMatrix _boundary_copy_cu(Mesh2D* mesh, CudaDataMatrix& arr, const CudaDataMatrix& copy_var) {
    auto arr1 = arr * mesh->_node_is_boundary_reverce_cu;
    auto copy_var1 = copy_var * mesh->_node_is_boundary_cu;
    return arr1 + copy_var1;
}

auto boundary_copy(const Eigen::Matrix<float, -1, 1>& copy_var) {
    return [copy_var] (Mesh2D* mesh, Eigen::Matrix<float, -1, 1>& arr) { return _boundary_copy(mesh, arr, copy_var); };
}

auto boundary_copy_cu(const Eigen::Matrix<float, -1, 1>& copy_var) {
    CudaDataMatrix cuda_copy_var;
    if (CFDArcoGlobalInit::cuda_enabled)
        cuda_copy_var = CudaDataMatrix::from_eigen(copy_var);
    return [cuda_copy_var] (Mesh2D* mesh, CudaDataMatrix& arr) { return _boundary_copy_cu(mesh, arr, cuda_copy_var); };
}

Eigen::Matrix<float, -1, 1> _boundary_neumann(Mesh2D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& grad_var) {
    auto redist = CFDArcoGlobalInit::get_redistributed(arr, "boundary_with_neumann");
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto& node : mesh->_nodes) {
        if (node->is_boundary()) {
            int q = 0;
            MatrixX4dRB is_bound{1, 4};
            is_bound.setConstant(0.0);
            MatrixX4dRB is_not_bound{1, 4};
            is_not_bound.setConstant(1.0);

            for (auto edge_id : node->_edges_id) {
                auto edge = mesh->_edges.at(edge_id);
                if (edge->is_boundary()) {
                    is_bound(q) = 1.0;
                    is_not_bound(q) = 0.0;
                }
                q++;
            }

            auto grad_cur_val = grad_var(i);
            auto nominator = 2*grad_cur_val*node->_volume;
            for (int j = 0; j < 4; ++j) {
                nominator = nominator + is_not_bound(j) * redist.at(j)(i) * mesh->_normal_y(i, j);
            }

            auto ghost_normals = mesh->_normal_y.block<1, 4>(i, 0).cwiseProduct(is_bound);
            auto demon = mesh->_normal_y.row(i).sum() + ghost_normals.sum();
            ret(i) = nominator / demon;

        } else {
            ret(i) = arr(i);
        }
        ++i;
    }
    return ret;
}

auto boundary_neumann(const Eigen::Matrix<float, -1, 1>& grad_var) {
    return [grad_var] (Mesh2D* mesh, Eigen::Matrix<float, -1, 1>& arr) { return _boundary_neumann(mesh, arr, grad_var); };
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program("cfdARCO");
    program.add_argument("-v", "--visualize").default_value(false).implicit_value(true);
    program.add_argument("--create_plot").default_value(false).implicit_value(true);
    program.add_argument("-L")
            .help("square size")
            .default_value(200)
            .scan<'i', int>();
    program.add_argument("-t", "--timesteps")
            .help("timesteps")
            .default_value(1000)
            .scan<'i', int>();
    program.add_argument("-c", "--cuda_enable").default_value(false).implicit_value(true);
    program.add_argument("--cuda_ranks")
            .default_value(1)
            .scan<'i', int>();
    program.add_argument("-s", "--store").default_value(false).implicit_value(true);
    program.add_argument("-st", "--store_stepping").default_value(false).implicit_value(true);
    program.add_argument("-sl", "--store_last").default_value(false).implicit_value(true);
    program.add_argument("--skip_history").default_value(false).implicit_value(true);
    program.add_argument("-d", "--dist")
            .default_value(std::string("cl"));
    program.add_argument("-p", "--priorities")
            .nargs(argparse::nargs_pattern::any)
            .default_value(std::vector<size_t>{})
            .scan<'i', size_t>();
    program.add_argument("--strange_mesh").default_value(false).implicit_value(true);
    program.add_argument("-m", "--mesh")
            .default_value(std::string(""));


    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    CFDArcoGlobalInit::initialize(argc, argv, program.get<bool>("skip_history"));

    bool visualize = program.get<bool>("visualize");
    bool create_plot = program.get<bool>("create_plot");

    size_t L = program.get<int>("L");
    size_t timesteps = program.get<int>("timesteps");
    size_t cuda_enable = program.get<bool>("cuda_enable");

    auto mesh = std::make_shared<Mesh2D>(L, L, 1, 1);
    if (program.get<std::string>("mesh") != "") {
        mesh = read_mesh(program.get<std::string>("mesh"));
    } else {
        mesh->init_basic_internals();
        if (program.get<bool>("strange_mesh")) {
            mesh->make_strange_internals();
        }
        mesh->compute();
    }

    DistributionStrategy dist;
    auto dist_str = program.get<std::string>("dist");
    if (dist_str == "cl") {
        dist = DistributionStrategy::Cluster;
    } else if (dist_str == "ln") {
        dist = DistributionStrategy::Linear;
    } else {
        std::cerr << "unknown dist strategy: " << dist_str << std::endl;
        std::exit(1);
    }

    auto priorities = program.get<std::vector<size_t>>("priorities");
    CFDArcoGlobalInit::make_node_distribution(mesh.get(), dist, priorities);

    if (cuda_enable && CFDArcoGlobalInit::get_rank() < program.get<int>("cuda_ranks") ) {
        CFDArcoGlobalInit::enable_cuda(mesh.get(), program.get<int>("cuda_ranks"));
    }

    Eigen::Matrix<float, -1, 1> T_initial = initial_T(mesh.get());
    auto T = Variable(mesh.get(), T_initial, boundary_T, boundary_T_cu, "T");

    std::vector<Variable*> space_vars {&T};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 1, space_vars);

    float tau = 0.3 * 0.3;

    std::vector<std::tuple<Variable*, char, Variable>> equation_system = {
            {d2t(T), '=', tau * lapl(T)},
    };

    auto equation = Equation(timesteps);

    std::vector<Variable*> all_vars {&T};

    if (program.get<bool>("store_stepping")) init_store_history_stepping({&T}, mesh.get());

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, visualize, {&T});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    if (program.get<bool>("store_last")) {
        if (cuda_enable) {
            T.current = T.current_cu.to_eigen(mesh->_num_nodes, 1);
        }
        T.history = {T.current, T.current};
    }

    if (program.get<bool>("store")) {
        if (program.get<bool>("store_stepping")) {
            finalize_history_stepping();
        } else {
            store_history({&T}, mesh.get());
        }
    }

    CFDArcoGlobalInit::finalize();
    return 0;
}