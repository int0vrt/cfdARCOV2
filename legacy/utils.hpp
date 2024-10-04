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
#ifndef CFDARCO_UTILS_HPP
#define CFDARCO_UTILS_HPP

#include "argparse/argparse.hpp"
#include "cfdarcho_main.hpp"
#include "io_operators.hpp"

inline argparse::ArgumentParser parse_args_base(int argc, char **argv) {
    argparse::ArgumentParser program("cfdARCO");
    program.add_argument("-v", "--visualize").default_value(false).implicit_value(true);
    program.add_argument("--create_plot").default_value(false).implicit_value(true);
    program.add_argument("-Lx")
            .help("square size")
            .default_value(200)
            .scan<'i', int>();
    program.add_argument("-Ly")
            .help("square size")
            .default_value(200)
            .scan<'i', int>();
    program.add_argument("-dx")
            .help("square size")
            .default_value(1.0)
            .scan<'g', double>();
    program.add_argument("-dy")
            .help("square size")
            .default_value(1.0)
            .scan<'g', double>();
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

    return program;
}


class SingleLibInitializer{
public:
    bool visualize;
    bool create_plot;
    size_t Lx;
    size_t Ly;
    double dx;
    double dy;
    int timesteps;
    bool cuda_enable;
    bool store_stepping;
    bool store_last;
    bool store;
    std::vector<size_t> priorities;
    std::shared_ptr<Mesh2D> mesh;
    std::vector<Variable*> store_vars;


    SingleLibInitializer(int argc, char **argv) {
        auto program = parse_args_base(argc, argv);

        CFDArcoGlobalInit::initialize(argc, argv, program.get<bool>("skip_history"));

        visualize = program.get<bool>("visualize");
        create_plot = program.get<bool>("create_plot");

        Lx = program.get<int>("Lx");
        Ly = program.get<int>("Ly");
        dx = program.get<double>("dx");
        dy = program.get<double>("dy");
        timesteps = program.get<int>("timesteps");
        cuda_enable = program.get<bool>("cuda_enable");
        store_stepping = program.get<bool>("store_stepping");
        store_last = program.get<bool>("store_last");
        store = program.get<bool>("store");

        mesh = std::make_shared<Mesh2D>(Lx, Ly, dx, dy);
        if (!program.get<std::string>("mesh").empty()) {
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

        priorities = program.get<std::vector<size_t>>("priorities");
        CFDArcoGlobalInit::make_node_distribution(mesh.get(), dist, priorities);

        if (cuda_enable && CFDArcoGlobalInit::get_rank() < program.get<int>("cuda_ranks") ) {
            CFDArcoGlobalInit::enable_cuda(mesh.get(), program.get<int>("cuda_ranks"));
        }

    }

    void init_store(const std::vector<Variable*>& vars_to_store) {
        store_vars = vars_to_store;
        if (store_stepping) init_store_history_stepping(vars_to_store, mesh.get());
    }

    void finalize() {
        for (const auto& var : store_vars) {
            if (store_last) {
                if (cuda_enable) {
                    var->current = var->current_cu.to_eigen(mesh->_num_nodes, 1);
                }
                var->history = {var->current, var->current};
            }
        }

        if (store) {
            if (store_stepping) {
                finalize_history_stepping();
            } else {
                store_history(store_vars, mesh.get());
            }
        }

        CFDArcoGlobalInit::finalize();
    }
};

inline void sett(Eigen::VectorXd& v, int rows, int cols, int x, int y, double val) {
    const size_t idx = x * rows + y;
    v[idx] = val;
}


#endif //CFDARCO_UTILS_HPP
