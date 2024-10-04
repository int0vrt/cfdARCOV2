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
//
// Created by yevhen on 7/17/24.
//

#ifndef CFDARCO_UTILS3D_HPP
#define CFDARCO_UTILS3D_HPP

#include "argparse/argparse.hpp"
#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "cfdarcho_main_3d.hpp"
#include "io_operators_tmpl.hpp"

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
    program.add_argument("-Lz")
            .help("square size")
            .default_value(200)
            .scan<'i', int>();
    program.add_argument("-dx")
            .help("square size")
            .default_value(1.0)
            .scan<'g', float>();
    program.add_argument("-dy")
            .help("square size")
            .default_value(1.0)
            .scan<'g', float>();
    program.add_argument("-dz")
            .help("square size")
            .default_value(1.0)
            .scan<'g', float>();
    program.add_argument("-t", "--timesteps")
            .help("timesteps")
            .default_value(1000)
            .scan<'i', int>();
    program.add_argument("-dt")
            .help("dt")
            .default_value(0.005)
            .scan<'g', float>();
    program.add_argument("-c", "--cuda_enable").default_value(false).implicit_value(true);
    program.add_argument("-hip", "--hip_enable").default_value(false).implicit_value(true);
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
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    return program;
}


class SingleLibInitializer3D {
public:
    bool visualize;
    bool create_plot;
    size_t Lx;
    size_t Ly;
    size_t Lz;
    float dx;
    float dy;
    float dz;
    int timesteps;
    float dt;
    bool cuda_enable;
    bool hip_enable;
    bool store_stepping;
    bool store_last;
    bool store;
    std::vector<size_t> priorities;
    std::shared_ptr<Mesh3D> mesh;
    std::vector<Variable *> store_vars;


    SingleLibInitializer3D(int argc, char **argv) {
        auto program = parse_args_base(argc, argv);

//        CFDArcoGlobalInit::initialize(argc, argv, program.get<bool>("skip_history"));
        CFDArcoGlobalInit::initialize(argc, argv, !(program.get<bool>("store")));

        visualize = program.get<bool>("visualize");
        create_plot = program.get<bool>("create_plot");

        Lx = program.get<int>("Lx");
        Ly = program.get<int>("Ly");
        Lz = program.get<int>("Lz");
        dx = program.get<float>("dx");
        dy = program.get<float>("dy");
        dz = program.get<float>("dz");
        timesteps = program.get<int>("timesteps");
        dt = program.get<float>("dt");
        cuda_enable = program.get<bool>("cuda_enable");
        hip_enable = program.get<bool>("hip_enable");
        store_stepping = program.get<bool>("store_stepping");
        store_last = program.get<bool>("store_last");
        store = program.get<bool>("store");

        if (cuda_enable && hip_enable) {
            throw std::runtime_error{"Can`t run HIP and CUDA simultaneously"};
        }

        if (!cuda_enable && !hip_enable) {
            mesh = std::make_shared<Mesh3D>(Lx, Ly, Lz, dx, dy, dz);
        } else {
            mesh = std::make_shared<CudaMesh3D>(Lx, Ly, Lz, dx, dy, dz);
        }
        if (!program.get<std::string>("mesh").empty()) {
//            mesh = read_mesh(program.get<std::string>("mesh"));
        } else {
            mesh->init_basic_internals();
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
//        CFDArcoGlobalInit::make_node_distribution(mesh.get(), dist, priorities);
        if ((cuda_enable || hip_enable) && CFDArcoGlobalInit::get_rank() < program.get<int>("cuda_ranks") ) {
            CFDArcoGlobalInit::enable_cuda(program.get<int>("cuda_ranks"), cuda_enable, hip_enable);
            dynamic_cast<CudaMesh3D*>(mesh.get())->host_to_device();
        }

    }

    void init_store(const std::vector<Variable *> &vars_to_store) {
        store_vars = vars_to_store;
//        if (store_stepping) init_store_history_stepping(vars_to_store, mesh.get());
    }

    void finalize() {
        for (const auto &var: store_vars) {
//            if (store_last) {
//                if (cuda_enable) {
//                    var->current = var->current_cu.to_eigen(mesh->_num_nodes, 1);
//                }
//                var->history = {var->current, var->current};
//            }
        }

        if (store) {
            if (store_stepping) {
//                finalize_history_stepping();
            } else {
                store_history(store_vars, mesh.get());
            }
        }

        CFDArcoGlobalInit::finalize();
    }
};


inline void sett(Eigen::Matrix<float, -1, 1> &v, int rows, int cols, int x, int y, int z, float val) {
    const size_t idx = x * rows * cols + y * rows + z;
    v[idx] = val;
}

template<typename MeshClass>
inline Eigen::Matrix<float, -1, 1> initial_with_val(MeshClass *mesh, float val) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;
    for (auto &node: mesh->_nodes) {
        ret(i) = val;
        ++i;
    }
    return ret;
}


#endif //CFDARCO_UTILS3D_HPP
