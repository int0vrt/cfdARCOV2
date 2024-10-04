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
#ifndef CFDARCO_CFDARCHO_MAIN_3D_HPP
#define CFDARCO_CFDARCHO_MAIN_3D_HPP

#include <vector>
//#include <mesh3d.hpp>
#include <filesystem>

//#include "decls.hpp"

namespace fs = std::filesystem;

enum class DistributionStrategy {
    Linear, Cluster
};

class CFDArcoGlobalInit {
public:
    static void initialize(int argc, char **argv, bool skip_history_, const fs::path &store_path = "./dumps");

    static void finalize();

//    static void make_node_distribution(Mesh3D *_mesh, DistributionStrategy distribution_strategy,
//                                       std::vector<size_t> priorities = {});
//
//    static std::vector<MatrixX4dRB> get_redistributed(const MatrixX4dRB &inst, const std::string &name);
//
//    static MatrixX4dRB recombine(const MatrixX4dRB &inst, const std::string &name);

    static inline int get_rank() { return world_rank; }

    static inline int get_size() { return world_size; }

    static void enable_cuda(int cuda_ranks, bool use_cuda, bool use_hip);

    static inline void enable_cuda() { enable_cuda(0, 1, 0); }

    static bool cuda_enabled;
    static bool hip_enabled;
    static bool skip_history;
    static bool store_stepping;
    static fs::path store_dir;

    CFDArcoGlobalInit(CFDArcoGlobalInit &other) = delete;

    void operator=(const CFDArcoGlobalInit &) = delete;


private:
    ~CFDArcoGlobalInit();

//    static std::vector<std::vector<size_t>> get_send_perspective(std::vector<size_t> &proc_node_distribution,
//                                                                 Mesh3D *mesh, size_t proc_rank);

//    static std::vector<std::vector<size_t>> node_distribution;
//    static std::vector<size_t> current_proc_node_distribution;
//    static std::vector<int> node_id_to_proc;
//    static std::vector<std::vector<size_t>> current_proc_node_receive_distribution;
//    static std::vector<std::vector<size_t>> current_proc_node_send_distribution;
//    static Mesh3D *mesh;
    static int world_size;
    static int world_rank;
};


#endif //CFDARCO_CFDARCHO_MAIN_3D_HPP
