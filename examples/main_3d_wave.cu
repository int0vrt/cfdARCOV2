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
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "utils3d.hpp"

Eigen::Matrix<float, -1, 1> boundary_sine(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) {

    Eigen::Matrix<float, -1, 1> ret{arr};
    ret[mesh->square_node_coord_to_idx(mesh->_x * 0.1, mesh->_y * 0.1, mesh->_z * 0.1)] = std::sin(static_cast<float>(dt_->_current_time_step_int) * 0.2);
    ret[mesh->square_node_coord_to_idx(mesh->_x * 0.9, mesh->_y * 0.9, mesh->_z * 0.9)] = std::sin(static_cast<float>(dt_->_current_time_step_int) * 0.2);

    return ret;
}

__global__ void boundary_sine_k(float *a, size_t dt_itr, size_t n_1, size_t n_2) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == n_1 || idx == n_2) {
        a[idx] = std::sin(static_cast<float>(dt_itr) * 0.2);
    }
}

CudaDataMatrixD boundary_sine_cu(Mesh3D* mesh, CudaDataMatrixD& arr, const DT* dt_) {
    CudaDataMatrixD arr_n{arr};

    size_t n_1 = mesh->square_node_coord_to_idx(mesh->_x * 0.1, mesh->_y * 0.1, mesh->_z * 0.1);
    size_t n_2 = mesh->square_node_coord_to_idx(mesh->_x * 0.9, mesh->_y * 0.9, mesh->_z * 0.9);

    int blocksize = 1024;
    int nblocks = std::ceil(static_cast<float>(arr_n._size) / static_cast<float>(blocksize));
    boundary_sine_k<<<nblocks, blocksize>>>(arr_n.data.get(), dt_->_current_time_step_int, n_1, n_2);
    sync_device();

    return arr_n;
}


int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};

    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto initial_zero = initial_with_val(mesh.get(), 0);
//    auto u = Variable(mesh.get(), initial_zero, boundary_sine, boundary_sine_cu, "u");
    auto u = Variable(mesh.get(), initial_zero, boundary_sine, "u");

    std::vector<Variable*> space_vars {&u};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.1, space_vars);

    float c = 0.3;

    EquationTemplate equation_system = {
            {d2t(u), '=', c * c * (d2dx(u) + d2dy(u)), true},
    };

    std::vector<Variable*> all_vars {&u};
    auto equation = Equation(timesteps);
    initializer.init_store(all_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, all_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}