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

// ============================================================================
// OPTIMIZED BOUNDARY CONDITIONS
// ============================================================================

// Pre-compute boundary node indices for wave equation
struct WaveBoundaryNodes {
    size_t node1_idx;
    size_t node2_idx;
    float frequency;
    
    WaveBoundaryNodes(Mesh3D *mesh, float freq = 0.2f) : frequency(freq) {
        node1_idx = mesh->square_node_coord_to_idx(mesh->_x * 0.1, mesh->_y * 0.1, mesh->_z * 0.1);
        node2_idx = mesh->square_node_coord_to_idx(mesh->_x * 0.9, mesh->_y * 0.9, mesh->_z * 0.9);
    }
};

// CPU version - time-dependent sine boundary condition
auto create_wave_boundary_condition(const WaveBoundaryNodes &boundary_nodes) {
    return [&boundary_nodes](Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
        Eigen::Matrix<float, -1, 1> ret = arr;
        float time_value = std::sin(static_cast<float>(dt_->_current_time_step_int) * boundary_nodes.frequency);
        ret(boundary_nodes.node1_idx) = time_value;
        ret(boundary_nodes.node2_idx) = time_value;
        return ret;
    };
}

// CUDA kernel for optimized boundary condition
__global__ void wave_boundary_kernel(float *arr, size_t node1_idx, size_t node2_idx, float time_value) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == node1_idx || idx == node2_idx) {
        arr[idx] = time_value;
    }
}

// CUDA version - time-dependent sine boundary condition
auto create_wave_boundary_condition_cu(const WaveBoundaryNodes &boundary_nodes) {
    return [&boundary_nodes](Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
        float time_value = std::sin(static_cast<float>(dt_->_current_time_step_int) * boundary_nodes.frequency);
        
        int blocksize = 1024;
        int nblocks = std::ceil(static_cast<float>(arr._size) / static_cast<float>(blocksize));
        wave_boundary_kernel<<<nblocks, blocksize>>>(
                arr.data.get(),
                boundary_nodes.node1_idx,
                boundary_nodes.node2_idx,
                time_value
        );
        sync_device();
        
    };
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};

    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Pre-compute boundary node information for optimized performance
    WaveBoundaryNodes boundary_nodes(mesh.get(), 0.2f);
    
    // Create boundary condition factories
    auto wave_bc = create_wave_boundary_condition(boundary_nodes);
    auto wave_bc_cu = create_wave_boundary_condition_cu(boundary_nodes);

    auto initial_zero = initial_with_val(mesh.get(), 0);
    auto u = Variable(mesh.get(), initial_zero, wave_bc, wave_bc_cu, "u");

    std::vector<Variable *> space_vars{&u};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, 0.1, space_vars);

    // Wave equation: ∂²u/∂t² = c²∇²u
    // where c = 0.3 is the wave speed
    float c = 0.3;

    EquationTemplate equation_system = {
        {d2t(u), '=', c * c * lapl(u), true},
    };

    std::vector<Variable *> all_vars{&u};
    auto equation = Equation(timesteps);
    initializer.init_store(all_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, all_vars);
    auto end = std::chrono::steady_clock::now();
    
    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Wave Equation Simulation Completed!" << std::endl;
        std::cout << "Wave speed (c) = " << c << std::endl;
        std::cout << "Boundary condition: sin(ωt) at two corner points" << std::endl;
        std::cout << "Frequency (ω) = " << boundary_nodes.frequency << std::endl;
        std::cout << "Expected behavior: Wave propagation from source points" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
}