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

// CPU version - zero gradient boundary condition
Eigen::Matrix<float, -1, 1> zero_gradient_bc(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
    return arr.cwiseProduct(mesh->_node_is_boundary_reverse);
}

// CUDA version - zero gradient boundary condition
CudaDataMatrixD zero_gradient_bc_cu(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
    auto *cuda_mesh = dynamic_cast<CudaMesh3D *>(mesh);
    return arr * cuda_mesh->_node_is_boundary_reverse_cu;
}

// ============================================================================
// INITIAL CONDITIONS
// ============================================================================

// Initial condition: 3D Gaussian pulse for all velocity components
Eigen::Matrix<float, -1, 1> initial_u(Mesh3D *mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float center_x = 0.3 * mesh->_lx;
    float center_y = 0.5 * mesh->_ly;
    float center_z = 0.5 * mesh->_lz;
    float sigma = 0.1 * std::min({mesh->_lx, mesh->_ly, mesh->_lz});

    for (auto &node: mesh->_nodes) {
        float x = node.x();
        float y = node.y();
        float z = node.z();
        
        float r_squared = (x - center_x) * (x - center_x) + 
                         (y - center_y) * (y - center_y) + 
                         (z - center_z) * (z - center_z);
        
        ret(i) = std::exp(-r_squared / (2.0 * sigma * sigma));
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_v(Mesh3D *mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float center_x = 0.5 * mesh->_lx;
    float center_y = 0.3 * mesh->_ly;
    float center_z = 0.5 * mesh->_lz;
    float sigma = 0.1 * std::min({mesh->_lx, mesh->_ly, mesh->_lz});

    for (auto &node: mesh->_nodes) {
        float x = node.x();
        float y = node.y();
        float z = node.z();
        
        float r_squared = (x - center_x) * (x - center_x) + 
                         (y - center_y) * (y - center_y) + 
                         (z - center_z) * (z - center_z);
        
        ret(i) = 0.5 * std::exp(-r_squared / (2.0 * sigma * sigma));
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_w(Mesh3D *mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float center_x = 0.5 * mesh->_lx;
    float center_y = 0.5 * mesh->_ly;
    float center_z = 0.3 * mesh->_lz;
    float sigma = 0.1 * std::min({mesh->_lx, mesh->_ly, mesh->_lz});

    for (auto &node: mesh->_nodes) {
        float x = node.x();
        float y = node.y();
        float z = node.z();
        
        float r_squared = (x - center_x) * (x - center_x) + 
                         (y - center_y) * (y - center_y) + 
                         (z - center_z) * (z - center_z);
        
        ret(i) = 0.3 * std::exp(-r_squared / (2.0 * sigma * sigma));
        ++i;
    }
    return ret;
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Initialize velocity components
    auto u_initial = initial_u(mesh.get());
    auto v_initial = initial_v(mesh.get());
    auto w_initial = initial_w(mesh.get());
    
    // Create variables with optimized boundary conditions
    auto u = Variable(mesh.get(), u_initial, zero_gradient_bc, zero_gradient_bc_cu, "u");
    auto v = Variable(mesh.get(), v_initial, zero_gradient_bc, zero_gradient_bc_cu, "v");
    auto w = Variable(mesh.get(), w_initial, zero_gradient_bc, zero_gradient_bc_cu, "w");

    // Create temporary variables for the solution with optimized boundary conditions
    auto u_tmp = Variable(mesh.get(), u_initial, zero_gradient_bc, zero_gradient_bc_cu, "u_tmp");
    auto v_tmp = Variable(mesh.get(), v_initial, zero_gradient_bc, zero_gradient_bc_cu, "v_tmp");
    auto w_tmp = Variable(mesh.get(), w_initial, zero_gradient_bc, zero_gradient_bc_cu, "w_tmp");

    std::vector<Variable *> space_vars{&u, &v, &w};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu,
                 initializer.dt, space_vars);

    // Burgers' equation parameters
    float nu = 0.1;  // Viscosity coefficient
    float dissip = 1;  // Artificial dissipation coefficient

    // Full 3D Burgers' equations using kernel building:
    // ∂u/∂t + u·∂u/∂x + v·∂u/∂y + w·∂u/∂z = ν·(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    // ∂v/∂t + u·∂v/∂x + v·∂v/∂y + w·∂v/∂z = ν·(∂²v/∂x² + ∂²v/∂y² + ∂²v/∂z²)
    // ∂w/∂t + u·∂w/∂x + v·∂w/∂y + w·∂w/∂z = ν·(∂²w/∂x² + ∂²w/∂y² + ∂²w/∂z²)
    
    EquationTemplate equation_system = {
        // FUSED KERNEL: All velocity components computed together
        // u-component: ∂u/∂t + u·∂u/∂x + v·∂u/∂y + w·∂u/∂z = ν·∇²u
        {&u_tmp, '=', -(u * d1dx(u) + v * d1dy(u) + w * d1dz(u)) + nu * lapl(u), false},

        // v-component: ∂v/∂t + u·∂v/∂x + v·∂v/∂y + w·∂v/∂z = ν·∇²v
        {&v_tmp, '=', -(u * d1dx(v) + v * d1dy(v) + w * d1dz(v)) + nu * lapl(v), false},

        // w-component: ∂w/∂t + u·∂w/∂x + v·∂w/∂y + w·∂w/∂z = ν·∇²w
        {&w_tmp, '=', -(u * d1dx(w) + v * d1dy(w) + w * d1dz(w)) + nu * lapl(w), false},

            // Time integration using explicit Euler with artificial dissipation
        {d1t(u), '=', u_tmp - (stabx(u) + staby(u) + stabz(u)) * dissip, false},
        {d1t(v), '=', v_tmp - (stabx(v) + staby(v) + stabz(v)) * dissip, false},
        {d1t(w), '=', w_tmp - (stabx(w) + staby(w) + stabz(w)) * dissip, false}
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{&u, &v, &w, &u_tmp, &v_tmp, &w_tmp};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Burgers' Equation Simulation Completed!" << std::endl;
        std::cout << "Viscosity coefficient (ν) = " << nu << std::endl;
        std::cout << "Artificial dissipation = " << dissip << std::endl;
        std::cout << "Expected behavior: Gaussian pulses will diffuse and advect" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
} 