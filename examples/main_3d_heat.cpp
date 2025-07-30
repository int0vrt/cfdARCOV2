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

// Pre-compute boundary source for heat equation (Dirichlet BC)
Eigen::Matrix<float, -1, 1> compute_heat_boundary_source(Mesh3D *mesh) {
    Eigen::Matrix<float, -1, 1> boundary_source = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    
    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        if (node.z() < 2 * mesh->_dz) {
            boundary_source(i) = 100.0f; // Hot boundary at bottom
        }
    }
    return boundary_source;
}

// ============================================================================
// INITIAL CONDITIONS
// ============================================================================

Eigen::Matrix<float, -1, 1> initial_T(Mesh3D *mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    ret.setConstant(0);

    size_t i = 0;
    for (auto &node: mesh->_nodes) {
        if (node.z() < 2 * mesh->_dz) {
            ret(i) = 100;
        } else {
            ret(i) = 0;
        }
        ++i;
    }
    return ret;
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Pre-compute boundary source for optimized performance
    auto boundary_source_ = compute_heat_boundary_source(mesh.get());
    auto boundary_source = Variable(mesh.get(), boundary_source_, boundary_none, boundary_none_cu, "boundary_source");

    Eigen::Matrix<float, -1, 1> T_initial = initial_T(mesh.get());
    auto T = Variable(mesh.get(), T_initial, zero_gradient_bc, zero_gradient_bc_cu, "T");

    std::vector<Variable *> space_vars{&T};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    // Heat equation: ∂T/∂t = α∇²T
    // where α = 3.0 is the thermal diffusivity
    EquationTemplate equation_system = {
        {d1t(T), '=', 3.0 * lapl(T) + boundary_source, true},
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(space_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();
    
    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Heat Equation Simulation Completed!" << std::endl;
        std::cout << "Thermal diffusivity (α) = 3.0" << std::endl;
        std::cout << "Boundary condition: T = 100°C at bottom, T = 0°C elsewhere" << std::endl;
        std::cout << "Expected behavior: Heat diffusion from hot bottom boundary" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
}