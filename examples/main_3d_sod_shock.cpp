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

// CPU version - no boundary condition (for internal variables)
Eigen::Matrix<float, -1, 1> no_bc(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
    return arr;
}

// CUDA version - no boundary condition (for internal variables)
CudaDataMatrixD no_bc_cu(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
    return arr;
}

// ============================================================================
// OPTIMIZED INITIAL CONDITIONS
// ============================================================================

// Pre-compute initial conditions for SOD shock tube
struct SodInitialConditions {
    Eigen::Matrix<float, -1, 1> rho_init;
    Eigen::Matrix<float, -1, 1> p_init;
    Eigen::Matrix<float, -1, 1> u_init;
    
    SodInitialConditions(Mesh3D *mesh) {
        rho_init = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
        p_init = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
        u_init = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
        
        float x_limit_lower = 0.5 * mesh->_lx;
        
        for (int i = 0; i < mesh->_num_nodes; ++i) {
            auto& node = mesh->_nodes[i];
            
            if (x_limit_lower < node.x()) {
                rho_init(i) = 1.0f;    // High density region
                p_init(i) = 1.0f;      // High pressure region
            } else {
                rho_init(i) = 0.125f;  // Low density region
                p_init(i) = 0.1f;      // Low pressure region
            }
            u_init(i) = 0.0f;          // Zero initial velocity
        }
    }
};

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Pre-compute initial conditions for optimized performance
    SodInitialConditions init_conditions(mesh.get());
    
    // Create variables with optimized boundary conditions
    auto rho = Variable(mesh.get(), init_conditions.rho_init, zero_gradient_bc, zero_gradient_bc_cu, "rho");
    auto u = Variable(mesh.get(), init_conditions.u_init, zero_gradient_bc, zero_gradient_bc_cu, "u");
    auto p = Variable(mesh.get(), init_conditions.p_init, zero_gradient_bc, zero_gradient_bc_cu, "p");

    // Compute derived variables
    Eigen::Matrix<float, -1, 1> mass_initial = rho.current.array() * mesh->_volumes.array();
    auto mass = Variable(mesh.get(), mass_initial, zero_gradient_bc, zero_gradient_bc_cu, "mass");

    Eigen::Matrix<float, -1, 1> rho_u_initial = rho.current.array() * u.current.array() * mesh->_volumes.array();
    auto rho_u = Variable(mesh.get(), rho_u_initial, zero_gradient_bc, zero_gradient_bc_cu, "rho_u");

    float gamma = 1.4f;  // Specific heat ratio for air

    auto E = p / (gamma - 1) + 0.5 * rho * ((u * u));
    Eigen::Matrix<float, -1, 1> E_initial =
            (p.current.array() / (gamma - 1) + 0.5 * rho.current.array() * (u.current.array() * u.current.array())) *
            mesh->_volumes.array();
    auto rho_e = Variable(mesh.get(), E_initial, zero_gradient_bc, zero_gradient_bc_cu, "rho_e");

    // Temporary variables for internal computations (no boundary conditions needed)
    auto mass_tmp = Variable(mesh.get(), mass_initial, no_bc, no_bc_cu, "mass_tmp");
    auto rho_u_tmp = Variable(mesh.get(), rho_u_initial, no_bc, no_bc_cu, "rho_u_tmp");
    auto rho_e_tmp = Variable(mesh.get(), E_initial, no_bc, no_bc_cu, "rho_e_tmp");

    std::vector<Variable *> space_vars{&u, &p, &rho};
    auto dt = DT(mesh.get(), UpdatePolicies::CourantFriedrichsLewy1D, UpdatePolicies::CourantFriedrichsLewy1DCu,
                 initializer.dt, space_vars);

    auto volumes_var = Variable(mesh.get(), mesh->_volumes, no_bc, no_bc_cu, "volumes_var");

    float dissip = 2.0f;  // Artificial dissipation coefficient
    
    // SOD shock tube equations using kernel building
    EquationTemplate equation_system = {
        // FUSED KERNEL: Primitive variable reconstruction
        NEntryVar{
            {&rho, '=', mass / volumes_var, true},
            {&u, '=', rho_u / rho / volumes_var, true},
            {&p, '=', (rho_e / volumes_var - 0.5 * rho * (u * u)) * (gamma - 1), true}
        }.restruct(),

        // FUSED KERNEL: Predictor step (first-order accurate)
        NEntryVar{
            {&rho, '=', rho - 0.5 * dt * (u * d1dx(rho) + rho * d1dx(u)), true},
            {&u, '=', u - 0.5 * dt * (u * d1dx(u) + (1 / rho) * d1dx(p)), true},
            {&p, '=', p - 0.5 * dt * (gamma * p * (d1dx(u)) + u * d1dx(p)), true}
        }.restruct(),

        // FUSED KERNEL: Flux computation
        NEntryVar{
            {&mass_tmp, '=', 0 * mass_tmp - (d1dx(rho * u)), true},
            {&rho_u_tmp, '=', 0 * rho_u_tmp - (d1dx(rho * u * u + p)), true},
            {&rho_e_tmp, '=', 0 * rho_e_tmp - (d1dx((E + p) * u)), true}
        }.restruct(),

        // FUSED KERNEL: Corrector step with artificial dissipation
        NEntryVar{
            {d1t(mass), '=', mass_tmp - stabx(rho) * dissip, false},
            {d1t(rho_u), '=', rho_u_tmp - stabx(rho * u) * dissip, false},
            {d1t(rho_e), '=', rho_e_tmp - stabx(E) * dissip, false}
        }.restruct()
    };

    auto equation = Equation(timesteps);
    initializer.init_store({&rho, &u, &p});

    std::vector<Variable *> all_vars{&rho, &u, &p, &mass, &rho_u, &rho_e, &mass_tmp, &rho_u_tmp, &rho_e_tmp, &volumes_var};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&rho, &u, &p});
    auto end = std::chrono::steady_clock::now();
    
    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\nSOD Shock Tube Simulation Completed!" << std::endl;
        std::cout << "Specific heat ratio (γ) = " << gamma << std::endl;
        std::cout << "Artificial dissipation = " << dissip << std::endl;
        std::cout << "Boundary condition: Zero gradient (outflow)" << std::endl;
        std::cout << "Expected features: Shock wave, contact discontinuity, rarefaction fan" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
}