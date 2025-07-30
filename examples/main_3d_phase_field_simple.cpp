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
#include "operators.hpp"
#include "equation.hpp"
#include "utils3d.hpp"
#include "phase_field_params.hpp"

// ============================================================================
// OPTIMIZED BOUNDARY CONDITIONS
// ============================================================================

// Pre-computed boundary masks for optimized performance
struct OptimizedBoundaryMasks {
    Eigen::Matrix<float, -1, 1> source_mask;
    Eigen::Matrix<float, -1, 1> zero_gradient_mask;
    
    OptimizedBoundaryMasks(Mesh3D *mesh) {
        source_mask = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
        zero_gradient_mask = mesh->_node_is_boundary_reverse;
        
        // Pre-compute source mask for center region
        for (int i = 0; i < mesh->_num_nodes; ++i) {
            auto& node = mesh->_nodes[i];
            float x = node.x() / mesh->_lx;
            float y = node.y() / mesh->_ly;
            float z = node.z() / mesh->_lz;
            
            float dx = x - 0.5f;
            float dy = y - 0.5f;
            float dz = z - 0.5f;
            float r = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (r < PhaseFieldParams::SimpleParams::source_radius) {
                source_mask(i) = 1.0f;  // Source region
            }
        }
    }
};

// CPU version - optimized constant source boundary condition
Eigen::Matrix<float, -1, 1> optimized_source_bc(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
    static OptimizedBoundaryMasks masks(mesh);
    return arr.cwiseProduct(masks.zero_gradient_mask) + masks.source_mask;
}

// CUDA version - optimized constant source boundary condition
CudaDataMatrixD optimized_source_bc_cu(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
    static OptimizedBoundaryMasks masks(mesh);
    auto cuda_masks = CudaDataMatrixD::from_eigen(masks.zero_gradient_mask);
    auto cuda_source = CudaDataMatrixD::from_eigen(masks.source_mask);
    return arr * cuda_masks + cuda_source;
}

// ============================================================================
// SIMPLE INITIAL CONDITIONS
// ============================================================================

// Simple initial conditions for testing
Eigen::Matrix<float, -1, 1> get_simple_initial_condition(Mesh3D *mesh) {
    using Params = PhaseFieldParams::SimpleParams;
    
    Eigen::Matrix<float, -1, 1> phi_init = Eigen::Matrix<float, -1, 1>::Constant(mesh->_num_nodes, -1.0f);
    
    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float x = node.x() / mesh->_lx;
        float y = node.y() / mesh->_ly;
        float z = node.z() / mesh->_lz;
        
        // Simple spherical seed at center
        float dx = x - 0.5f;
        float dy = y - 0.5f;
        float dz = z - 0.5f;
        float r = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (r < Params::seed_radius) {
            phi_init(i) = 1.0f;  // Solid phase
        } else if (r < Params::transition_radius) {
            // Smooth transition
            float transition = (Params::transition_radius - r) / (Params::transition_radius - Params::seed_radius);
            phi_init(i) = 1.0f - 2.0f * transition;
        }
        // else: liquid phase (-1.0f)
    }
    
    return phi_init;
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Use centralized parameters
    using Params = PhaseFieldParams::SimpleParams;

    // Simple initial condition
    auto phi_init = get_simple_initial_condition(mesh.get());
    
    // Create variable with optimized boundary conditions
    auto phi = Variable(mesh.get(), phi_init, optimized_source_bc, optimized_source_bc_cu, "phi");

    // Time step variables
    std::vector<Variable *> space_vars{&phi};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    // Simple phase field equation with constant source: ∂φ/∂t = M∇²φ - M/ε²(φ³ - φ) + S
    EquationTemplate equation_system = {
        NEntryVar{
            {&phi, '=', phi + dt * (Params::M * lapl(phi) - Params::M / (Params::epsilon * Params::epsilon) * (phi * phi * phi - phi) + Params::source_strength), true}
        }.restruct()
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{&phi};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\nSimple Phase Field Test with Optimized Constant Source Completed!" << std::endl;
        std::cout << "Interface thickness (ε): " << Params::epsilon << std::endl;
        std::cout << "Mobility (M): " << Params::M << std::endl;
        std::cout << "Source strength (S): " << Params::source_strength << std::endl;
        std::cout << "Time step: " << initializer.dt << std::endl;
        std::cout << "Grid resolution: " << mesh->_x << "x" << mesh->_y << "x" << mesh->_z << std::endl;
        std::cout << "Boundary condition: Optimized constant source at center" << std::endl;
        std::cout << "Expected: Crystal should grow and remain stable with constant source" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
} 