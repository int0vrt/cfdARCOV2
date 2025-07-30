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

// ============================================================================
// INITIAL CONDITIONS
// ============================================================================

// Pre-compute initial conditions with perturbations for Turing patterns
struct TuringInitialConditions {
    Eigen::Matrix<float, -1, 1> u_init;
    Eigen::Matrix<float, -1, 1> v_init;
    
    TuringInitialConditions(Mesh3D *mesh, float A, float B) {
        u_init = Eigen::Matrix<float, -1, 1>::Constant(mesh->_num_nodes, A);
        v_init = Eigen::Matrix<float, -1, 1>::Constant(mesh->_num_nodes, B / A);
        
        // Add small random perturbations to trigger pattern formation
        std::srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < mesh->_num_nodes; ++i) {
            auto& node = mesh->_nodes[i];
            float x = node.x() / mesh->_lx;
            float y = node.y() / mesh->_ly;
            float z = node.z() / mesh->_lz;
            
            // Add small sinusoidal perturbations
            float perturbation = 0.1f * (std::sin(4.0f * M_PI * x) * std::cos(3.0f * M_PI * y) * std::sin(2.0f * M_PI * z));
            u_init(i) += perturbation;
            v_init(i) += perturbation;
            
            // Add small random noise
            float noise = 1.0f * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f);
            u_init(i) += noise;
            v_init(i) += noise;
        }
    }
};

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Turing pattern parameters (classic Brusselator model)
    const float Du = 0.1f;   // Diffusion coefficient for u
    const float Dv = 0.05f;  // Diffusion coefficient for v
    const float A = 4.5f;    // Brusselator parameter A
    const float B = 7.0f;    // Brusselator parameter B

    // Pre-compute initial conditions for optimized performance
    TuringInitialConditions init_conditions(mesh.get(), A, B);
    
    // Create variables with optimized boundary conditions
    auto u = Variable(mesh.get(), init_conditions.u_init, zero_gradient_bc, zero_gradient_bc_cu, "u");
    auto v = Variable(mesh.get(), init_conditions.v_init, zero_gradient_bc, zero_gradient_bc_cu, "v");

    // Time step variables
    std::vector<Variable *> space_vars{&u, &v};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    // Brusselator reaction-diffusion equations using kernel building:
    // ∂u/∂t = Du∇²u + A - (B+1)u + u²v
    // ∂v/∂t = Dv∇²v + Bu - u²v
    EquationTemplate equation_system = {
        // FUSED KERNEL: Both u and v components computed together
        NEntryVar{
                // u-component: ∂u/∂t = Du∇²u + A - (B+1)u + u²v
                {&u, '=', u + dt * (Du * (d2dx(u) + d2dy(u)) + A - (B + 1.0f) * u + u * u * v), true},
        }.restruct(),
        NEntryVar{
            // v-component: ∂v/∂t = Dv∇²v + Bu - u²v
            {&v, '=', v + dt * (Dv * (d2dx(v) + d2dy(v)) + B * u - u * u * v), true}
        }.restruct()
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{&u, &v};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\nTuring Pattern Formation Simulation Completed!" << std::endl;
        std::cout << "Brusselator parameters: A = " << A << ", B = " << B << std::endl;
        std::cout << "Diffusion coefficients: Du = " << Du << ", Dv = " << Dv << std::endl;
        std::cout << "Boundary condition: Periodic (zero gradient)" << std::endl;
        std::cout << "Expected features: Self-organizing patterns, spots, stripes, or labyrinthine structures" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
} 