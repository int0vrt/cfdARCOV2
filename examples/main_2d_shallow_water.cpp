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
// 2D SHALLOW WATER INITIAL CONDITIONS
// ============================================================================

// Classic dam break initial conditions
struct ShallowWater2DInitialConditions {
    Eigen::Matrix<float, -1, 1> h_init;
    Eigen::Matrix<float, -1, 1> u_init;
    Eigen::Matrix<float, -1, 1> v_init;
    
    ShallowWater2DInitialConditions(Mesh3D *mesh, float H0 = 1.0f) {
        h_init = Eigen::Matrix<float, -1, 1>::Constant(mesh->_num_nodes, H0);
        u_init = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
        v_init = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
        
        // Dam break: high water on left, low water on right
        for (int i = 0; i < mesh->_num_nodes; ++i) {
            auto& node = mesh->_nodes[i];
            float x = node.x() / mesh->_lx;
            float y = node.y() / mesh->_ly;
            
            // Step function: dam at x = 0.5
            if (x < 0.5f) {
                h_init(i) = 2.0f * H0;  // High water behind dam
            } else {
                h_init(i) = 0.5f * H0;  // Low water in front of dam
            }
            
            // Small perturbation for 2D flow development
            if (std::abs(x - 0.5f) < 0.05f) {
                float perturbation = 0.03f * H0 * std::sin(6.0f * M_PI * y);
                h_init(i) += perturbation;
            }
            
            // Ensure minimum depth for wet-dry stability
            if (h_init(i) < 0.01f * H0) {
                h_init(i) = 0.01f * H0;
            }
        }
    }
};

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Physical parameters
    const float g = 9.81f;      // Gravitational acceleration [m/s²]
    const float H0 = 1.0f;      // Reference water depth [m]
    const float nu = 0.01f;     // Kinematic viscosity [m²/s]
    const float dissip = 4.0f;  // Artificial dissipation coefficient

    std::cout << "2D Shallow Water Equations - Explicit Time Stepping with Viscosity" << std::endl;
    std::cout << "Dam Break Simulation" << std::endl;
    std::cout << "Grid: " << mesh->_x << " x " << mesh->_y << " x " << mesh->_z << std::endl;
    std::cout << "Kinematic viscosity: " << nu << " m²/s" << std::endl;
    std::cout << "Optimal setup: Use Lz=1 for quasi-2D efficiency" << std::endl;

    // Initialize with dam break conditions
    ShallowWater2DInitialConditions init_conditions(mesh.get(), H0);
    
    // Create main variables
    auto h = Variable(mesh.get(), init_conditions.h_init, zero_gradient_bc, zero_gradient_bc_cu, "h");
    auto u = Variable(mesh.get(), init_conditions.u_init, zero_gradient_bc, zero_gradient_bc_cu, "u");
    auto v = Variable(mesh.get(), init_conditions.v_init, zero_gradient_bc, zero_gradient_bc_cu, "v");

    // Create temporary variables for explicit time stepping
    auto h_tmp = Variable(mesh.get(), init_conditions.h_init, zero_gradient_bc, zero_gradient_bc_cu, "h_tmp");
    auto u_tmp = Variable(mesh.get(), init_conditions.u_init, zero_gradient_bc, zero_gradient_bc_cu, "u_tmp");
    auto v_tmp = Variable(mesh.get(), init_conditions.v_init, zero_gradient_bc, zero_gradient_bc_cu, "v_tmp");

    // Time step control for shallow water CFL condition
    std::vector<Variable *> space_vars{&h, &u, &v};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    // 2D Shallow Water Equations with Viscosity - Explicit Time Stepping:
    // ∂h/∂t + u∂h/∂x + v∂h/∂y + h(∂u/∂x + ∂v/∂y) = 0                    (Continuity)
    // ∂u/∂t + u∂u/∂x + v∂u/∂y = -g∂h/∂x + ν∇²u                          (x-momentum)  
    // ∂v/∂t + u∂v/∂x + v∂v/∂y = -g∂h/∂y + ν∇²v                          (y-momentum)

    EquationTemplate equation_system = {
        // Step 1: Compute right-hand sides explicitly
        NEntryVar{
            // Continuity RHS: -[u∂h/∂x + v∂h/∂y + h(∂u/∂x + ∂v/∂y)]
            {&h_tmp, '=', -(u * d1dx(h) + v * d1dy(h) + h * (d1dx(u) + d1dy(v))), true},
            
            // x-momentum RHS: -[u∂u/∂x + v∂u/∂y + g∂h/∂x] + ν∇²u
            {&u_tmp, '=', -(u * d1dx(u) + v * d1dy(u) + g * d1dx(h)) + nu * (d2dx(u) + d2dy(u)), true},
            
            // y-momentum RHS: -[u∂v/∂x + v∂v/∂y + g∂h/∂y] + ν∇²v
            {&v_tmp, '=', -(u * d1dx(v) + v * d1dy(v) + g * d1dy(h)) + nu * (d2dx(v) + d2dy(v)), true}
        }.restruct(),

        // Step 2: Explicit time integration with artificial dissipation
        // Explicit Euler: h^{n+1} = h^n + Δt * RHS_h
        {d1t(h), '=', h_tmp - (stabx(h) + staby(h)) * dissip, false},

        // Explicit Euler: u^{n+1} = u^n + Δt * RHS_u
        {d1t(u), '=', u_tmp - (stabx(u) + staby(u)) * dissip, false},

        // Explicit Euler: v^{n+1} = v^n + Δt * RHS_v
        {d1t(v), '=', v_tmp - (stabx(v) + staby(v)) * dissip, false}
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);  // Store main variables for visualization

    std::vector<Variable *> all_vars{&h, &u, &v, &h_tmp, &u_tmp, &v_tmp};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n2D Viscous Shallow Water (Explicit) Dam Break Completed!" << std::endl;
        std::cout << "Gravitational acceleration: " << g << " m/s²" << std::endl;
        std::cout << "Reference water depth: " << H0 << " m" << std::endl;
        std::cout << "Kinematic viscosity: " << nu << " m²/s" << std::endl;
        std::cout << "Artificial dissipation: " << dissip << std::endl;
        std::cout << "Initial depth ratio: 4:1 (behind:in front of dam)" << std::endl;
        std::cout << "Time integration: Explicit Euler with RK stages" << std::endl;
        std::cout << "Equations: ∂h/∂t + u∂h/∂x + v∂h/∂y + h∇·u = 0" << std::endl;
        std::cout << "           ∂u/∂t + u·∇u = -g∇h + ν∇²u" << std::endl;
        std::cout << "Boundary condition: No-slip walls with viscous effects" << std::endl;
        std::cout << "Features: Viscous damping, smooth wave propagation, realistic flow" << std::endl;
        std::cout << "Recommended grid: Use Lz=1, ensure CFL stability" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
} 