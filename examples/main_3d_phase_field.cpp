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

int main(int argc, char** argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    const float phi_solid = 1.0f;
    const float phi_liquid = -1.0f;
    const float W = 0.01f;       // Interface width
    const float tau = 0.0003f;   // Phase relaxation time
    const float lambda = 50.0f;  // Coupling strength
    const float T_M = 0.0f;      // Melting temperature
    const float D = 0.0002f;     // Thermal diffusivity
    const float L = 1.0f;        // Latent heat

    // Initial conditions
    Eigen::Matrix<float, -1, 1> phi_initial(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> T_initial(mesh->_num_nodes);

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float r = std::sqrt(std::pow(node.x() - 0.5f * mesh->_lx, 2) +
                            std::pow(node.y() - 0.5f * mesh->_ly, 2) +
                            std::pow(node.z() - 0.5f * mesh->_lz, 2));
        if (r < 0.05f * mesh->_lx) {
            phi_initial(i) = phi_solid;
            T_initial(i) = -0.0f;
        } else {
            phi_initial(i) = phi_liquid;
            T_initial(i) = -0.8f;
        }
    }

    auto phi = Variable(mesh.get(), phi_initial, zero_gradient_bc, zero_gradient_bc_cu, "phi");
    auto T = Variable(mesh.get(), T_initial, zero_gradient_bc, zero_gradient_bc_cu, "T");

    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, {&phi, &T});

    EquationTemplate eqs = {
            NEntryVar{
                {&phi, '=', phi + dt * (W * W * lapl(phi) - (phi * (phi * phi - 1.0f) + lambda * (T - T_M) * (1.0f - phi * phi))) / tau, true},
            }.restruct(),
            NEntryVar{
                {&T,   '=', T + dt * (D * lapl(T) + (L / 2.0f) * phi * dt), true}
            }.restruct(),
    };

    auto equation = Equation(timesteps);
    initializer.init_store({&phi, &T});

    std::vector<Variable*> all_vars{&phi, &T};
    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, eqs, &dt, initializer.visualize, {&phi});
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Phase Field Simulation Completed!" << std::endl;
        std::cout << "Interface width (W): " << W << std::endl;
        std::cout << "Phase relaxation time (τ): " << tau << std::endl;
        std::cout << "Coupling strength (λ): " << lambda << std::endl;
        std::cout << "Thermal diffusivity (D): " << D << std::endl;
        std::cout << "Expected behavior: Solid-liquid phase transition" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
}
