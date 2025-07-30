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
// DIPOLE SOURCE
// ============================================================================

// Pre-compute dipole source locations and parameters
struct DipoleSource {
    CudaDataMatrix<size_t> cu_source_indices;
    CudaDataMatrix<float> cu_source_amplitudes;
    std::vector<size_t> source_indices;
    std::vector<float> source_amplitudes;
    float frequency;
    float t0;
    float sigma;
    
    DipoleSource(Mesh3D *mesh, float freq = 0.1f, float amp = 1.0f) : frequency(freq) {
        t0 = 1.0f / frequency;
        sigma = t0 / 3.0f;
        
        // Pre-compute source locations - dipole antenna at center
        for (int i = 0; i < mesh->_num_nodes; ++i) {
            auto& node = mesh->_nodes[i];
            float x = node.x() / mesh->_lx;
            float y = node.y() / mesh->_ly;
            float z = node.z() / mesh->_lz;
            
            // Two point sources with opposite phase for dipole
            float r1 = std::sqrt((x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f) + (z - 0.35f) * (z - 0.35f));
            float r2 = std::sqrt((x - 0.5f) * (x - 0.5f) + (y - 0.5f) * (y - 0.5f) + (z - 0.65f) * (z - 0.65f));
            
            if (r1 < 0.02f) {
                source_indices.push_back(i);
                source_amplitudes.push_back(amp);
            } else if (r2 < 0.02f) {
                source_indices.push_back(i);
                source_amplitudes.push_back(-amp);
            }
        }

        cu_source_indices =     CudaDataMatrix<size_t>{source_indices.data(), source_indices.size()};
        cu_source_amplitudes =  CudaDataMatrix<float>{source_amplitudes.data(), source_amplitudes.size()};
    }
};

// Dipole source boundary condition with time-dependent excitation
auto create_dipole_source_bc(const DipoleSource &dipole) {
    return [&dipole](Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
        // Apply perfect conductor BC first
        Eigen::Matrix<float, -1, 1> result = arr.cwiseProduct(mesh->_node_is_boundary_reverse);
        
        // Add time-dependent dipole source
        float t = dt_->_current_time_dbl;
        float time_envelope = std::exp(-(t - dipole.t0) * (t - dipole.t0) / (2.0f * dipole.sigma * dipole.sigma));
        float source_term = std::sin(2.0f * M_PI * dipole.frequency * t) * time_envelope;

        for (size_t i = 0; i < dipole.source_indices.size(); ++i) {
            result(dipole.source_indices[i]) += dipole.source_amplitudes[i] * source_term;
        }
        
        return result;
    };
}

__global__ void
boundary_source_dipol(float *a, size_t *source_indices, float *source_amplitudes, float t, float t0, float sigma, float frequency, size_t n_source) {
    auto idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    float time_envelope = std::exp(-(t - t0) * (t - t0) / (2.0f * sigma * sigma));
    float source_term = std::sin(2.0f * M_PI * frequency * t) * time_envelope;

    if (idx_x < n_source) {
        auto src_idx = source_indices[idx_x];
        a[src_idx] += source_amplitudes[idx_x] * source_term;
    }
}

auto create_create_dipole_source_bc_cu(const DipoleSource &dipole) {
    return [&dipole](Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
        auto* cu_mesh = static_cast<CudaMesh3D*>(mesh);
        mul_mtrx_inp(arr, cu_mesh->_node_is_boundary_reverse_cu);

        dim3 blocksize = { 32 };
        dim3 nblocks = {
                (unsigned int) std::ceil((float) dipole.source_indices.size() / (float) blocksize.x),
        };
        boundary_source_dipol<<<nblocks, blocksize>>>(
                arr.data.get(),
                dipole.cu_source_indices.data.get(),
                dipole.cu_source_amplitudes.data.get(),
                dt_->_current_time_dbl,
                dipole.t0,
                dipole.sigma,
                dipole.frequency,
                dipole.source_indices.size());
        sync_device();
    };
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    // Physical parameters (normalized units)
    const float c = 1.0f;  // Normalized speed of light
    const float dt_max = 0.5f * std::min({mesh->_dx, mesh->_dy, mesh->_dz}) / c;  // CFL condition
    
    // Use CFL-limited timestep for stability
    float actual_dt = std::min(initializer.dt, dt_max);
    
    // Create dipole source
    DipoleSource dipole_source(mesh.get(), 0.1f, 1.0f);  // Low frequency for better resolution
    auto dipole_bc = create_dipole_source_bc(dipole_source);
    auto dipole_bc_cu = create_create_dipole_source_bc_cu(dipole_source);

    // Initialize electromagnetic fields to zero
    Eigen::Matrix<float, -1, 1> zero_field = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    
    // Create variables - use native CUDA BC for simple cases, CPU fallback for complex dipole source
    auto Ex = Variable(mesh.get(), zero_field, zero_gradient_bc, zero_gradient_bc_cu, "Ex");
    auto Ey = Variable(mesh.get(), zero_field, zero_gradient_bc, zero_gradient_bc_cu, "Ey");
    auto Ez = Variable(mesh.get(), zero_field, dipole_bc, dipole_bc_cu, "Ez");
    auto Hx = Variable(mesh.get(), zero_field, zero_gradient_bc, zero_gradient_bc_cu, "Hx");
    auto Hy = Variable(mesh.get(), zero_field, zero_gradient_bc, zero_gradient_bc_cu, "Hy");
    auto Hz = Variable(mesh.get(), zero_field, zero_gradient_bc, zero_gradient_bc_cu, "Hz");

    // Time step with CFL-limited dt
    std::vector<Variable *> space_vars{&Ex, &Ey, &Ez, &Hx, &Hy, &Hz};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, actual_dt, space_vars);

    // Maxwell's equations using explicit time-stepping (FDTD method)
    // ∂E/∂t = ∇ × H    (Faraday's law)
    // ∂H/∂t = -∇ × E   (Ampère's law in vacuum)
    EquationTemplate equation_system = {
        // Update electric field: E^{n+1} = E^n + dt * (∇ × H^n)
        NEntryVar{
            {&Ex, '=', Ex + dt * (d1dy(Hz) - d1dz(Hy)), true},
            {&Ey, '=', Ey + dt * (d1dz(Hx) - d1dx(Hz)), true},
            {&Ez, '=', Ez + dt * (d1dx(Hy) - d1dy(Hx)), true}
        }.restruct(),

        // Update magnetic field: H^{n+1} = H^n - dt * (∇ × E^{n+1})
        NEntryVar{
            {&Hx, '=', Hx - dt * (d1dy(Ez) - d1dz(Ey)), true},
            {&Hy, '=', Hy - dt * (d1dz(Ex) - d1dx(Ez)), true},
            {&Hz, '=', Hz - dt * (d1dx(Ey) - d1dy(Ex)), true}
        }.restruct()
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{&Ex, &Ey, &Ez, &Hx, &Hy, &Hz};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Maxwell Equations (FDTD) Simulation Completed!" << std::endl;
        std::cout << "Dipole frequency: " << dipole_source.frequency << " (normalized)" << std::endl;
        std::cout << "Time step: " << actual_dt << " (CFL limited)" << std::endl;
        std::cout << "CFL limit: " << dt_max << std::endl;
        std::cout << "Dipole source points: " << dipole_source.source_indices.size() << std::endl;
        std::cout << "Boundary condition: Perfect conductor (PEC) walls" << std::endl;
        std::cout << "Expected features: Dipole radiation pattern, spherical wavefronts" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
} 