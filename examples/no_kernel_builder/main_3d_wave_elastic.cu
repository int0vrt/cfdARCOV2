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
#include <iostream>
#include <chrono>
#include <argparse/argparse.hpp>
#include <cmath>

#include "operators.hpp"
#include "equation.hpp"
#include "utils3d.hpp"

Eigen::Matrix<float, -1, 1> get_damp_matrix(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, float dt) {
    const float damping_width = 0.2f; // percentage of domain
    const float alpha_max = 4.0f;
    const int power = 2; // polynomial order
    Eigen::Matrix<float, -1, 1> damped = mesh->_node_is_boundary_reverse;

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float dx = std::min(1.0f, std::min(node.x(), mesh->_lx - node.x()) / (damping_width * mesh->_lx));
        float dy = std::min(1.0f, std::min(node.y(), mesh->_ly - node.y()) / (damping_width * mesh->_ly));
        float dz = std::min(1.0f, std::min(node.z(), mesh->_lz - node.z()) / (damping_width * mesh->_lz));
        float d = std::min({dx, dy, dz, 1.0f});
        if (d < 1.0f) {
            d = dx * dy * dz;
            float alpha = alpha_max * std::pow(1.0f - d, power);
            damped(i) *= std::exp(-alpha * dt);
        }

    }
    return damped;
}

auto create_boundary_absorbing(Eigen::Matrix<float, -1, 1> &damp_matrix) {
    return [&damp_matrix](Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
        return arr.cwiseProduct(damp_matrix);
    };
}

auto create_boundary_absorbing_cu(CudaDataMatrixD &damp_matrix) {
    return [&damp_matrix](Mesh3D *mesh_in, CudaDataMatrixD &arr, const DT *dt_) {
        return arr * damp_matrix;
    };
}

auto create_boundary_source_term_combined(Eigen::Matrix<float, -1, 1> &damp_matrix) {
    return [&damp_matrix](Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
        float f0 = 25.0f; // dominant freq in Hz
        float t0 = 0;

        Eigen::Matrix<float, -1, 1> src(mesh->_num_nodes);
        for (int i = 0; i < mesh->_num_nodes; ++i) {
            auto& node = mesh->_nodes[i];
            float r = std::sqrt(std::pow(node.x() - 0.5 * mesh->_lx, 2) +
                                std::pow(node.y() - 0.5 * mesh->_ly, 2) +
                                std::pow(node.z() - 0.03 * mesh->_lz, 2));
            if (r < 0.1 * mesh->_lx) {
                float t_shift = dt_->_current_time_dbl - t0;
                float pi_f0_t = static_cast<float>(M_PI) * f0 * t_shift;
                float value = (1.0f - 2.0f * pi_f0_t * pi_f0_t) * std::exp(-pi_f0_t * pi_f0_t);
                src(i) = value;
            } else {
                src(i) = 0.0f;
            }
        }

        Eigen::Matrix<float, -1, 1> arr_with_force = arr + src;
        return arr_with_force.cwiseProduct(damp_matrix);
    };
}

__global__ void
boundary_source_with_dump(float *a, float *damp, float time, size_t nx, size_t ny, size_t nz, float lx, float ly,
                          float lz, float dx, float dy, float dz) {
    auto idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    auto idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    float f0 = 25.0f; // dominant freq in Hz
    float t0 = 0;

    if (idx_x < nx && idx_y < ny && idx_z < nz) {
        size_t idx_raw = idx_z * nx * ny + idx_y * nx + idx_x;

        float val = a[idx_raw];

        float x = idx_x * dx;
        float y = idx_y * dx;
        float z = idx_z * dx;

        float r = std::sqrt(std::pow(x - 0.5 * lx, 2) +
                            std::pow(y - 0.5 * ly, 2) +
                            std::pow(z - 0.03 * lz, 2));

        if (r < 0.1 * lx) {
            float t_shift = time - t0;
            float pi_f0_t = static_cast<float>(M_PI) * f0 * t_shift;
            float s_value = (1.0f - 2.0f * pi_f0_t * pi_f0_t) * std::exp(-pi_f0_t * pi_f0_t);
            val += s_value;
        }

        a[idx_raw] = val * damp[idx_raw];
    }
}

auto create_boundary_source_term_combined_cu(CudaDataMatrixD &damp_matrix) {
    return [&damp_matrix](Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
        CudaDataMatrixD arr_n{arr};

        dim3 blocksize = {8, 8, 8};
        dim3 nblocks = {
                (unsigned int) std::ceil((float) mesh->_x / (float) blocksize.x),
                (unsigned int) std::ceil((float) mesh->_y / (float) blocksize.y),
                (unsigned int) std::ceil((float) mesh->_z / (float) blocksize.z),
        };
        boundary_source_with_dump<<<nblocks, blocksize>>>(arr_n.data.get(), damp_matrix.data.get(),
                                                          dt_->_current_time_dbl,
                                                          mesh->_x, mesh->_y, mesh->_z,
                                                          mesh->_lx, mesh->_ly, mesh->_lz,
                                                          mesh->_dx, mesh->_dy, mesh->_dz);
        sync_device();

        return arr_n;
    };
}

// Mineral heterogeneity setup
struct Mineral {
    float rho;
    float vp;
    float vs;
    float center_x;
    float center_y;
    float center_z;
    float radius;
};

auto assign_minerals(Mesh3D *mesh) {
    std::vector<Mineral> regions = {
            {2700.0f, 6000.0f, 3464.0f, 0.5f, 0.5f, 0.8f, 0.3f}, // granite (deep)
            {2500.0f, 4500.0f, 2600.0f, 0.5f, 0.5f, 0.5f, 0.3f}, // sandstone (middle)
            {2200.0f, 3200.0f, 1800.0f, 0.5f, 0.5f, 0.2f, 0.3f}  // soil (top)
    };

    Eigen::Matrix<float, -1, 1> rho(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> lambda(mesh->_num_nodes);
    Eigen::Matrix<float, -1, 1> mu(mesh->_num_nodes);

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float x = node.x() / mesh->_lx;
        float y = node.y() / mesh->_ly;
        float z = node.z() / mesh->_lz;

        bool assigned = false;
        for (const auto &m: regions) {
            float dx = x - m.center_x;
            float dy = y - m.center_y;
            float dz = z - m.center_z;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 < m.radius * m.radius) {
                rho(i) = m.rho;
                mu(i) = m.rho * m.vs * m.vs;
                lambda(i) = m.rho * (m.vp * m.vp - 2 * m.vs * m.vs);
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            rho(i) = 2500.0f;
            mu(i) = 2500.0f * 3000.0f * 3000.0f;
            lambda(i) = 2500.0f * (5000.0f * 5000.0f - 2 * 3000.0f * 3000.0f);
        }
    }
    return std::tuple{
            rho,
            lambda,
            mu
    };
}

Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
    return arr;
}

CudaDataMatrixD boundary_none_cu(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
    return arr;
}


int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto [rho_v, lambda_v, mu_v] = assign_minerals(mesh.get());
    auto rho = Variable(mesh.get(), rho_v, boundary_none, "rho");
    auto lambda = Variable(mesh.get(), lambda_v, boundary_none, "lambda");
    auto mu = Variable(mesh.get(), mu_v, boundary_none, "mu");

    Eigen::Matrix<float, -1, 1> zero_field = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
    auto damp_matrix = get_damp_matrix(mesh.get(), zero_field, initializer.dt);
    auto damp_matrix_cu = CudaDataMatrixD::from_eigen(damp_matrix);

    auto vel = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "vel");

    auto vx = Variable(mesh.get(), zero_field, create_boundary_source_term_combined(damp_matrix),
                       create_boundary_source_term_combined_cu(damp_matrix_cu), "vx");
    auto vy = Variable(mesh.get(), zero_field, create_boundary_source_term_combined(damp_matrix),
                       create_boundary_source_term_combined_cu(damp_matrix_cu), "vy");
    auto vz = Variable(mesh.get(), zero_field, create_boundary_source_term_combined(damp_matrix),
                       create_boundary_source_term_combined_cu(damp_matrix_cu), "vz");

    auto vx_interp_x = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vx_interp_x");
    auto vy_interp_y = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vy_interp_y");
    auto vz_interp_z = Variable(mesh.get(), zero_field, boundary_none, boundary_none_cu, "vz_interp_z");

    auto sxx = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "sxx");
    auto syy = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "syy");
    auto szz = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "szz");
    auto sxy = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "sxy");
    auto sxz = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "sxz");
    auto syz = Variable(mesh.get(), zero_field, create_boundary_absorbing(damp_matrix),
                        create_boundary_absorbing_cu(damp_matrix_cu), "syz");

    std::vector<Variable *> space_vars{&vel};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    EquationTemplate equation_system = {
            {&vel,         '=', vx + vy + vz,                                                false},
            {&vx_interp_x, '=', interp(vx) + dt / rho * (d1dx(sxx) + d1dy(sxy) + d1dz(sxz)), false},
            {&vy_interp_y, '=', interp(vy) + dt / rho * (d1dx(sxy) + d1dy(syy) + d1dz(syz)), false},
            {&vz_interp_z, '=', interp(vz) + dt / rho * (d1dx(sxz) + d1dy(syz) + d1dz(szz)), false},

            {&vx, '=', 1 * interp(vx_interp_x, true), false},
            {&vy, '=', 1 * interp(vy_interp_y, true), false},
            {&vz, '=', 1 * interp(vz_interp_z, true), false},


            {&sxx, '=', sxx + dt * ((lambda + 2 * mu) * d1dx(vx) + lambda * (d1dy(vy) + d1dz(vz))), false},
            {&syy, '=', syy + dt * ((lambda + 2 * mu) * d1dy(vy) + lambda * (d1dx(vx) + d1dz(vz))), false},
            {&szz, '=', szz + dt * ((lambda + 2 * mu) * d1dz(vz) + lambda * (d1dx(vx) + d1dy(vy))), false},
            {&sxy, '=', sxy + dt * mu * (d1dy(vx) + d1dx(vy)),                                      false},
            {&sxz, '=', sxz + dt * mu * (d1dz(vx) + d1dx(vz)),                                      false},
            {&syz, '=', syz + dt * mu * (d1dz(vy) + d1dy(vz)),                                      false},
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    std::vector<Variable *> all_vars{&vx, &vy, &vz, &sxx, &syy, &szz, &sxy, &sxz, &syz, &vel, &vx_interp_x,
                                     &vy_interp_y, &vz_interp_z};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();

    if (CFDArcoGlobalInit::get_rank() == 0) {
        std::cout << "\n3D Elastic Wave Simulation Completed!" << std::endl;
        std::cout << "Heterogeneous mineral distribution: granite, sandstone, soil" << std::endl;
        std::cout << "Boundary condition: Absorbing boundaries with damping" << std::endl;
        std::cout << "Source: Ricker wavelet (f0 = 25 Hz)" << std::endl;
        std::cout << "Expected features: P-waves, S-waves, reflections, wave propagation" << std::endl;
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << " [us]" << std::endl;
    }

    initializer.finalize();
    return 0;
}