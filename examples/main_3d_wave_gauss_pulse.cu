/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2024 cfdARCHO

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

#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "utils3d.hpp"

Eigen::Matrix<float, -1, 1> _boundary_copy_2d_via_3d(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& copy_var) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = mesh->_dx + 0.001;
    float x_limit_upper = mesh->_lx - mesh->_dx - 0.001;
    float y_limit_lower = mesh->_dy + 0.001;
    float y_limit_upper = mesh->_ly - mesh->_dy - 0.001;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x() && node->x() < x_limit_upper && y_limit_lower < node->y() && node->y() < y_limit_upper) {
            ret(i) = arr(i);
        } else {
            ret(i) = copy_var(i);
        }
        ++i;
    }

    return ret;
}

//auto boundary_copy(const Eigen::Matrix<float, -1, 1>& copy_var) {
//    return [copy_var] (Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) { return _boundary_copy_2d_via_3d(mesh, arr, copy_var); };
//}

Eigen::Matrix<float, -1, 1> _boundary_zero_2d_via_3d(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    int nvals = 1;

    float x_limit_lower = nvals * mesh->_dx + 0.001;
    float x_limit_upper = mesh->_lx - nvals * mesh->_dx - 0.001;
    float y_limit_lower = nvals * mesh->_dy + 0.001;
    float y_limit_upper = mesh->_ly - nvals * mesh->_dy - 0.001;

    for (auto& node : mesh->_nodes) {
        if (x_limit_lower < node->x() && node->x() < x_limit_upper && y_limit_lower < node->y() && node->y() < y_limit_upper) {
            ret(i) = arr(i);
        } else {
            ret(i) = 0;
        }
        ++i;
    }

    return ret;
}

auto boundary_zero(const Eigen::Matrix<float, -1, 1>& copy_var) {
    return [copy_var] (Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) { return _boundary_zero_2d_via_3d(mesh, arr); };
}


//Eigen::Matrix<float, -1, 1> initial_u(Mesh3D* mesh) {
//    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
//    int i = 0;
//
//    float A = 1.0;
//    float sigma = 0.1;
//
//    for (int x = 0; x < mesh->_x; ++x) {
//        for (int y = 0; y < mesh->_y; ++y) {
//            float x_val = std::abs(0.5f - mesh->_lx / (float) mesh->_x * (float) x);
//            float y_val = std::abs(0.5f - mesh->_ly / (float) mesh->_y * (float) y);
////            float x_val = (mesh->_lx / (float) mesh->_x * (float) x) - 0.5f;
////            float y_val = (mesh->_ly / (float) mesh->_y * (float) y) - 0.5f;
//            float res = A * std::exp(-(x_val * x_val + y_val * y_val) / (2 * sigma * sigma));
//            ret(i) = res;
//
//            i++;
//        }
//    }
//
//    return ret;
//}

#include <cmath>

Eigen::Matrix<float, -1, 1> initial_u(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float A = 1.0;
    float sigma = 0.1;

    for (int x = 0; x < mesh->_x; ++x) {
        for (int y = 0; y < mesh->_y; ++y) {
//            float x_val = std::abs(0.5f - mesh->_lx / (float) mesh->_x * (float) x);
//            float y_val = std::abs(0.5f - mesh->_ly / (float) mesh->_y * (float) y);
            float y_val = mesh->_ly / (float) mesh->_y * (float) y;
            float x_val = mesh->_lx / (float) mesh->_x * (float) x;
            float res = A * std::sin(M_PI * x_val) * std::sin(M_PI * y_val);
            ret(i) = res;

            i++;
        }
    }

    return ret;
}

Eigen::Matrix<float, -1, 1> _boundary_copy(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& copy_var) {
    auto arr1 = arr.cwiseProduct(mesh->_node_is_boundary_reverse);
    auto copy_var1 = copy_var.cwiseProduct(mesh->_node_is_boundary);
    return arr1 + copy_var1;
}

inline auto boundary_copy(const Eigen::Matrix<float, -1, 1>& copy_var) {
    return [copy_var] (Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) { return _boundary_copy(mesh, arr, copy_var); };
}

CudaDataMatrixD _boundary_copy_cu(Mesh3D* mesh_in, CudaDataMatrixD& arr, const CudaDataMatrixD& copy_var, const DT* dt_) {
    auto* mesh = dynamic_cast<CudaMesh3D*>(mesh_in);
    auto arr1 = arr * mesh->_node_is_boundary_reverse_cu;
    auto copy_var1 = copy_var * mesh->_node_is_boundary_cu;
    return arr1 + copy_var1;
}

inline auto boundary_copy_cu(const Eigen::Matrix<float, -1, 1>& copy_var) {
    CudaDataMatrixD cuda_copy_var;
    if (CFDArcoGlobalInit::cuda_enabled) {
        cuda_copy_var = CudaDataMatrixD::from_eigen(copy_var);
    }

    return [cuda_copy_var](Mesh3D *mesh, CudaDataMatrixD &arr, const DT* dt_) {
        if (CFDArcoGlobalInit::cuda_enabled)
            return _boundary_copy_cu(mesh, arr, cuda_copy_var, dt_);
        else
            return cuda_copy_var;
    };
}

Eigen::Matrix<float, -1, 1> boundary_sine(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) {

    Eigen::Matrix<float, -1, 1> ret{arr};
//    ret[mesh->square_node_coord_to_idx(mesh->_x * 0.1, mesh->_y * 0.1, mesh->_z * 0.1)] = std::sin(static_cast<float>(dt_->_current_time_step_int) * 0.);
//    ret[mesh->square_node_coord_to_idx(mesh->_x * 0.9, mesh->_y * 0.9, mesh->_z * 0.9)] = std::sin(static_cast<float>(dt_->_current_time_step_int) * 0.);

    return ret;
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};

    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto initial_ = initial_u(mesh.get());
    initial_ = _boundary_zero_2d_via_3d(mesh.get(), initial_);
    auto _zero = initial_with_val(mesh.get(), 0);
//    auto u = Variable(mesh.get(), initial_, boundary_copy(_zero), boundary_copy_cu(_zero), "u");
    auto u = Variable(mesh.get(), initial_, boundary_zero(_zero), "u");

    std::vector<Variable*> space_vars {&u};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    float c = 0.3;

    EquationTemplate equation_system = {
            {d2t(u), '=', c * c * (d2dx(u) + d2dy(u)), false},
    };

    std::vector<Variable*> all_vars {&u};
    auto equation = Equation(timesteps);
    initializer.init_store(all_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, all_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}