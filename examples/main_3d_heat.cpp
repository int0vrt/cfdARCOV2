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
//#include <matplot/matplot.h>
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "utils3d.hpp"

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

Eigen::Matrix<float, -1, 1> initial_T(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    ret.setConstant(0);

    size_t i = 0;
    for (auto& node : mesh->_nodes) {
        if (node->z() < 2 * mesh->_dz) {
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

    Eigen::Matrix<float, -1, 1> T_initial = initial_T(mesh.get());
    auto T = Variable(mesh.get(), T_initial, boundary_copy(T_initial), boundary_copy_cu(T_initial), "T");

    std::vector<Variable*> space_vars {&T};
    auto dt = DT(mesh.get(), UpdatePolicies::constant_dt, UpdatePolicies::constant_dt_cu, initializer.dt, space_vars);

    EquationTemplate equation_system = {
            {d1t(T), '=', 3.0 * lapl(T), true},
    };

    auto equation = Equation(timesteps);
    initializer.init_store(space_vars);

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(space_vars, equation_system, &dt, initializer.visualize, space_vars);
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}