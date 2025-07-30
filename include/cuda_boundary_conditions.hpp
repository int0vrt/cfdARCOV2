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
#pragma once

// empty boundary condition

inline void boundary_none_cu(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {}

// zero gradient boundary condition

inline void zero_gradient_bc_cu(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_) {
    auto *cuda_mesh = dynamic_cast<CudaMesh3D *>(mesh);
    mul_mtrx_inp(arr, cuda_mesh->_node_is_boundary_reverse_cu);
}

// copy boundary condition

inline auto boundary_copy_cu(Mesh3D *mesh_in, const Eigen::Matrix<float, -1, 1> &copy_var) {
    auto *mesh = dynamic_cast<CudaMesh3D *>(mesh_in);
    CudaDataMatrixD cuda_copy_var = CudaDataMatrixD::from_eigen(copy_var);
    mul_mtrx_inp(cuda_copy_var, mesh->_node_is_boundary_cu);

    return [cuda_copy_var](Mesh3D *mesh_in, CudaDataMatrixD &arr, const DT *dt_) {
        auto *mesh = dynamic_cast<CudaMesh3D *>(mesh_in);
        mul_mtrx_inp(arr, mesh->_node_is_boundary_reverse_cu);
        return add_mtrx_inp(arr, cuda_copy_var);
    };
}
