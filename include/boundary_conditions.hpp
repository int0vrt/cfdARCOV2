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

inline Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) { return arr; }

// zero gradient boundary condition

inline Eigen::Matrix<float, -1, 1> zero_gradient_bc(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
    return arr.cwiseProduct(mesh->_node_is_boundary_reverse);
}

// copy boundary condition

inline auto boundary_copy(const Eigen::Matrix<float, -1, 1> &copy_var) {
    return [copy_var](Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT *dt_) {
        auto arr1 = arr.cwiseProduct(mesh->_node_is_boundary_reverse);
        auto copy_var1 = copy_var.cwiseProduct(mesh->_node_is_boundary);
        return arr1 + copy_var1;
    };
}

