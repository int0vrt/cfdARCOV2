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

#ifndef CFDARCO_VAL_UTILS_HPP
#define CFDARCO_VAL_UTILS_HPP

#include "mesh2d.hpp"
#include "cfdarcho_main.hpp"

Eigen::VectorXd initial_with_val(Mesh2D* mesh, double val);

Eigen::VectorXd _boundary_copy(Mesh2D* mesh, Eigen::VectorXd& arr, const Eigen::VectorXd& copy_var);

CudaDataMatrix _boundary_copy_cu(Mesh2D* mesh, CudaDataMatrix& arr, const CudaDataMatrix& copy_var);

inline auto boundary_copy(const Eigen::VectorXd& copy_var) {
    return [copy_var] (Mesh2D* mesh, Eigen::VectorXd& arr) { return _boundary_copy(mesh, arr, copy_var); };
}

inline auto boundary_copy_cu(const Eigen::VectorXd& copy_var) {
    CudaDataMatrix cuda_copy_var;
    if (CFDArcoGlobalInit::cuda_enabled) {
        cuda_copy_var = CudaDataMatrix::from_eigen(copy_var);
    }

    return [cuda_copy_var](Mesh2D *mesh, CudaDataMatrix &arr) {
        if (CFDArcoGlobalInit::cuda_enabled)
            return _boundary_copy_cu(mesh, arr, cuda_copy_var);
        else
            return cuda_copy_var;
    };
}


Eigen::VectorXd _boundary_neumann(Mesh2D* mesh, Eigen::VectorXd& arr, const Eigen::VectorXd& grad_var);

inline auto boundary_neumann(const Eigen::VectorXd& grad_var) {
    return [grad_var] (Mesh2D* mesh, Eigen::VectorXd& arr) { return _boundary_neumann(mesh, arr, grad_var); };
}

inline Eigen::VectorXd boundary_none(Mesh2D* mesh, Eigen::VectorXd& arr) {
    return arr;
}

inline CudaDataMatrix boundary_none_cu(Mesh2D* mesh, CudaDataMatrix& arr) {
    return arr;
}

Eigen::VectorXd boundary_zero(Mesh2D* mesh, Eigen::VectorXd& arr);

CudaDataMatrix boundary_zero_cu(Mesh2D* mesh, CudaDataMatrix& arr);

#endif //CFDARCO_VAL_UTILS_HPP
