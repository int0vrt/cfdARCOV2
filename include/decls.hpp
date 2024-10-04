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
#ifndef CFDARCO_DECLS_HPP
#define CFDARCO_DECLS_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


using MatrixX4dRB = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>;
using MatrixX6dRB = Eigen::Matrix<float, -1, -1, Eigen::ColMajor>;
using MatrixX6Idx = Eigen::Matrix<size_t, -1, -1, Eigen::ColMajor>;
using MatrixX6SignIdx = Eigen::Matrix<ptrdiff_t, -1, -1, Eigen::ColMajor>;
using TensorX6dRB = Eigen::Tensor<float, 3, Eigen::ColMajor>;

template<typename T>
using MatrixX6T = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;

#endif //CFDARCO_DECLS_HPP
