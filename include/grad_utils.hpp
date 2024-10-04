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
#ifndef CFDARCO_GRAD_UTILS_HPP
#define CFDARCO_GRAD_UTILS_HPP

#include "fvm3d.hpp"

template<typename MeshClass>
MatrixX6dRB interpolate_to_face_linear(MeshClass *mesh, Eigen::Matrix<float, -1, 1> *var) {
    MatrixX6dRB ret{mesh->_num_nodes, MeshClass::n_faces};

//#pragma omp parallel for
    for (long i = 0; i < mesh->_num_nodes; ++i) {
        auto crr = var->operator()(i);
#pragma unroll
        for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
            ret(i, fc) = (crr + var->operator()(mesh->_ids(i, fc))) * 0.5;
        }
    }
    return ret;
}

template<typename MeshClass>
std::vector<MatrixX6dRB>
interpolate_to_face_upwing(MeshClass *mesh, Eigen::Matrix<float, -1, 1> *var, std::vector<Eigen::Matrix<float, -1, 1>> *grad) {
    std::vector<MatrixX6dRB> ret{};

//#pragma unroll
    for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
        MatrixX6dRB ret_dim{mesh->_num_nodes, MeshClass::n_faces};
//#pragma omp parallel for
        for (long i = 0; i < mesh->_num_nodes; ++i) {
            auto crr = var->operator()(i);
            auto crr_grad = grad->at(dm)(i);
#pragma unroll
            for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
                ret_dim(i, fc) = crr + mesh->_len_node_center_to_face(i, fc) * crr_grad *
                                       mesh->_normals_alt_all.at(dm)->operator()(i, fc);
            }
        }
        ret.push_back(ret_dim);
    }
    return ret;
}

template<typename MeshClass, bool use_bound_free>
MatrixX6dRB collect_vals_neigh_faces(MeshClass *mesh, MatrixX6dRB *var) {
    MatrixX6dRB ret{mesh->_num_nodes, MeshClass::n_faces};

//#pragma omp parallel for
    for (long i = 0; i < mesh->_num_nodes; ++i) {
#pragma unroll
        for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
            if constexpr (use_bound_free) {
                if (mesh->_ids_bound_free(i, fc) == -1) {
                    ret(i, fc) = 0;
                    continue;
                }
            }
            auto neigh_id = mesh->_ids(i, fc);
            ret(i, fc) = var->operator()(neigh_id, mesh->_nodes.at(neigh_id)->opposite_face_id(fc));
        }
    }
    return ret;
}

template<typename MeshClass, bool use_alt_normals = false>
std::vector<Eigen::Matrix<float, -1, 1>> gauss_grad(MeshClass *mesh, MatrixX6dRB *face_interpolated) {
    std::vector<Eigen::Matrix<float, -1, 1>> ret{};

#pragma unroll
    for (int i = 0; i < MeshClass::n_dims; ++i) {
        auto *normal_ptr = mesh->_normals_all.at(i);
        if constexpr (!use_alt_normals) {
            normal_ptr = mesh->_normals_alt_all.at(i);
        }
        MatrixX6dRB face_times_norm = face_interpolated->cwiseProduct(*normal_ptr);
        MatrixX6dRB face_times_area = face_times_norm.cwiseProduct(mesh->_face_areas);
        Eigen::Matrix<float, -1, 1> summed_faces = face_times_area.rowwise().sum();
        ret.push_back(summed_faces.cwiseQuotient(mesh->_volumes));
    }

    return ret;
}

template<typename MeshClass>
MatrixX6dRB
corrected_surface_normal_grad(MeshClass *mesh, std::vector<Eigen::Matrix<float, -1, 1>> &cell_grad, Eigen::Matrix<float, -1, 1> *var) {
    MatrixX6dRB ret{mesh->_num_nodes, MeshClass::n_faces};

    // implicit
#pragma omp parallel for
    for (long i = 0; i < mesh->_num_nodes; ++i) {
        auto crr = var->operator()(i);
#pragma unroll
        for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
            ret(i, fc) = (-crr + var->operator()(mesh->_ids(i, fc)));
        }
    }
    ret = ret.cwiseProduct(mesh->_alpha_d);

//    // correction
//#pragma unroll
//    for (int i = 0; i < MeshClass::n_dims; ++i) {
//        auto corr_term = mesh->_n_min_alpha_d.at(i).cwiseProduct(interpolate_to_face_linear(mesh, &(cell_grad.at(i))));
//        ret += corr_term;
//    }

    return ret;
}


#endif //CFDARCO_GRAD_UTILS_HPP
