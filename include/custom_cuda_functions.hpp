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
#ifndef CFDARCO_CUSTOM_CUDA_FUNCTIONS_HPP
#define CFDARCO_CUSTOM_CUDA_FUNCTIONS_HPP

//#include "cuda_operators.hpp"
#include "decls.hpp"
#include "mesh3d.hpp"

__device__ inline size_t global_stride(size_t n, int fc, size_t idx) { return n * fc + idx; }
__device__ inline size_t local_stride(size_t n, int fc, size_t idx) { return fc; }


template<typename MeshClass, typename T>
__device__ inline void read_face_vars_from_neigh(const T * __restrict__ var,
                                                 const size_t ids[MeshClass::n_faces],
                                                 T ret[MeshClass::n_faces],
                                                 size_t n) {
#pragma unroll
    for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
        ret[fc] = var[global_stride(n, fc, ids[fc])];
    }
}

template<int N, typename T>
__device__ inline void read_n_vars_from_self(const T * __restrict__ var,
                                             T ret[N],
                                             size_t idx, size_t n) {
#pragma unroll
    for (int fc = 0; fc < N; ++fc) {
        ret[fc] = var[global_stride(n, fc, idx)];
    }
}

template<typename MeshClass, typename T>
__device__ inline void read_n_vars_from_neigh(const T * __restrict__ var,
                                              const size_t ids[MeshClass::n_faces],
                                              T ret[MeshClass::n_faces]) {
#pragma unroll
    for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
        ret[fc] = var[ids[fc]];
    }
}


template<typename MeshClass, typename T>
__device__ inline void read_normals_from_self(const T * __restrict__ var[MeshClass::n_dims],
                                              T ret[MeshClass::n_dims][MeshClass::n_faces],
                                              size_t idx, size_t n) {
#pragma unroll
    for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
        for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
            ret[dm][fc] = var[dm][global_stride(n, fc, idx)];
        }
    }
}

template<int N, typename T>
__device__ inline void write_n_vars_to_self(T * __restrict__ var,
                                            const T ret[N],
                                            size_t idx, size_t n) {
#pragma unroll
    for (int fc = 0; fc < N; ++fc) {
        var[global_stride(n, fc, idx)] = ret[fc];
    }
}

template<typename T>
__device__ inline void write_one_var_to_self(T * __restrict__ var,
                                            const T ret,
                                            size_t idx, size_t n) {
    var[global_stride(n, 0, idx)] = ret;
}

__device__ int inline opposite_face_id(int face_id) {
    if (face_id == 0) return 1;
    if (face_id == 1) return 0;
    if (face_id == 2) return 3;
    if (face_id == 3) return 2;
    if (face_id == 4) return 5;
    if (face_id == 5) return 4;
    return 0;
}

template<typename MeshClass>
__device__ inline void read_face_vars_from_neigh_opposite_face(const float * __restrict__ var[MeshClass::n_dims],
                                                               const size_t ids[MeshClass::n_faces],
                                                               float ret[MeshClass::n_dims][MeshClass::n_faces],
                                                               size_t n) {
#pragma unroll
    for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
        auto neigh_id = ids[fc];
#pragma unroll
        for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
            ret[dm][fc] = var[dm][global_stride(n, opposite_face_id(fc), neigh_id)];
        }
    }
}

template<typename T>
__device__ inline T read_scalar_from_self(const T * __restrict__ var, size_t idx, size_t n) {
    return var[global_stride(n, 0, idx)];
}

template<typename MeshClass>
__device__ inline void interpolate_to_face_linear_cu_k(const float crr,
                                                       const float var_neigh[MeshClass::n_faces],
                                                       float ret[MeshClass::n_faces]) {
#pragma unroll
    for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
        ret[fc] = (crr + var_neigh[fc]) * 0.5;
    }
}

template<typename MeshClass>
__device__ inline void interpolate_to_face_upwing_cu_k(const float crr,
                                                       const float grad[MeshClass::n_dims],
                                                       const float len_node_center_to_face[MeshClass::n_faces],
                                                       const float normals[MeshClass::n_dims][MeshClass::n_faces],
                                                       float ret[MeshClass::n_dims][MeshClass::n_faces]) {
#pragma unroll
    for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
        auto crr_grad = grad[dm];
#pragma unroll
        for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
            auto len_to_face = len_node_center_to_face[fc];
            auto normal = normals[dm][fc];
            ret[dm][fc] = crr + len_to_face * crr_grad * normal;
        }
    }
}

template<typename MeshClass>
__device__ inline void gauss_grad_cu_k(const float face_interpolated[MeshClass::n_faces],
                                       const float normals[MeshClass::n_dims][MeshClass::n_faces],
                                       const float face_area[MeshClass::n_faces],
                                       const float volume,
                                       float ret[MeshClass::n_dims]) {
#pragma unroll
    for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
        float accum = 0.0;

#pragma unroll
        for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
            float face_times_norm = face_interpolated[fc] * normals[dm][fc];
            float face_times_area = face_times_norm * face_area[fc];
            accum += face_times_area;
        }

        ret[dm] = accum / volume;
    }
}

template<typename MeshClass>
__device__ inline void corrected_surface_normal_grad_cu_k(const float crr,
                                                          const float var_neigh[MeshClass::n_faces],
                                                          const float alpha_d[MeshClass::n_faces],
                                                          float ret[MeshClass::n_faces]) {
#pragma unroll
    for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
        ret[fc] = (-crr + var_neigh[fc]) * alpha_d[fc];
    }

    // TODO: add correction
}


#endif //CFDARCO_CUSTOM_CUDA_FUNCTIONS_HPP
