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
#include "cuda_operators.hpp"
#include "decls.hpp"
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <custom_cuda_functions.hpp>

#define BLOCK_SIZE 64

__global__ void add_mtrx_k(const float *a, const float *b, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

CudaDataMatrixD add_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    add_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);

    return res;
}

__global__ void sub_mtrx_k(const float *a, const float *b, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

CudaDataMatrixD sub_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    sub_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);

    return res;
}

__global__ void mul_mtrx_k(const float *a, const float *b, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

CudaDataMatrixD mul_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    mul_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);

    return res;
}

__global__ void mul_mtrx_by_float_k(const float *a, float b, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b;
    }
}

CudaDataMatrixD mul_mtrx(const CudaDataMatrixD &a, const float b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    mul_mtrx_by_float_k<<<nblocks, blocksize>>>(a.data.get(), b, res.data.get(), a._size);
    return res;
}

__global__ void div_mtrx_k(const float *a, const float *b, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

CudaDataMatrixD div_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    div_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), b.data.get(), res.data.get(), a._size);

    return res;
}

__global__ void div_const_k(const float *a, const float b, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b;
    }
}

CudaDataMatrixD div_const(const CudaDataMatrixD &a, const float b) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    div_const_k<<<nblocks, blocksize>>>(a.data.get(), b, res.data.get(), a._size);

    return res;
}

__global__ void neg_mtrx_k(const float *a, float *c, int n) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = -a[idx];
    }
}

CudaDataMatrixD neg_mtrx(const CudaDataMatrixD &a) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};
    neg_mtrx_k<<<nblocks, blocksize>>>(a.data.get(), res.data.get(), a._size);

    return res;
}

__global__ void cfl_cu_k(
        float dl, float gamma,
        const float * __restrict__ p_in,
        const float * __restrict__ rho_in,
        const float * __restrict__ u_in,
        const float * __restrict__ v_in,
        const float * __restrict__ w_in,
        int rows,
        float * __restrict__ value_memory
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        auto p = p_in[idx];
        auto rho = rho_in[idx];
        auto u = u_in[idx];
        auto v = v_in[idx];
        auto w = w_in[idx];

        auto p1 = sqrt((gamma * p) / rho);
        auto p2 = sqrt(u * u + v * v + w * w);
        auto result = dl * (1.0 / (p1 + p2));
        value_memory[idx] = result;
    }
}

float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u,
              const CudaDataMatrixD &v, const CudaDataMatrixD &w) {
    int rows = p._size;
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(rows) / static_cast<float>(blocksize));

    CudaDataMatrixD memory_value{static_cast<size_t>(rows)};

    cfl_cu_k<<<nblocks, blocksize>>>(
            dl,
            gamma,
            p.data.get(),
            rho.data.get(),
            u.data.get(),
            v.data.get(),
            w.data.get(),
            rows,
            memory_value.data.get()
    );

    sync_device();
    thrust::device_ptr<float> ptr = thrust::device_pointer_cast<float>(memory_value.data.get());
    auto min_ptr = thrust::min_element(ptr, ptr + rows);

    return *min_ptr;
}

__global__ void cfl_cu_k(
        float dl, float gamma,
        const float * __restrict__ p_in,
        const float * __restrict__ rho_in,
        const float * __restrict__ u_in,
        int rows,
        float * __restrict__ value_memory
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows) {
        auto p = p_in[idx];
        auto rho = rho_in[idx];
        auto u = u_in[idx];

        auto p1 = sqrt((gamma * p) / rho);
        auto p2 = sqrt(u * u);
        auto result = dl * (1.0 / (p1 + p2));
        value_memory[idx] = result;
    }
}

float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u) {
    int rows = p._size;
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(rows) / static_cast<float>(blocksize));

    CudaDataMatrixD memory_value{static_cast<size_t>(rows)};

    cfl_cu_k<<<nblocks, blocksize>>>(
            dl,
            gamma,
            p.data.get(),
            rho.data.get(),
            u.data.get(),
            rows,
            memory_value.data.get()
    );

    sync_device();
    thrust::device_ptr<float> ptr = thrust::device_pointer_cast<float>(memory_value.data.get());
    auto min_ptr = thrust::min_element(ptr, ptr + rows);

    return *min_ptr;
}

template<typename MeshClass>
__global__ void eval_grad_k(const float * __restrict__ var_ptr,
                          const size_t * __restrict__ ids_ptr,
                          const float * __restrict__ normals_x_ptr,
                          const float * __restrict__ normals_y_ptr,
                          const float * __restrict__ normals_z_ptr,
                          const float * __restrict__ face_area_ptr,
                          const float * __restrict__ volume_ptr,
                          float * __restrict__ ret_ptr,
                          bool clc_x, bool clc_y, bool clc_z,
                          size_t n) {

    float interpolation_ret[MeshClass::n_faces];
    size_t face_neigh_ids[MeshClass::n_faces];
    float face_area[MeshClass::n_faces];
    float var_neigh[MeshClass::n_faces];
    float normals_face[MeshClass::n_dims][MeshClass::n_faces];
    const float* normals_ptr[MeshClass::n_dims] = {
            normals_x_ptr,
            normals_y_ptr,
            normals_z_ptr
    };
    float grad[MeshClass::n_dims];

    auto idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float crr = read_scalar_from_self(var_ptr, idx, n);
        float volume = read_scalar_from_self(volume_ptr, idx, n);

        read_n_vars_from_self<MeshClass::n_faces>(ids_ptr, face_neigh_ids, idx, n);
        read_normals_from_self<MeshClass>(normals_ptr, normals_face, idx, n);
        read_n_vars_from_self<MeshClass::n_faces>(face_area_ptr, face_area, idx, n);

        read_n_vars_from_neigh<MeshClass>(var_ptr, face_neigh_ids, var_neigh);

        interpolate_to_face_linear_cu_k<MeshClass>(crr, var_neigh, interpolation_ret);
        gauss_grad_cu_k<MeshClass>(interpolation_ret, normals_face, face_area, volume, grad);

        float accum[1];
        accum[0] = 0.0;
        if (clc_x) accum[0] += grad[0];
        if (clc_y) accum[0] += grad[1];
        if (clc_z) accum[0] += grad[2];

        write_n_vars_to_self<1>(ret_ptr, accum, idx, n);
    }
}

CudaDataMatrixD eval_grad(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};

    eval_grad_k<Mesh3D><<<nblocks, blocksize>>>(
            a.data.get(),
            mesh->_ids_cu.data.get(),

            mesh->_normal_alt_x_cu.data.get(),
            mesh->_normal_alt_y_cu.data.get(),
            mesh->_normal_alt_z_cu.data.get(),

            mesh->_face_areas_cu.data.get(),
            mesh->_volumes_cu.data.get(),
            res.data.get(),
            clc_x, clc_y, clc_z,
            a._size);
    sync_device();
    return res;
}

template<typename MeshClass>
__global__ void eval_stab_first_stage_k(const float * __restrict__ var_ptr,
                            const size_t * __restrict__ ids_ptr,
                            const float * __restrict__ normals_x_ptr,
                            const float * __restrict__ normals_y_ptr,
                            const float * __restrict__ normals_z_ptr,
                            const float * __restrict__ face_area_ptr,
                            const float * __restrict__ volume_ptr,
                            const float * __restrict__ len_node_center_to_face_ptr,
                            float * __restrict__ ret_x_ptr,
                            float * __restrict__ ret_y_ptr,
                            float * __restrict__ ret_z_ptr,
                            bool clc_x, bool clc_y, bool clc_z,
                            size_t n) {

    float interpolation_ret[MeshClass::n_faces];
    float len_node_center_to_face[MeshClass::n_faces];
    size_t face_neigh_ids[MeshClass::n_faces];
    float face_area[MeshClass::n_faces];
    float var_neigh[MeshClass::n_faces];
    float normals_face[MeshClass::n_dims][MeshClass::n_faces];
    const float* normals_ptr[MeshClass::n_dims] = {
            normals_x_ptr,
            normals_y_ptr,
            normals_z_ptr
    };
    float grad[MeshClass::n_dims];
    float interpolated_upwing[MeshClass::n_dims][MeshClass::n_faces];

    auto idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float crr = read_scalar_from_self(var_ptr, idx, n);
        float volume = read_scalar_from_self(volume_ptr, idx, n);

        read_n_vars_from_self<MeshClass::n_faces>(len_node_center_to_face_ptr, len_node_center_to_face, idx, n);
        read_n_vars_from_self<MeshClass::n_faces>(ids_ptr, face_neigh_ids, idx, n);
        read_normals_from_self<MeshClass>(normals_ptr, normals_face, idx, n);
        read_n_vars_from_self<MeshClass::n_faces>(face_area_ptr, face_area, idx, n);

        read_n_vars_from_neigh<MeshClass>(var_ptr, face_neigh_ids, var_neigh);

        interpolate_to_face_linear_cu_k<MeshClass>(crr, var_neigh, interpolation_ret);
        gauss_grad_cu_k<MeshClass>(interpolation_ret, normals_face, face_area, volume, grad);

        interpolate_to_face_upwing_cu_k<MeshClass>(crr, grad, len_node_center_to_face, normals_face, interpolated_upwing);

        write_n_vars_to_self<MeshClass::n_faces>(ret_x_ptr, interpolated_upwing[0], idx, n);
        write_n_vars_to_self<MeshClass::n_faces>(ret_y_ptr, interpolated_upwing[1], idx, n);
        write_n_vars_to_self<MeshClass::n_faces>(ret_z_ptr, interpolated_upwing[2], idx, n);
    }
}

template<typename MeshClass>
__global__ void eval_stab_second_stage_k(
                                        const float * __restrict__ interpolated_upwing_x_prt,
                                        const float * __restrict__ interpolated_upwing_y_prt,
                                        const float * __restrict__ interpolated_upwing_z_prt,
                                        const size_t * __restrict__ ids_ptr,
                                        const float * __restrict__ normals_x_ptr,
                                        const float * __restrict__ normals_y_ptr,
                                        const float * __restrict__ normals_z_ptr,
                                        const float * __restrict__ face_area_ptr,
                                        const float * __restrict__ volume_ptr,
                                        const float * __restrict__ len_node_center_to_face_ptr,
                                        float * __restrict__ ret_ptr,
                                        bool clc_x, bool clc_y, bool clc_z,
                                        size_t n) {

    float interpolated_upwing[MeshClass::n_dims][MeshClass::n_faces];
    float partial_res[MeshClass::n_dims];
    float summed_faces[MeshClass::n_faces];
    float interpolated_upwing_collected[MeshClass::n_dims][MeshClass::n_faces];
    float len_node_center_to_face[MeshClass::n_faces];
    size_t face_neigh_ids[MeshClass::n_faces];
    float face_area[MeshClass::n_faces];
    float var_neigh[MeshClass::n_faces];
    float normals_face[MeshClass::n_dims][MeshClass::n_faces];
    const float* normals_ptr[MeshClass::n_dims] = {
            normals_x_ptr,
            normals_y_ptr,
            normals_z_ptr
    };
    const float* interpolated_upwing_ptr[MeshClass::n_dims] = {
            interpolated_upwing_x_prt,
            interpolated_upwing_y_prt,
            interpolated_upwing_z_prt
    };
    const bool dm_enabled[] = {
            clc_x,
            clc_y,
            clc_z
    };
    float grad[MeshClass::n_dims];

    float accum[1];
    accum[0] = 0.0;

    auto idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float volume = read_scalar_from_self(volume_ptr, idx, n);

        read_n_vars_from_self<MeshClass::n_faces>(ids_ptr, face_neigh_ids, idx, n);
        read_n_vars_from_self<MeshClass::n_faces>(len_node_center_to_face_ptr, len_node_center_to_face, idx, n);
#pragma unroll
        for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
            read_n_vars_from_self<MeshClass::n_faces>(normals_ptr[dm], normals_face[dm], idx, n);
            read_n_vars_from_self<MeshClass::n_faces>(interpolated_upwing_ptr[dm], interpolated_upwing[dm], idx, n);
        }
        read_n_vars_from_self<MeshClass::n_faces>(face_area_ptr, face_area, idx, n);

        read_face_vars_from_neigh_opposite_face<MeshClass>(interpolated_upwing_ptr, face_neigh_ids, interpolated_upwing_collected, n);


#pragma unroll
        for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
            if (!dm_enabled[dm]) continue;

#pragma unroll
            for (int fc = 0; fc < MeshClass::n_faces; ++fc) {
                summed_faces[fc] = 0.5 * (interpolated_upwing[dm][fc] - interpolated_upwing_collected[dm][fc]);
            }
            gauss_grad_cu_k<MeshClass>(summed_faces, normals_face, face_area, volume, partial_res);
            accum[0] += partial_res[dm];
        }

        write_n_vars_to_self<1>(ret_ptr, accum, idx, n);
    }
}

CudaDataMatrixD eval_stab(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};

    CudaDataMatrixD interpolated_upwing_x{a._size * CudaMesh3D::n_faces};
    CudaDataMatrixD interpolated_upwing_y{a._size * CudaMesh3D::n_faces};
    CudaDataMatrixD interpolated_upwing_z{a._size * CudaMesh3D::n_faces};

    eval_stab_first_stage_k<Mesh3D><<<nblocks, blocksize>>>(
            a.data.get(),
            mesh->_ids_cu.data.get(),

            mesh->_normal_alt_x_cu.data.get(),
            mesh->_normal_alt_y_cu.data.get(),
            mesh->_normal_alt_z_cu.data.get(),

            mesh->_face_areas_cu.data.get(),
            mesh->_volumes_cu.data.get(),
            mesh->_len_node_center_to_face_cu.data.get(),

            interpolated_upwing_x.data.get(),
            interpolated_upwing_y.data.get(),
            interpolated_upwing_z.data.get(),

            clc_x, clc_y, clc_z,
            a._size);
    sync_device();

    eval_stab_second_stage_k<Mesh3D><<<nblocks, blocksize>>>(
            interpolated_upwing_x.data.get(),
            interpolated_upwing_y.data.get(),
            interpolated_upwing_z.data.get(),

            mesh->_ids_cu.data.get(),

            mesh->_normal_x_cu.data.get(),
            mesh->_normal_y_cu.data.get(),
            mesh->_normal_z_cu.data.get(),

            mesh->_face_areas_cu.data.get(),
            mesh->_volumes_cu.data.get(),
            mesh->_len_node_center_to_face_cu.data.get(),

            res.data.get(),

            clc_x, clc_y, clc_z,
            a._size);
    sync_device();

    return res;
}


template<typename MeshClass>
__global__ void eval_grad2_k(const float * __restrict__ var_ptr,
                          const size_t * __restrict__ ids_ptr,
                          const float * __restrict__ normals_alt_x_ptr,
                          const float * __restrict__ normals_alt_y_ptr,
                          const float * __restrict__ normals_alt_z_ptr,
                         const float * __restrict__ normals_x_ptr,
                         const float * __restrict__ normals_y_ptr,
                         const float * __restrict__ normals_z_ptr,
                          const float * __restrict__ face_area_ptr,
                          const float * __restrict__ alpha_d_ptr,
                          const float * __restrict__ volume_ptr,
                          float * __restrict__ ret_ptr,
                          bool clc_x, bool clc_y, bool clc_z,
                          size_t n) {

    size_t face_neigh_ids[MeshClass::n_faces];
    float face_area[MeshClass::n_faces];
    float alpha_d[MeshClass::n_faces];
    float var_neigh[MeshClass::n_faces];
    float normals_face[MeshClass::n_dims][MeshClass::n_faces];
    const float* normals_ptr[MeshClass::n_dims] = {
            normals_x_ptr,
            normals_y_ptr,
            normals_z_ptr
    };
    float corrected_surface_normal_grad[MeshClass::n_faces];
    float lapl[MeshClass::n_dims];

    auto idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float crr = read_scalar_from_self(var_ptr, idx, n);
        float volume = read_scalar_from_self(volume_ptr, idx, n);

        read_n_vars_from_self<MeshClass::n_faces>(ids_ptr, face_neigh_ids, idx, n);
        read_n_vars_from_self<MeshClass::n_faces>(face_area_ptr, face_area, idx, n);
        read_n_vars_from_self<MeshClass::n_faces>(alpha_d_ptr, alpha_d, idx, n);

        for (int dm = 0; dm < MeshClass::n_dims; ++dm) {
            read_n_vars_from_self<MeshClass::n_faces>(normals_ptr[dm], normals_face[dm], idx, n);
        }

        read_n_vars_from_neigh<MeshClass>(var_ptr, face_neigh_ids, var_neigh);
        corrected_surface_normal_grad_cu_k<MeshClass>(crr, var_neigh, alpha_d, corrected_surface_normal_grad);
        gauss_grad_cu_k<MeshClass>(corrected_surface_normal_grad, normals_face, face_area, volume, lapl);

        float accum[1];
        accum[0] = 0.0;
        if (clc_x) accum[0] += lapl[0];
        if (clc_y) accum[0] += lapl[1];
        if (clc_z) accum[0] += lapl[2];

        write_n_vars_to_self<1>(ret_ptr, accum, idx, n);
    }
}

CudaDataMatrixD eval_grad2(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {
    int blocksize = BLOCK_SIZE;
    int nblocks = std::ceil(static_cast<float>(a._size) / static_cast<float>(blocksize));
    CudaDataMatrixD res{a._size};

    eval_grad2_k<Mesh3D><<<nblocks, blocksize>>>(a.data.get(),
                                       mesh->_ids_cu.data.get(),

                                        mesh->_normal_alt_x_cu.data.get(),
                                        mesh->_normal_alt_y_cu.data.get(),
                                        mesh->_normal_alt_z_cu.data.get(),
                                        mesh->_normal_x_cu.data.get(),
                                        mesh->_normal_y_cu.data.get(),
                                        mesh->_normal_z_cu.data.get(),

                                       mesh->_face_areas_cu.data.get(),
                                       mesh->_alpha_d_cu.data.get(),
                                       mesh->_volumes_cu.data.get(),
                                       res.data.get(),
                                       clc_x, clc_y, clc_z,
                                       a._size);
    sync_device();
    return res;
}

