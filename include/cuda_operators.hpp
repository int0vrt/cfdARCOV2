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
#ifndef CFDARCO_CUDA_OPERATORS_HPP
#define CFDARCO_CUDA_OPERATORS_HPP

#include "decls.hpp"
#include "cuda_data_matrix.hpp"
#include "mesh3d.hpp"


#ifdef CFDARCHO_CUDA_ENABLE


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

CudaDataMatrixD add_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b);

CudaDataMatrixD sub_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b);

CudaDataMatrixD mul_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b);

CudaDataMatrixD mul_mtrx(const CudaDataMatrixD &a, float b);

CudaDataMatrixD div_mtrx(const CudaDataMatrixD &a, const CudaDataMatrixD &b);

CudaDataMatrixD neg_mtrx(const CudaDataMatrixD &a);

CudaDataMatrixD rowwice_sum(const CudaDataMatrixD &a, int rows, int cols);

CudaDataMatrixD mul_mtrx_rowwice(const CudaDataMatrixD &a, const CudaDataMatrixD &b, int rows, int cols);

CudaDataMatrixD mul_mtrx_rowjump(const CudaDataMatrixD &a, const CudaDataMatrixD &b, int rows, int cols, int col_id);

CudaDataMatrixD from_multiple_cols(const std::vector<CudaDataMatrixD> &a);

CudaDataMatrixD get_col(const CudaDataMatrixD &a, int rows, int cols, int col_id);

CudaDataMatrixD div_const(const CudaDataMatrixD &a, float b);

CudaDataMatrixD eval_grad2(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z);

CudaDataMatrixD eval_grad(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z);

CudaDataMatrixD eval_stab(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z);

float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u,
              const CudaDataMatrixD &v, const CudaDataMatrixD &w);

float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u);

inline CudaDataMatrixD operator+(const CudaDataMatrixD &obj_l, const CudaDataMatrixD &obj_r) {
    return add_mtrx(obj_l, obj_r);
}

inline CudaDataMatrixD operator-(const CudaDataMatrixD &obj_l, const CudaDataMatrixD &obj_r) {
    return sub_mtrx(obj_l, obj_r);
}

inline CudaDataMatrixD operator*(const CudaDataMatrixD &obj_l, const CudaDataMatrixD &obj_r) {
    return mul_mtrx(obj_l, obj_r);
}

inline CudaDataMatrixD operator/(const CudaDataMatrixD &obj_l, const CudaDataMatrixD &obj_r) {
    return div_mtrx(obj_l, obj_r);
}

inline CudaDataMatrixD operator-(const CudaDataMatrixD &obj_l) {
    return neg_mtrx(obj_l);
}

inline void sync_device() {
    auto err = cudaDeviceSynchronize();
    gpuErrchk(err);
}

#elif CFDARCHO_HIP_ENABLE

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line, bool abort = true) {
    if (code != hipSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


inline CudaDataMatrixD add_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD sub_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx(const CudaDataMatrixD& a, const float b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD div_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD neg_mtrx(const CudaDataMatrixD& a) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD rowwice_sum(const CudaDataMatrixD& a, int rows, int cols) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx_rowwice(const CudaDataMatrixD& a, const CudaDataMatrixD& b, int rows, int cols) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx_rowjump(const CudaDataMatrixD& a, const CudaDataMatrixD& b, int rows, int cols, int col_id) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD from_multiple_cols(const std::vector<CudaDataMatrixD>& a) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD get_col(const CudaDataMatrixD& a, int rows, int cols, int col_id) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD div_const(const CudaDataMatrixD& a, const float b) {throw std::runtime_error{"CUDA not available"};}

inline CudaDataMatrixD eval_grad2(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD eval_grad(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD eval_stab(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {throw std::runtime_error{"CUDA not available"};}
inline float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u, const CudaDataMatrixD &v, const CudaDataMatrixD &w) {throw std::runtime_error{"CUDA not available"};}
inline float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u) {throw std::runtime_error{"CUDA not available"};}

inline CudaDataMatrixD operator+(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator-(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator*(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator/(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator-(const CudaDataMatrixD& obj_l) {throw std::runtime_error{"CUDA not available"};}

inline void sync_device() {
    auto err = hipDeviceSynchronize();
    gpuErrchk(err);
}

#else

#include <memory>

inline CudaDataMatrixD add_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD sub_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx(const CudaDataMatrixD& a, const float b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD div_mtrx(const CudaDataMatrixD& a, const CudaDataMatrixD& b) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD neg_mtrx(const CudaDataMatrixD& a) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD rowwice_sum(const CudaDataMatrixD& a, int rows, int cols) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx_rowwice(const CudaDataMatrixD& a, const CudaDataMatrixD& b, int rows, int cols) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD mul_mtrx_rowjump(const CudaDataMatrixD& a, const CudaDataMatrixD& b, int rows, int cols, int col_id) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD from_multiple_cols(const std::vector<CudaDataMatrixD>& a) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD get_col(const CudaDataMatrixD& a, int rows, int cols, int col_id) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD div_const(const CudaDataMatrixD& a, const float b) {throw std::runtime_error{"CUDA not available"};}

inline CudaDataMatrixD eval_grad2(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD eval_grad(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD eval_stab(CudaMesh3D* mesh, const CudaDataMatrixD &a, bool clc_x, bool clc_y, bool clc_z) {throw std::runtime_error{"CUDA not available"};}
inline float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u, const CudaDataMatrixD &v, const CudaDataMatrixD &w) {throw std::runtime_error{"CUDA not available"};}
inline float cfl_cu(float dl, float gamma, const CudaDataMatrixD &p, const CudaDataMatrixD &rho, const CudaDataMatrixD &u) {throw std::runtime_error{"CUDA not available"};}

inline CudaDataMatrixD operator+(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator-(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator*(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator/(const CudaDataMatrixD& obj_l, const CudaDataMatrixD& obj_r) {throw std::runtime_error{"CUDA not available"};}
inline CudaDataMatrixD operator-(const CudaDataMatrixD& obj_l) {throw std::runtime_error{"CUDA not available"};}
inline void sync_device() {throw std::runtime_error{"CUDA not available"};}
#endif

#endif //CFDARCO_CUDA_OPERATORS_HPP
