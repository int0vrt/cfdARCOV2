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
#ifndef CFDARCO_CUDA_DATA_MATRIX_HPP
#define CFDARCO_CUDA_DATA_MATRIX_HPP

#include <memory>
#include <new>
#include <iostream>


#include "cfdarcho_main_3d.hpp"
#ifdef CFDARCHO_HIP_ENABLE
#include "hip/hip_runtime.h"
#endif
#ifdef CFDARCHO_CUDA_ENABLE
#include <cuda_runtime_api.h>
#include "pool_allocator.hpp"
#endif

template<typename T>
class CudaDeleter {
public:
    explicit CudaDeleter(size_t size) : _size{size} {}

    void operator()(void *p) const {
#ifndef CFDARCO_SKIP_RMM
        if (_size > 0) {
#ifdef CFDARCHO_CUDA_ENABLE
            if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
                Allocator::cuda_mem_pool->deallocate(p, _size * sizeof(T));
            }
#endif
#ifdef CFDARCHO_HIP_ENABLE
            if (CFDArcoGlobalInit::hip_enabled) {
                hipFree(p);
            }
#endif
        }
#endif
    }

    size_t _size;
};

template<typename T>
class CudaDataMatrix {
public:
    CudaDataMatrix() {
        data = std::shared_ptr<T>(nullptr, CudaDeleter<T>{0});
        _size = 0;
    }

    explicit CudaDataMatrix(size_t size) : _size{size} {
#ifndef CFDARCO_SKIP_RMM
        void* ptr = nullptr;
#ifdef CFDARCHO_CUDA_ENABLE
        if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
            ptr = Allocator::cuda_mem_pool->allocate(_size * sizeof(T));
        }
#endif
#ifdef CFDARCHO_HIP_ENABLE
        if (CFDArcoGlobalInit::hip_enabled) {
            hipMalloc(&ptr, _size * sizeof(T));
        }
#endif
        data = std::shared_ptr<T>(static_cast<T *>(ptr), CudaDeleter<T>{_size});
#endif
    }

    CudaDataMatrix(const CudaDataMatrix &oth) {
#ifndef CFDARCO_SKIP_RMM
        _size = oth._size;
        if (oth.data != nullptr) {
            void* ptr = nullptr;
#ifdef CFDARCHO_CUDA_ENABLE
            if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
                ptr = Allocator::cuda_mem_pool->allocate(_size * sizeof(T));
            }
#endif
#ifdef CFDARCHO_HIP_ENABLE
            if (CFDArcoGlobalInit::hip_enabled) {
                hipMalloc(&ptr, _size * sizeof(T));
            }
#endif
            data = std::shared_ptr<T>(static_cast<T *>(ptr), CudaDeleter<T>{_size});
#ifdef CFDARCHO_CUDA_ENABLE
            if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
                cudaMemcpy(data.get(), oth.data.get(), _size * sizeof(T), cudaMemcpyDefault);
            }
#endif
#ifdef CFDARCHO_HIP_ENABLE
            if (CFDArcoGlobalInit::hip_enabled) {
                hipMemcpy(data.get(), oth.data.get(), _size * sizeof(T), hipMemcpyDefault);
            }
#endif
        }
#endif
    }

    CudaDataMatrix(size_t size, T const_val) : _size{size} {
#ifndef CFDARCO_SKIP_RMM
        void* ptr = nullptr;
#ifdef CFDARCHO_CUDA_ENABLE
        if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
            ptr = Allocator::cuda_mem_pool->allocate(_size * sizeof(T));
        }
#endif
#ifdef CFDARCHO_HIP_ENABLE
        if (CFDArcoGlobalInit::hip_enabled) {
            hipMalloc(&ptr, _size * sizeof(T));
        }
#endif
        data = std::shared_ptr<T>(static_cast<T *>(ptr), CudaDeleter<T>{_size});

        std::vector<T> copy_mem(_size, const_val);
#ifdef CFDARCHO_CUDA_ENABLE
        if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
            cudaMemcpy(data.get(), copy_mem.data(), _size * sizeof(T), cudaMemcpyHostToDevice);
        }
#endif
#ifdef CFDARCHO_HIP_ENABLE
        if (CFDArcoGlobalInit::hip_enabled) {
            hipMemcpy(data.get(), copy_mem.data(), _size * sizeof(T), hipMemcpyHostToDevice);
        }
#endif
#endif
    }

    Eigen::Vector<T, -1> to_eigen() {
#ifndef CFDARCO_SKIP_RMM
        Eigen::Vector<T, -1> ret{_size};
#ifdef CFDARCHO_CUDA_ENABLE
        if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
            cudaMemcpy(ret.data(), data.get(), _size * sizeof(T), cudaMemcpyDeviceToHost);
        }
#endif
#ifdef CFDARCHO_HIP_ENABLE
        if (CFDArcoGlobalInit::hip_enabled) {
            hipMemcpy(ret.data(), data.get(), _size * sizeof(T), hipMemcpyDeviceToHost);
        }
#endif
        return ret;
#endif
    }

    static CudaDataMatrix from_eigen(MatrixX6T<T> mtrx) {
#ifndef CFDARCO_SKIP_RMM
        auto mtrx_size = mtrx.size();
        CudaDataMatrix ret{static_cast<size_t>(mtrx_size)};
#ifdef CFDARCHO_CUDA_ENABLE
        if (CFDArcoGlobalInit::cuda_enabled && Allocator::allocator_alive) {
            cudaMemcpy(ret.data.get(), mtrx.data(), mtrx_size * sizeof(T), cudaMemcpyHostToDevice);
        }
#endif
#ifdef CFDARCHO_HIP_ENABLE
        if (CFDArcoGlobalInit::hip_enabled) {
            hipMemcpy(ret.data.get(), mtrx.data(), mtrx_size * sizeof(T), hipMemcpyHostToDevice);
        }
#endif
        return ret;
#endif
    }

//    void set(int rows, int cols, int x, int y, int z, float val) {
//#ifndef CFDARCO_SKIP_RMM
//        const size_t idx = x * rows * cols + y * cols + z;
//        cudaMemcpy(data.get() + idx, &val, sizeof(double), cudaMemcpyHostToDevice);
//#endif
//    }

    size_t _size;
    std::shared_ptr<T> data;
};

using CudaDataMatrixD = CudaDataMatrix<float>;


#endif //CFDARCO_CUDA_DATA_MATRIX_HPP
