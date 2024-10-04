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
#ifndef CFDARCO_POOL_ALLOCATOR_HPP
#define CFDARCO_POOL_ALLOCATOR_HPP

#ifndef CFDARCO_SKIP_RMM

#include <memory>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <occa.hpp>

class Allocator {
public:
    static std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> cuda_mem_pool;
    static bool allocator_alive;
};

#else

class Allocator {
public:
    static bool allocator_alive;
};

#endif

#endif //CFDARCO_POOL_ALLOCATOR_HPP
