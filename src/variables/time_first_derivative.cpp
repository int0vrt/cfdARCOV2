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
#include "variables/time_first_derivative.hpp"
#include "equation.hpp"

std::shared_ptr<Variable> DtVar::clone() const {
    auto *new_obj = new DtVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}

DtVar::DtVar(Variable *var_, int) {
    var = std::shared_ptr<Variable>{var_, [](Variable *) {}};
    mesh = var_->mesh;
    num_nodes = var->num_nodes;

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        next_current_cu = CudaDataMatrixD{var->current_cu._size};

        jit_declarations = {
                fmt::format("__shared__ __align__(16) float {}_shmem[pipeline_stages][BLOCKSIZE];", var->name),
                fmt::format("float {};", var->name),
        };
        jit_memops = {
                fmt::format("read_n_vars_from_self_pipe<1>({0}_ptr, &({0}_shmem[pipe][0]), idx, n);", var->name),
        };
        jit_operations = {
                fmt::format("{0} = {0}_shmem[pipe][threadIdx.x];", var->name),
        };
        jit_assign = {fmt::format(
                "float temp_current_{0} = {0}; \n"
                "float next_current_{0}; \n"
                "next_current_{0} = dt * {{0}} + temp_current_{0}; \n"
                "next_current_{0}_ptr[idx] = next_current_{0}; \n",
                var->name)};
    }
}

Eigen::Matrix<float, -1, 1> DtVar::extract(Eigen::Matrix<float, -1, 1> &left_part, float dt) {
    return dt * left_part + var->current;
}

CudaDataMatrixD DtVar::extract_cu(CudaDataMatrixD &left_part, float dt) {
    auto res = mul_mtrx(left_part, dt) + var->current_cu;
    return res;
}

void DtVar::solve(Variable *equation, DT *dt) {
    EqSolver::solve_dt(equation, this, var.get(), dt);
}

void DtVar::reset_input_pointers() {
    jit_inputs = {
            {fmt::format("float * __restrict__ next_current_{}_ptr",
                         var->name),                                &next_current_cu, TypeEnum::DoublePointer},
            {fmt::format("float * __restrict__ {}_ptr", var->name), &var->current_cu, TypeEnum::DoublePointer},
            {fmt::format("float dt"),                               &(dt_),           TypeEnum::Double},
    };
}
