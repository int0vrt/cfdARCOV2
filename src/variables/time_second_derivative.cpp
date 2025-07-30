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
#include "variables/time_second_derivative.hpp"
#include "equation.hpp"

std::shared_ptr<Variable> D2tVar::clone() const {
    auto *new_obj = new D2tVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}

D2tVar::D2tVar(Variable *var_, int) {
    var = std::shared_ptr<Variable>{var_, [](Variable *) {}};
    var->is_dt2 = true;
    is_dt2 = true;
    mesh = var_->mesh;
    num_nodes = var->num_nodes;

    pre_current = Eigen::Matrix<float, -1, 1>::Constant(mesh->_num_nodes, 0);
    next_current = Eigen::Matrix<float, -1, 1>::Constant(mesh->_num_nodes, 0);

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        pre_current_cu = CudaDataMatrixD{var->current_cu._size, 0};
        next_current_cu = CudaDataMatrixD{var->current_cu._size, 0};

        jit_declarations = {
                fmt::format("__shared__ __align__(16) float pre_current_{0}_shmem[pipeline_stages][BLOCKSIZE];",
                            var->name),
        };

        jit_memops = {
                fmt::format(
                        "read_n_vars_from_self_pipe<1>(pre_current_{0}_ptr, &(pre_current_{0}_shmem[pipe][0]), idx, n);",
                        var->name),
        };

        jit_assign = {fmt::format(
                "float temp_current_{0} = {0}; \n"
                "float next_current_{0}; \n"
                "float pre_current_{0} = pre_current_{0}_shmem[pipe][threadIdx.x]; \n"
                "if (d2t_second_stage) {{{{ \n"
                "next_current_{0} = dt * dt * {{0}} + 2 * temp_current_{0} - pre_current_{0}; \n"
                "}}}} else {{{{ \n"
                "next_current_{0} = dt * {{0}} + temp_current_{0}; \n"
                "}}}} \n"
                "next_current_{0}_ptr[idx] = next_current_{0}; \n"
                "pre_current_{0}_ptr[idx] = temp_current_{0}; \n"

                "\n", var->name)};
    }
}

Eigen::Matrix<float, -1, 1> D2tVar::extract(Eigen::Matrix<float, -1, 1> &left_part, float dt) {
//    if (var->history.size() > 1) {
//        return dt * dt * left_part + 2 * var->current - var->history.at(var->history.size() - 2);
//    }
//    return dt * left_part + var->current;

    if (second_stage) {
        pre_current = next_current;
        next_current = var->current;

        auto res = left_part * dt * dt + var->current * 2 - pre_current;

        second_stage = second_stage_next;
        second_stage_next = second_stage_next_next;
        second_stage_next_next = 1;

        return res;
    }

    pre_current = next_current;
    next_current = var->current;

    auto res = left_part * dt + var->current;

    second_stage = second_stage_next;
    second_stage_next = second_stage_next_next;
    second_stage_next_next = 1;

    return res;
}

CudaDataMatrixD D2tVar::extract_cu(CudaDataMatrixD &left_part, float dt) {
    if (second_stage) {
        pre_current_cu = mul_mtrx(next_current_cu, 1);
        next_current_cu = mul_mtrx(var->current_cu, 1);

        auto res = mul_mtrx(left_part, dt * dt) + mul_mtrx(var->current_cu, 2) - pre_current_cu;

        second_stage = second_stage_next;
        second_stage_next = second_stage_next_next;
        second_stage_next_next = 1;

        return res;
    }

    pre_current_cu = mul_mtrx(next_current_cu, 1);
    next_current_cu = mul_mtrx(var->current_cu, 1);

    auto res = mul_mtrx(left_part, dt) + var->current_cu;

    second_stage = second_stage_next;
    second_stage_next = second_stage_next_next;
    second_stage_next_next = 1;

    return res;
}

void D2tVar::solve(Variable *equation, DT *dt) {
    EqSolver::solve_dt(equation, this, var.get(), dt);
}

void D2tVar::reset_input_pointers() {
    jit_inputs = {
            {fmt::format("float * __restrict__ next_current_{}_ptr",
                         var->name),                                            &next_current_cu, TypeEnum::DoublePointer},
            {fmt::format("float * __restrict__ pre_current_{}_ptr",
                         var->name),                                            &pre_current_cu,  TypeEnum::DoublePointer},
            {fmt::format("float dt"),                                           &(dt_),           TypeEnum::Double},
            {fmt::format("size_t d2t_second_stage"),                            &second_stage,    TypeEnum::SizeT},
    };
}
