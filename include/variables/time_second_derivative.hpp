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
#pragma once

#include "variables/variable.hpp"
#include "cuda_operators.hpp"

class D2tVar : public Variable {
public:
    D2tVar(Variable *var_, int);

    Eigen::Matrix<float, -1, 1> extract(Eigen::Matrix<float, -1, 1> &left_part, float dt) override;

    CudaDataMatrixD extract_cu(CudaDataMatrixD &left_part, float dt) override;

    void solve(Variable *equation, DT *dt) override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    inline void set_dt(float dt) override {
        dt_ = dt;
        second_stage = second_stage_next;
        second_stage_next = second_stage_next_next;
        second_stage_next_next = 1;
    }

    inline void jit_post() override {
        sync_device();
        var->set_current(next_current_cu, 0);
    };

    void reset_input_pointers() override;

    std::vector<std::string> get_jit_declarations() override {
        return jit_declarations;
    }

    std::vector<std::string> get_jit_memops() override {
        return jit_memops;
    }

    std::shared_ptr<Variable> var;
    float dt_ = 0;
    size_t second_stage = 0;
    size_t second_stage_next = 0;
    size_t second_stage_next_next = 0;

    Eigen::Matrix<float, -1, 1> pre_current;
    Eigen::Matrix<float, -1, 1> next_current;
    CudaDataMatrixD pre_current_cu;
    CudaDataMatrixD next_current_cu;

};


inline auto d2t(Variable &var) {
    auto varr = new D2tVar(&var, 0);
    varr->name = "d2t(" + var.name + ")";
    return varr;
}

inline auto d2t(Variable &&var) {
    auto varr = new D2tVar(&var, 0);
    varr->name = "d2t(" + var.name + ")";
    return varr;
}