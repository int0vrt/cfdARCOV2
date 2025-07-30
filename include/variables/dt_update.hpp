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

class UpdatePolicies {
public:
    static float CourantFriedrichsLewy(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewy3D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewy1D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewy3DCu(float CFL, std::vector<CudaDataMatrixD *> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewy1DCu(float CFL, std::vector<CudaDataMatrixD *> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewyCu(float CFL, std::vector<CudaDataMatrixD> &space_vars, Mesh3D *mesh);

    static inline float constant_dt(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
        return CFL;
    }

    static inline float constant_dt_cu(float CFL, std::vector<CudaDataMatrixD *> &space_vars, Mesh3D *mesh) {
        return CFL;
    }
};

class DT : public Variable {
public:
    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       float CFL_, std::vector<Variable *> space_vars_);

    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       std::function<float(float, std::vector<CudaDataMatrixD *> &, Mesh3D *mesh)> update_fn_cu_, float CFL_,
       std::vector<Variable *> space_vars_);

    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       float value_) : DT(mesh_, std::move(update_fn_), value_, {}) {};

    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       std::function<float(float, std::vector<CudaDataMatrixD *> &, Mesh3D *mesh)> update_fn_cu_, float value_) : DT(
            mesh_, std::move(update_fn_), std::move(update_fn_cu_), value_, {}) {}

    void update();

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;

    std::vector<std::tuple<std::string, void *, TypeEnum>> get_jit_inputs() override;

    std::vector<std::string> get_jit_operations() override;

    std::vector<std::string> get_jit_memops() override;

    void reset_input_pointers() override;

    std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn;
    std::function<float(float, std::vector<CudaDataMatrixD *> &, Mesh3D *mesh)> update_fn_cu;
    std::vector<Variable *> space_vars;
    float _dt = 0.0;
    float CFL = 0.0;
    bool has_update_fn_cu = false;

    size_t _current_time_step_int = 0;
    float _current_time_dbl = 0.0;
};

class CurrTime : public Variable {
public:
    CurrTime(Mesh3D *mesh_, DT *dt_);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;

    std::vector<std::tuple<std::string, void *, TypeEnum>> get_jit_inputs() override;

    std::vector<std::string> get_jit_operations() override;

    std::vector<std::string> get_jit_memops() override;

    void reset_input_pointers() override;

    DT *_dt;
};
