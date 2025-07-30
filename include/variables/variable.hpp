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

#include "mesh3d.hpp"
#include "decls.hpp"
#include "jit_config.h"

class DT;

using BoundaryFN = std::function<Eigen::Matrix<float, -1, 1>(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr,
                                                             const DT *dt_)>;
using BoundaryFNCU = std::function<void(Mesh3D *mesh, CudaDataMatrixD &arr, const DT *dt_)>;  // inplace boundary!!!


class Variable {
public:
    Variable();

    Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_,
             std::string name_ = "");

    Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_,
             BoundaryFNCU boundary_conditions_cu_, std::string name_ = "");

    Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
             std::function<MatrixX4dRB(MatrixX4dRB &, MatrixX4dRB &)> op_, std::string &name_);

    Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
             std::function<CudaDataMatrixD(CudaDataMatrixD &, CudaDataMatrixD &)> op_, std::string &name_,
             std::vector<std::string> jit_declarations_,
             std::vector<std::tuple<std::string, void *, TypeEnum>> jit_inputs_,
             std::vector<std::string> jit_operations_, std::vector<std::string> jit_memops_,
             std::vector<std::string> jit_returns_);

    Variable(Mesh3D *mesh_, float value);

    Variable(Eigen::Matrix<float, -1, 1> &curr_);

    Variable(Variable &);

    Variable(const Variable &);

    [[nodiscard]] virtual std::shared_ptr<Variable> clone() const;

    void set_bound(const DT *dt_);

    void set_bound_cu(const DT *dt_);

    void add_history();

    virtual std::vector<std::string> get_jit_declarations();

    virtual std::vector<std::tuple<std::string, void *, TypeEnum>> get_jit_inputs();

    virtual std::vector<std::string> get_jit_operations();

    virtual std::vector<std::string> get_jit_memops();

    virtual std::vector<std::string> get_jit_returns() { return jit_returns; }

    virtual Eigen::Matrix<float, -1, 1> extract(Eigen::Matrix<float, -1, 1> &left_part, float dt);

    virtual CudaDataMatrixD extract_cu(CudaDataMatrixD &left_part, float dt);

    virtual MatrixX4dRB evaluate();

    virtual CudaDataMatrixD evaluate_cu();

    void set_current(Eigen::Matrix<float, -1, 1> &current_);

    void set_current(CudaDataMatrixD &current_, bool copy_to_host);

    [[nodiscard]] std::vector<Eigen::Matrix<float, -1, 1>> get_history() const;

    virtual void solve(Variable *equation, DT *dt);

public:
    std::string name;
    Mesh3D *mesh = nullptr;
    Eigen::Matrix<float, -1, 1> current;
    CudaDataMatrixD current_cu{};
    BoundaryFN boundary_conditions;
    BoundaryFNCU boundary_conditions_cu;
    std::vector<Eigen::Matrix<float, -1, 1>> history{};
    size_t num_nodes = 0;
    bool has_boundary_conditions_cu = false;
    bool is_subvariable = false;
    bool is_constvar = false;
    bool is_basically_created = false;
    bool is_dt2 = false;

//    from subvariable
    std::shared_ptr<Variable> left_operand = nullptr;
    std::shared_ptr<Variable> right_operand = nullptr;
    std::function<MatrixX4dRB(MatrixX4dRB &, MatrixX4dRB &)> op;
    std::function<CudaDataMatrixD(CudaDataMatrixD &, CudaDataMatrixD &)> op_cu;

//    JIT strings
    std::vector<std::string> jit_declarations{};
    std::vector<std::tuple<std::string, void *, TypeEnum>> jit_inputs{};
    std::vector<std::string> jit_memops{};
    std::vector<std::string> jit_operations{};
    std::vector<std::string> jit_returns{};
    std::vector<std::string> jit_assign{};

    Variable operator+(const Variable &obj_r) const;

    Variable operator-(const Variable &obj_r) const;

    Variable operator*(const Variable &obj_r) const;

    Variable operator/(const Variable &obj_r) const;

    Variable operator-() const;

    virtual inline void set_dt(float dt) {};

    virtual inline void jit_post() {};

    virtual void reset_input_pointers();

    [[nodiscard]] Variable exp() const;
};


Variable operator+(float obj_l, const Variable &obj_r);

Variable operator-(float obj_l, const Variable &obj_r);

Variable operator*(float obj_l, const Variable &obj_r);

Variable operator/(float obj_l, const Variable &obj_r);

Variable operator+(const Variable &obj_l, float obj_r);

Variable operator-(const Variable &obj_l, float obj_r);

Variable operator*(const Variable &obj_l, float obj_r);

Variable operator/(const Variable &obj_l, float obj_r);

Variable exp(const Variable &obj);

Variable abs(const Variable &obj);
