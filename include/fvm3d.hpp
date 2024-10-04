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
#ifndef CFDARCO_FVM3D_HPP
#define CFDARCO_FVM3D_HPP

#include "mesh3d.hpp"
#include "decls.hpp"
#include "jit_config.h"
#include "cuda_operators.hpp"
#include <optional>
#include <tuple>
#include <iostream>
#include <memory>
#include <utility>
#include <mpi.h>

#if USE_GPU
#include <occa.hpp>
#endif

class Variable;

class DT;

using BoundaryFN = std::function<Eigen::Matrix<float, -1, 1>(Mesh3D *mesh, Eigen::Matrix<float, -1, 1> &arr, const DT* dt_)>;
using BoundaryFNCU = std::function<CudaDataMatrixD(Mesh3D *mesh, CudaDataMatrixD &arr, const DT* dt_)>;

using EquationTemplate = std::vector<std::tuple<Variable *, char, Variable, bool>>;

enum class TypeEnum {
    ConstDoublePointer,
    DoublePointer,
    ConstSizeTPointer,
    Boolean,
    SizeT,
    Double,
    Int,
};

class Variable {
public:
    Variable();

    Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_, std::string name_ = "");

    Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_,
             BoundaryFNCU boundary_conditions_cu_, std::string name_ = "");

    Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
             std::function<MatrixX4dRB(MatrixX4dRB &, MatrixX4dRB &)> op_, std::string &name_);

    Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
             std::function<CudaDataMatrixD(CudaDataMatrixD &, CudaDataMatrixD &)> op_, std::string &name_,
             std::vector<std::string> jit_declarations_, std::vector<std::tuple<std::string, void*, TypeEnum>> jit_inputs_,
             std::vector<std::string> jit_operations_, std::vector<std::string> jit_returns_);

    Variable(Mesh3D *mesh_, float value);

    Variable(Eigen::Matrix<float, -1, 1> &curr_);

    Variable(Variable &);

    Variable(const Variable &);

    [[nodiscard]] virtual std::shared_ptr<Variable> clone() const;

    void set_bound(const DT* dt_);

    void set_bound_cu(const DT* dt_);

    void add_history();

    virtual std::vector<std::string> get_jit_declarations();
    virtual std::vector<std::tuple<std::string, void*, TypeEnum>> get_jit_inputs();
    virtual std::vector<std::string> get_jit_operations();
//    std::vector<std::string> get_jit_returns();

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
    CudaDataMatrixD current_cu;
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
    std::vector<std::tuple<std::string, void*, TypeEnum>> jit_inputs{};
    std::vector<std::string> jit_operations{};
    std::vector<std::string> jit_returns{};
    std::string jit_assign{};

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

class UpdatePolicies {
public:
    static float CourantFriedrichsLewy(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewy3D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh);
    static float CourantFriedrichsLewy1D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewy3DCu(float CFL, std::vector<CudaDataMatrixD*> &space_vars, Mesh3D *mesh);
    static float CourantFriedrichsLewy1DCu(float CFL, std::vector<CudaDataMatrixD*> &space_vars, Mesh3D *mesh);

    static float CourantFriedrichsLewyCu(float CFL, std::vector<CudaDataMatrixD> &space_vars, Mesh3D *mesh);

    static inline float constant_dt(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
        return CFL;
    }

    static inline float constant_dt_cu(float CFL, std::vector<CudaDataMatrixD*> &space_vars, Mesh3D *mesh) {
        return CFL;
    }
};

class PointerVariable : public Variable {
public:
    PointerVariable(Mesh3D *mesh_, float *ptr);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    float *_ptr = nullptr;
};


class DT : public Variable {
public:
    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       float CFL_, std::vector<Variable *> space_vars_);

    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       std::function<float(float, std::vector<CudaDataMatrixD*> &, Mesh3D *mesh)> update_fn_cu_, float CFL_,
       std::vector<Variable *> space_vars_);

    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       float value_) : DT(mesh_, std::move(update_fn_), value_, {}) {};

    DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       std::function<float(float, std::vector<CudaDataMatrixD*> &, Mesh3D *mesh)> update_fn_cu_, float value_) : DT(
            mesh_, std::move(update_fn_), std::move(update_fn_cu_), value_, {}) {}

    void update();

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;
    std::vector<std::tuple<std::string, void*, TypeEnum>> get_jit_inputs() override;
    std::vector<std::string> get_jit_operations() override;

    void reset_input_pointers() override;

    std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn;
    std::function<float(float, std::vector<CudaDataMatrixD*> &, Mesh3D *mesh)> update_fn_cu;
    std::vector<Variable *> space_vars;
    float _dt = 0.0;
    float CFL = 0.0;
    bool has_update_fn_cu = false;

    size_t _current_time_step_int = 0;
    float _current_time_dbl = 0.0;
};

class DtVar : public Variable {
public:
    DtVar(Variable *var_, int);

    Eigen::Matrix<float, -1, 1> extract(Eigen::Matrix<float, -1, 1> &left_part, float dt) override;

    CudaDataMatrixD extract_cu(CudaDataMatrixD &left_part, float dt) override;

    void solve(Variable *equation, DT *dt) override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    inline void set_dt(float dt) override {
        dt_ = dt;
    }

    inline void jit_post() override {
        sync_device();
        var->set_current(next_current_cu, 0);
    };

    void reset_input_pointers() override;

    float dt_ = 0;
    std::shared_ptr<Variable> var;
    CudaDataMatrixD next_current_cu;
};

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

    std::shared_ptr<Variable> var;
    float dt_ = 0;
    size_t second_stage = 0;
    size_t second_stage_next = 0;
    size_t second_stage_next_next = 0;

    CudaDataMatrixD pre_current_cu;
    CudaDataMatrixD next_current_cu;

};


class GradVar : public Variable {
public:
    explicit GradVar(Variable *var_, bool clc_x_ = true, bool clc_y_ = true, bool clc_z_ = true);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;
    std::vector<std::tuple<std::string, void*, TypeEnum>> get_jit_inputs() override;
    std::vector<std::string> get_jit_operations() override;

    void reset_input_pointers() override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
    bool clc_z;
};

class Grad2Var : public Variable {
public:
    explicit Grad2Var(Variable *var_, bool clc_x_ = true, bool clc_y_ = true, bool clc_z_ = true);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;
    std::vector<std::tuple<std::string, void*, TypeEnum>> get_jit_inputs() override;
    std::vector<std::string> get_jit_operations() override;

    void reset_input_pointers() override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
    bool clc_z;
};


class StabVar : public Variable {
public:
    explicit StabVar(Variable *var_, bool clc_x_ = true, bool clc_y_ = true, bool clc_z_ = true);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
    bool clc_z;
};

#if USE_GPU
class KernelBuilder {
public:
    explicit KernelBuilder(Variable* left_val_, Variable* expression_) {
        expression = expression_;
//        expression = std::shared_ptr<Variable>{expression_->clone()};
        left_val = left_val_;
    };

    void build();
    void run();

//    std::shared_ptr<Variable> expression;
    Variable* expression;
    Variable* left_val;
    occa::kernel compute_kernel;
    CudaDataMatrixD ret;
};
#endif

inline auto d1t(Variable &var) {
    auto varr = new DtVar(&var, 0);
    varr->name = "d1t(" + var.name + ")";
    return varr;
}

inline auto d1t(Variable &&var) {
    auto varr = new DtVar(&var, 0);
    varr->name = "d1t(" + var.name + ")";
    return varr;
}

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

inline auto d1dx(Variable &var) {
    return GradVar(&var, true, false, false);
}

inline auto d1dx(Variable &&var) {
    return GradVar(&var, true, false, false);
}

inline auto d1dy(Variable &var) {
    return GradVar(&var, false, true, false);
}

inline auto d1dy(Variable &&var) {
    return GradVar(&var, false, true, false);
}

inline auto d1dz(Variable &var) {
    return GradVar(&var, false, false, true);
}

inline auto d1dz(Variable &&var) {
    return GradVar(&var, false, false, true);
}

inline auto d2dx(Variable &var) {
    return Grad2Var(&var, true, false, false);
}

inline auto d2dx(Variable &&var) {
    return Grad2Var(&var, true, false, false);
}

inline auto d2dy(Variable &var) {
    return Grad2Var(&var, false, true, false);
}

inline auto d2dy(Variable &&var) {
    return Grad2Var(&var, false, true, false);
}

inline auto d2dz(Variable &var) {
    return Grad2Var(&var, false, false, true);
}

inline auto d2dz(Variable &&var) {
    return Grad2Var(&var, false, false, true);
}

inline auto lapl(Variable &var) {
//    auto varr = new Grad2Var(&var, true, true, true);
//    return varr;
    return Grad2Var(&var, true, true, true);
}

inline auto lapl(Variable &&var) {
//    auto varr = new Grad2Var(&var, true, true, true);
//    return varr;
    return Grad2Var(&var, true, true, true);
}

inline auto stab_tot(Variable &var) {
    return StabVar(&var, true, true, true);
}

inline auto stab_tot(Variable &&var) {
    return StabVar(&var, true, true, true);
}

inline auto stabx(Variable &var) {
    return StabVar(&var, true, false, false);
}

inline auto stabx(Variable &&var) {
    return StabVar(&var, true, false, false);
}

inline auto staby(Variable &var) {
    return StabVar(&var, false, true, false);
}

inline auto staby(Variable &&var) {
    return StabVar(&var, false, true, false);
}

inline auto stabz(Variable &var) {
    return StabVar(&var, false, false, true);
}

inline auto stabz(Variable &&var) {
    return StabVar(&var, false, false, true);
}



class EqSolver {
public:
    static void solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt);
};


class Equation {
public:
    explicit Equation(size_t timesteps_);

    void
    evaluate(std::vector<Variable *> &all_vars, EquationTemplate &equation_system,
             DT *dt, bool visualize, std::vector<Variable *> store_vars = {}) const;

    size_t timesteps;
};

#endif //CFDARCO_FVM3D_HPP
