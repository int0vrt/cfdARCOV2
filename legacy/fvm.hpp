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
#ifndef CFDARCO_FVM_HPP
#define CFDARCO_FVM_HPP

#include "mesh2d.hpp"
#include "decls.hpp"
#include "cuda_operators.hpp"
#include <optional>
#include <tuple>
#include <memory>
#include <mpi.h>

class Variable;
//class _SubVariable;
class DT;
class _GradEstimated;

using BoundaryFN = std::function<Eigen::VectorXd(Mesh2D* mesh, Eigen::VectorXd& arr)>;
using BoundaryFNCU = std::function<CudaDataMatrix(Mesh2D* mesh, CudaDataMatrix& arr)>;

using EquationTemplate = std::vector<std::tuple<Variable*, char, Variable>>;

struct Tup3 {
    MatrixX4dRB el1;
    MatrixX4dRB el2;
    MatrixX4dRB el3;
};

class Variable {
public:
    Variable();
    Variable(Mesh2D* mesh_, Eigen::VectorXd& initial_, BoundaryFN boundary_conditions_, std::string name_="");
    Variable(Mesh2D* mesh_, Eigen::VectorXd& initial_, BoundaryFN boundary_conditions_, BoundaryFNCU boundary_conditions_cu_, std::string name_="");
    Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_, std::function<MatrixX4dRB(MatrixX4dRB&, MatrixX4dRB&)> op_, std::string& name_);
    Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_, std::function<CudaDataMatrix(CudaDataMatrix&, CudaDataMatrix&)> op_, std::string& name_);
    Variable(Mesh2D* mesh_, double value);
    Variable(Eigen::VectorXd& curr_);

    Variable(Variable&);
    Variable(const Variable&);
//    Variable(Variable&&) = delete;
//    Variable(const Variable&&) = delete;

    virtual std::shared_ptr<Variable> clone() const;

    void set_bound();
    void set_bound_cu();
    void add_history();
    MatrixX4dRB* estimate_grads();
    std::tuple<CudaDataMatrix, CudaDataMatrix> estimate_grads_cu();
    _GradEstimated dx();
    _GradEstimated dy();
    virtual Tup3* get_interface_vars_first_order();
    std::tuple<CudaDataMatrix, CudaDataMatrix, CudaDataMatrix> get_interface_vars_first_order_cu();
    virtual Tup3* get_interface_vars_second_order();
    std::tuple<CudaDataMatrix, CudaDataMatrix, CudaDataMatrix> get_interface_vars_second_order_cu();
    virtual Eigen::VectorXd extract(Eigen::VectorXd& left_part, double dt);
    virtual CudaDataMatrix extract_cu(CudaDataMatrix& left_part, double dt);
    virtual MatrixX4dRB evaluate();
    virtual CudaDataMatrix evaluate_cu();
    void set_current(Eigen::VectorXd& current_);
    void set_current(CudaDataMatrix& current_, bool copy_to_host);
    std::vector<Eigen::VectorXd> get_history();
    virtual void solve(Variable* equation, DT* dt);

public:
    std::string name;
    Mesh2D *mesh = nullptr;
    Eigen::VectorXd current;
    CudaDataMatrix current_cu;
    std::vector<MatrixX4dRB> current_redist;
    MatrixX4dRB current_redist_mtrx;
    MatrixX4dRB grad_redist_mtrx_x;
    MatrixX4dRB grad_redist_mtrx_y;
    std::vector<MatrixX4dRB> grad_redist;
    std::vector<CudaDataMatrix> current_redist_cu;
    std::vector<std::tuple<CudaDataMatrix, CudaDataMatrix>> grad_redist_cu;
    BoundaryFN boundary_conditions;
    BoundaryFNCU boundary_conditions_cu;
    std::vector<Eigen::VectorXd> history {};
    size_t num_nodes = 0;
    bool has_boundary_conditions_cu = false;
    bool is_subvariable = false;
    bool is_constvar = false;
    bool is_basically_created = false;
    bool is_dt2 = false;

//    from subvariable
    std::shared_ptr<Variable> left_operand = nullptr;
    std::shared_ptr<Variable> right_operand = nullptr;
    std::function<MatrixX4dRB(MatrixX4dRB&, MatrixX4dRB&)> op;
    std::function<CudaDataMatrix(CudaDataMatrix&, CudaDataMatrix&)> op_cu;

//    cache
    bool estimate_grid_cache_valid = false;
    bool get_first_order_cache_valid = false;
    bool get_second_order_cache_valid = false;
    MatrixX4dRB estimate_grid_cache;
    Tup3 get_first_order_cache;
    Tup3 get_second_order_cache;

    std::tuple<CudaDataMatrix, CudaDataMatrix> estimate_grid_cache_cu;
    std::tuple<CudaDataMatrix, CudaDataMatrix, CudaDataMatrix> get_first_order_cache_cu;
    std::tuple<CudaDataMatrix, CudaDataMatrix, CudaDataMatrix> get_second_order_cache_cu;

    Variable operator+(const Variable & obj_r) const;
    Variable operator-(const Variable & obj_r) const;
    Variable operator*(const Variable & obj_r) const;
    Variable operator/(const Variable & obj_r) const;
    Variable operator-() const;
    Variable exp() const;
};

Variable operator+(const double obj_l, const Variable & obj_r);
Variable operator-(const double obj_l, const Variable & obj_r);
Variable operator*(const double obj_l, const Variable & obj_r);
Variable operator/(const double obj_l, const Variable & obj_r);
Variable operator+(const Variable & obj_l, const double obj_r);
Variable operator-(const Variable & obj_l, const double obj_r);
Variable operator*(const Variable & obj_l, const double obj_r);
Variable operator/(const Variable & obj_l, const double obj_r);
Variable exp(const Variable & obj);

class _GradEstimated : public Variable {
public:
    explicit _GradEstimated(Variable *var_, bool clc_x_=true, bool clc_y_=true);

    MatrixX4dRB evaluate() override;
    CudaDataMatrix evaluate_cu() override;
    std::shared_ptr<Variable> clone() const override;

    Variable* var;
    bool clc_x;
    bool clc_y;
};


class UpdatePolicies {
public:
    static double CourantFriedrichsLewy(double CFL, std::vector<Eigen::VectorXd>& space_vars, Mesh2D* mesh);
    static double CourantFriedrichsLewyCu(double CFL, std::vector<CudaDataMatrix>& space_vars, Mesh2D* mesh);

    static inline double constant_dt(double CFL, std::vector<Eigen::VectorXd>& space_vars, Mesh2D* mesh) {
        return CFL;
    }
    static inline double constant_dt_cu(double CFL, std::vector<CudaDataMatrix>& space_vars, Mesh2D* mesh) {
        return CFL;
    }
};

class PointerVariable : public Variable {
public:
    PointerVariable(Mesh2D* mesh_, double* ptr);
    MatrixX4dRB evaluate() override;
    CudaDataMatrix evaluate_cu() override;
    std::shared_ptr<Variable> clone() const override;

    double* _ptr = nullptr;
};


class DT : public Variable {
public:
    DT(Mesh2D* mesh_, std::function<double(double, std::vector<Eigen::VectorXd>&, Mesh2D* mesh)> update_fn_, double CFL_, std::vector<Variable*> space_vars_);
    DT(Mesh2D* mesh_, std::function<double(double, std::vector<Eigen::VectorXd>&, Mesh2D* mesh)> update_fn_,
       std::function<double(double, std::vector<CudaDataMatrix>&, Mesh2D* mesh)>update_fn_cu_, double CFL_, std::vector<Variable*> space_vars_);
    DT(Mesh2D* mesh_, std::function<double(double, std::vector<Eigen::VectorXd>&, Mesh2D* mesh)> update_fn_, double value_) : DT(mesh_, update_fn_, value_, {}) {};
    DT(Mesh2D* mesh_, std::function<double(double, std::vector<Eigen::VectorXd>&, Mesh2D* mesh)> update_fn_,
       std::function<double(double, std::vector<CudaDataMatrix>&, Mesh2D* mesh)>update_fn_cu_, double value_) : DT(mesh_, update_fn_, update_fn_cu_, value_, {}) {}
    void update();
    MatrixX4dRB evaluate() override;
    CudaDataMatrix evaluate_cu() override;
    std::shared_ptr<Variable> clone() const override;

    std::function<double(double, std::vector<Eigen::VectorXd>&, Mesh2D* mesh)> update_fn;
    std::function<double(double, std::vector<CudaDataMatrix>&, Mesh2D* mesh)> update_fn_cu;
    std::vector<Variable*> space_vars;
    double _dt = 0.0;
    double CFL = 0.0;
    bool has_update_fn_cu = false;

    size_t _current_time_step_int = 0;
    double _current_time_dbl = 0.0;
};

class Variable2d : Variable {
public:
    using Variable::Variable;
    using Variable::current;
    using Variable::operator*;
    using Variable::operator+;
    using Variable::operator-;
    using Variable::operator/;
};


class _DT : public Variable {
public:
    _DT(Variable* var_, int);

    Eigen::VectorXd extract(Eigen::VectorXd& left_part, double dt) override;
    CudaDataMatrix extract_cu(CudaDataMatrix& left_part, double dt) override;
    void solve(Variable* equation, DT* dt) override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
};

class _D2T : public Variable {
public:
    _D2T(Variable* var_, int);

    Eigen::VectorXd extract(Eigen::VectorXd& left_part, double dt) override;
    CudaDataMatrix extract_cu(CudaDataMatrix& left_part, double dt) override;
    void solve(Variable* equation, DT* dt) override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
};


class _Grad : public Variable {
public:
    _Grad(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    MatrixX4dRB evaluate() override;
    CudaDataMatrix evaluate_cu() override;
    std::shared_ptr<Variable> clone() const override;
    Tup3* get_interface_vars_first_order() override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
};

class _Grad2 : public Variable {
public:
    _Grad2(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    MatrixX4dRB evaluate() override;
    CudaDataMatrix evaluate_cu() override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
};


class _Stab : public Variable {
public:
    _Stab(Variable* var_, bool clc_x_=1, bool clc_y_=1);

    MatrixX4dRB evaluate() override;
    CudaDataMatrix evaluate_cu() override;
    std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
};


inline auto d1t(Variable& var) {
    auto varr = new _DT(&var, 0);
    varr->name = "d1t(" + var.name + ")";
    return varr;
}

inline auto d1t(Variable&& var) {
    auto varr = new _DT(&var, 0);
    varr->name = "d1t(" + var.name + ")";
    return varr;
}

inline auto d2t(Variable& var) {
    auto varr = new _D2T(&var, 0);
    varr->name = "d2t(" + var.name + ")";
    return varr;
}

inline auto d2t(Variable&& var) {
    auto varr = new _D2T(&var, 0);
    varr->name = "d2t(" + var.name + ")";
    return varr;
}

inline auto d1dx(Variable& var) {
    return _Grad(&var, true, false);
}

inline auto d1dx(Variable&& var) {
    return _Grad(&var, true, false);
}

inline auto d1dy(Variable& var) {
    return _Grad(&var,  false, true);
}

inline auto d1dy(Variable&& var) {
    return _Grad(&var,  false, true);
}

inline auto d2dx(Variable& var) {
    return _Grad2(&var, true, false);
}

inline auto d2dx(Variable&& var) {
    return _Grad2(&var, true, false);
}

inline auto d2dy(Variable& var) {
    return _Grad2(&var,  false, true);
}

inline auto d2dy(Variable&& var) {
    return _Grad2(&var,  false, true);
}

inline auto lapl(Variable& var) {
    return _Grad2(&var, true, true);
}

inline auto lapl(Variable&& var) {
    return _Grad2(&var, true, true);
}

inline auto stab_x(Variable& var) {
    return _Stab(&var, true, false);
}

inline auto stab_x(Variable&& var) {
    return _Stab(&var, true, false);
}

inline auto stab_y(Variable& var) {
    return _Stab(&var,  false, true);
}

inline auto stab_y(Variable&& var) {
    return _Stab(&var,  false, true);
}

inline auto stab_tot(Variable& var) {
    return _Stab(&var, true, true);
}

inline auto stab_tot(Variable&& var) {
    return _Stab(&var, true, true);
}


class EqSolver {
public:
    static void solve_dt(Variable* equation, Variable* time_var, Variable* set_var, DT* dt);
};


class Equation {
public:
    Equation(size_t timesteps_);
    void evaluate(std::vector<Variable*>&all_vars, std::vector<std::tuple<Variable*, char, Variable>>&equation_system, DT* dt, bool visualize, std::vector<Variable*> store_vars = {});

    size_t timesteps;
};


MatrixX4dRB to_grid(const Mesh2D* mesh, Eigen::VectorXd& values);
MatrixX4dRB to_grid_local(const Mesh2D* mesh, Eigen::VectorXd& values);
Eigen::VectorXd from_grid(const Mesh2D* mesh, MatrixX4dRB& grid);

template<typename Scalar, typename Matrix>
inline static std::vector< std::vector<Scalar> > from_eigen_matrix( const Matrix & M ){
    std::vector< std::vector<Scalar> > m;
    m.resize(M.rows(), std::vector<Scalar>(M.cols(), 0));
    for(size_t i = 0; i < m.size(); i++)
        for(size_t j = 0; j < m.front().size(); j++)
            m[i][j] = M(i,j);
    return m;
}


#endif //CFDARCO_FVM_HPP
