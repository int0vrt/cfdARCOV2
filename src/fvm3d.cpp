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
// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com


#include "fvm3d.hpp"
#include "cfdarcho_main_3d.hpp"
#include "grad_utils.hpp"

#include <utility>
#include <fmt/core.h>
#include <indicators/progress_bar.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <jit_config.h>

Variable::Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
    has_boundary_conditions_cu = false;
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
        jit_declarations = {
                fmt::format("size_t face_neigh_ids[{}];", Mesh3D::n_faces),
                fmt::format("float {};", name),
                fmt::format("float {}_neigh[{}];", name, Mesh3D::n_faces)
        };
        jit_operations = {
                fmt::format("{0} = {0}_ptr[idx];", name),

                fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(ids_ptr, face_neigh_ids, idx, n);"),

                fmt::format("read_n_vars_from_neigh<Mesh3D>({0}_ptr, face_neigh_ids, {0}_neigh);", name),

        };
        jit_returns = { name };
        jit_assign = fmt::format("{}_ptr[idx] = {{}};", name);
    }

}

Variable::Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_,
                   BoundaryFNCU boundary_conditions_cu_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)},
        boundary_conditions_cu{std::move(boundary_conditions_cu_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
    has_boundary_conditions_cu = true;
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
        jit_declarations = {
                fmt::format("size_t face_neigh_ids[{}];", Mesh3D::n_faces),
                fmt::format("float {};", name),
                fmt::format("float {}_neigh[{}];", name, Mesh3D::n_faces)
        };
        jit_operations = {
                fmt::format("{0} = {0}_ptr[idx];", name),

                fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(ids_ptr, face_neigh_ids, idx, n);"),

                fmt::format("read_n_vars_from_neigh<Mesh3D>({0}_ptr, face_neigh_ids, {0}_neigh);", name),

        };
        jit_returns = { name };
        jit_assign = fmt::format("{}_ptr[idx] = {{}};", name);
    }
}

Variable::Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
                   std::function<MatrixX4dRB(MatrixX4dRB &, MatrixX4dRB &)> op_, std::string &name_) :
        op{std::move(op_)}, name{name_} {
    num_nodes = 0;
    if (left_operand_->mesh != nullptr) {
        mesh = left_operand_->mesh;
        num_nodes = mesh->_num_nodes;
    } else if (right_operand_->mesh != nullptr) {
        mesh = right_operand_->mesh;
        num_nodes = mesh->_num_nodes;
    }
    is_subvariable = true;
    is_basically_created = false;
    left_operand = left_operand_;
    right_operand = right_operand_;
}

Variable::Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
                   std::function<CudaDataMatrixD(CudaDataMatrixD &, CudaDataMatrixD &)> op_, std::string &name_,
                   std::vector<std::string> jit_declarations_, std::vector<std::tuple<std::string, void*, TypeEnum>> jit_inputs_,
                   std::vector<std::string> jit_operations_, std::vector<std::string> jit_returns_) :
        op_cu{std::move(op_)}, name{name_}, jit_declarations{std::move(jit_declarations_)}, jit_inputs{std::move(jit_inputs_)},
        jit_operations{std::move(jit_operations_)}, jit_returns{std::move(jit_returns_)} {
    num_nodes = 0;
    if (left_operand_->mesh != nullptr) {
        mesh = left_operand_->mesh;
        num_nodes = mesh->_num_nodes;
    } else if (right_operand_->mesh != nullptr) {
        mesh = right_operand_->mesh;
        num_nodes = mesh->_num_nodes;
    }
    is_subvariable = true;
    is_basically_created = false;
    left_operand = left_operand_;
    right_operand = right_operand_;
}

Variable::Variable(Mesh3D *mesh_, float value) : mesh{mesh_} {
    current = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    current.setConstant(value);

    num_nodes = mesh->_num_nodes;
    std::string var_name = fmt::format("constval_{}", value);
    var_name.erase(std::remove(var_name.begin(), var_name.end(), '.'), var_name.end());
    name = var_name;
    is_constvar = true;
    is_basically_created = false;

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
        jit_declarations = {
                fmt::format("const float {} = {};", var_name, value),
                fmt::format("const float {0}_neigh[{1}] = {{ {2}, {2}, {2}, {2}, {2}, {2} }};", name, Mesh3D::n_faces, value)
        };

        jit_inputs = {};
        jit_operations = {};
        jit_returns = { fmt::format("{}", var_name) };
    }

}

Variable::Variable() {
    current = {};
    boundary_conditions = {};
    name = "uninitialized";
    num_nodes = 0;
    is_basically_created = false;
}

Variable::Variable(Eigen::Matrix<float, -1, 1> &curr_) {
    current = curr_;
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)
        current_cu = CudaDataMatrixD::from_eigen(current);
    num_nodes = curr_.rows();
    name = "arr";
    is_constvar = true;
    is_basically_created = false;
}

Variable::Variable(Variable &copy_var) {
    name = copy_var.name;
    mesh = copy_var.mesh;
    current = copy_var.current;
    current_cu = copy_var.current_cu;
    boundary_conditions = copy_var.boundary_conditions;
    boundary_conditions_cu = copy_var.boundary_conditions_cu;
    has_boundary_conditions_cu = copy_var.has_boundary_conditions_cu;
    history = copy_var.history;
    num_nodes = copy_var.num_nodes;
    is_subvariable = copy_var.is_subvariable;
    is_constvar = copy_var.is_constvar;
    op = copy_var.op;
    op_cu = copy_var.op_cu;
    is_dt2 = copy_var.is_dt2;
    is_basically_created = false;

    jit_declarations = copy_var.jit_declarations;
    jit_inputs = copy_var.jit_inputs;
    jit_operations = copy_var.jit_operations;
    jit_returns = copy_var.jit_returns;
    jit_assign = copy_var.jit_assign;

    if (copy_var.left_operand) {
        left_operand = std::shared_ptr<Variable>{copy_var.left_operand->clone()};
    }
    if (copy_var.right_operand) {
        right_operand = std::shared_ptr<Variable>{copy_var.right_operand->clone()};
    }
}

Variable::Variable(const Variable &copy_var) {
    name = copy_var.name;
    mesh = copy_var.mesh;
    current = copy_var.current;
    current_cu = copy_var.current_cu;
    boundary_conditions = copy_var.boundary_conditions;
    boundary_conditions_cu = copy_var.boundary_conditions_cu;
    has_boundary_conditions_cu = copy_var.has_boundary_conditions_cu;
    history = copy_var.history;
    num_nodes = copy_var.num_nodes;
    is_subvariable = copy_var.is_subvariable;
    is_constvar = copy_var.is_constvar;
    op = copy_var.op;
    op_cu = copy_var.op_cu;
    is_dt2 = copy_var.is_dt2;
    is_basically_created = false;

    jit_declarations = copy_var.jit_declarations;
    jit_inputs = copy_var.jit_inputs;
    jit_operations = copy_var.jit_operations;
    jit_returns = copy_var.jit_returns;
    jit_assign = copy_var.jit_assign;

    if (copy_var.left_operand) {
        left_operand = std::shared_ptr<Variable>{copy_var.left_operand->clone()};
    }
    if (copy_var.right_operand) {
        right_operand = std::shared_ptr<Variable>{copy_var.right_operand->clone()};
    }
}

std::shared_ptr<Variable> Variable::clone() const {
    if (is_basically_created) {
        return std::shared_ptr<Variable>{const_cast<Variable *>(this), [](Variable *) {}};
    }
    auto *new_obj = new Variable(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> PointerVariable::clone() const {
    return std::shared_ptr<Variable>{const_cast<PointerVariable *>(this), [](Variable *) {}};
}

std::shared_ptr<Variable> DT::clone() const {
    return std::shared_ptr<Variable>{const_cast<DT *>(this), [](Variable *) {}};
}

std::shared_ptr<Variable> DtVar::clone() const {
    auto *new_obj = new DtVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> D2tVar::clone() const {
    auto *new_obj = new D2tVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> GradVar::clone() const {
    auto *new_obj = new GradVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> Grad2Var::clone() const {
    auto *new_obj = new Grad2Var(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> StabVar::clone() const {
    auto *new_obj = new StabVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}


void Variable::set_bound(const DT* dt_) {
    current = boundary_conditions(mesh, current, dt_);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
    }
}

void Variable::set_bound_cu(const DT* dt_) {
    current_cu = boundary_conditions_cu(mesh, current_cu, dt_);
    if (CFDArcoGlobalInit::get_size() > 1) {
        current = current_cu.to_eigen();
    }
}

void Variable::add_history() {
//    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled && CFDArcoGlobalInit::skip_history && CFDArcoGlobalInit::store_stepping) {
//        current = current_cu.to_eigen(num_nodes, 1);
//    }
//
//    if (CFDArcoGlobalInit::skip_history && !is_dt2) {
//        return;
//    }
//
//    if (CFDArcoGlobalInit::skip_history && is_dt2) {
//        history.push_back({current});
//
//        if (history.size() > 3) {
//            history.erase(history.begin());
//        }
//
//        return;
//    }
    if (!CFDArcoGlobalInit::skip_history) {
        if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
            current = current_cu.to_eigen();
        }
        history.push_back({current});
    }

}

Eigen::Matrix<float, -1, 1> Variable::extract(Eigen::Matrix<float, -1, 1> &left_part, float dt) {
    return left_part;
}

CudaDataMatrixD Variable::extract_cu(CudaDataMatrixD &left_part, float dt) {
    return left_part;
}

MatrixX4dRB Variable::evaluate() {
    if (!is_subvariable) {
        return current;
    }

    auto val_l = left_operand->evaluate();
    auto val_r = right_operand->evaluate();
    return op(val_l, val_r);
}

CudaDataMatrixD Variable::evaluate_cu() {
    if (!is_subvariable) {
        return current_cu;
    }

    auto val_l = left_operand->evaluate_cu();
    auto val_r = right_operand->evaluate_cu();
    return op_cu(val_l, val_r);
}

void Variable::set_current(Eigen::Matrix<float, -1, 1> &current_) {
    current = current_;
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
    }
}

void Variable::set_current(CudaDataMatrixD &current_, bool copy_to_host) {
    current_cu = current_;
    if (copy_to_host) {
        current = current_.to_eigen();
    }
}

std::vector<Eigen::Matrix<float, -1, 1>> Variable::get_history() const {
    return history;
}

void Variable::solve(Variable *equation, DT *dt) {
    EqSolver::solve_dt(equation, this, this, dt);
}

std::tuple<std::shared_ptr<Variable>, std::shared_ptr<Variable>>
get_that_vars(const Variable *obj_l, const Variable &obj_r) {
    std::shared_ptr<Variable> l_p;
    std::shared_ptr<Variable> r_p;
    if (obj_l->is_subvariable || obj_l->is_constvar) {
        l_p = std::shared_ptr<Variable>{obj_l->clone()};
    } else {
        l_p = std::shared_ptr<Variable>(const_cast<Variable *>(obj_l), [](Variable *) {});
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = std::shared_ptr<Variable>{obj_r.clone()};
    } else {
        r_p = std::shared_ptr<Variable>(const_cast<Variable *>(&obj_r), [](Variable *) {});
    }

    return {l_p, r_p};
}

#define BINARY_OP(OPER_SiGN, OPER_NAME, OPER_SiGN_STR, OPER_EIGEN) Variable Variable::operator OPER_SiGN (const Variable &obj_r) const { \
    std::string name_ = this->name + "_" OPER_NAME "_" + obj_r.name; \
    auto [l_p, r_p] = get_that_vars(this, obj_r); \
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) { \
        std::vector<std::string> jit_declarations_{ \
            fmt::format("float {}_" OPER_NAME "_{};", this->name, obj_r.name), \
            fmt::format("float {}_" OPER_NAME "_{}_neigh[{}];", this->name, obj_r.name, Mesh3D::n_faces), \
        }; \
        std::vector<std::tuple<std::string, void*, TypeEnum>> jit_inputs_{}; \
        std::vector<std::string> jit_operations_{ \
            fmt::format("{}_" OPER_NAME "_{} = {} " OPER_SiGN_STR " {};", this->name, obj_r.name, this->name, obj_r.name), \
            fmt::format("{}_" OPER_NAME "_{}_neigh[0] = {}_neigh[0] " OPER_SiGN_STR " {}_neigh[0];", this->name, obj_r.name, this->name, obj_r.name), \
            fmt::format("{}_" OPER_NAME "_{}_neigh[1] = {}_neigh[1] " OPER_SiGN_STR " {}_neigh[1];", this->name, obj_r.name, this->name, obj_r.name), \
            fmt::format("{}_" OPER_NAME "_{}_neigh[2] = {}_neigh[2] " OPER_SiGN_STR " {}_neigh[2];", this->name, obj_r.name, this->name, obj_r.name), \
            fmt::format("{}_" OPER_NAME "_{}_neigh[3] = {}_neigh[3] " OPER_SiGN_STR " {}_neigh[3];", this->name, obj_r.name, this->name, obj_r.name), \
            fmt::format("{}_" OPER_NAME "_{}_neigh[4] = {}_neigh[4] " OPER_SiGN_STR " {}_neigh[4];", this->name, obj_r.name, this->name, obj_r.name), \
            fmt::format("{}_" OPER_NAME "_{}_neigh[5] = {}_neigh[5] " OPER_SiGN_STR " {}_neigh[5];", this->name, obj_r.name, this->name, obj_r.name), \
        }; \
        std::vector<std::string> jit_returns_{ fmt::format("{}_" OPER_NAME "_{}", this->name, obj_r.name) }; \
        return {l_p, r_p, [](CudaDataMatrixD &lft, CudaDataMatrixD &rht) { return lft OPER_SiGN rht; }, name_, jit_declarations_, jit_inputs_, jit_operations_, jit_returns_}; \
    } \
    return {l_p, r_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return OPER_EIGEN; }, name_}; \
}

BINARY_OP(+, "plus", "+", lft + rht)
BINARY_OP(-, "minus", "-", lft - rht)
BINARY_OP(*, "mult", "*", lft.cwiseProduct(rht))
BINARY_OP(/, "div", "/", lft.cwiseQuotient(rht))

Variable Variable::operator-() const {
    std::string name_ = "neg_" + this->name;
    auto [l_p, r_p] = get_that_vars(this, *this);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        std::vector<std::string> jit_declarations_{
            fmt::format("float neg_{};", this->name),
            fmt::format("float neg_{}_neigh[{}];", this->name, Mesh3D::n_faces),
            };
        std::vector<std::tuple<std::string, void*, TypeEnum>> jit_inputs_{};
        std::vector<std::string> jit_operations_{
            fmt::format("neg_{} = -{};", this->name, this->name),
            fmt::format("neg_{}_neigh[0] = -{}_neigh[0];", this->name, this->name),
            fmt::format("neg_{}_neigh[1] = -{}_neigh[1];", this->name, this->name),
            fmt::format("neg_{}_neigh[2] = -{}_neigh[2];", this->name, this->name),
            fmt::format("neg_{}_neigh[3] = -{}_neigh[3];", this->name, this->name),
            fmt::format("neg_{}_neigh[4] = -{}_neigh[4];", this->name, this->name),
            fmt::format("neg_{}_neigh[5] = -{}_neigh[5];", this->name, this->name),
            };
        std::vector<std::string> jit_returns_{ fmt::format("neg_{}", this->name) };

        return {l_p, r_p, [](CudaDataMatrixD &lft, CudaDataMatrixD &rht) { return -lft; }, name_, jit_declarations_, jit_inputs_, jit_operations_, jit_returns_};
    }
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return -lft; }, name_};
}

Variable operator+(const float obj_l, const Variable &obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l + obj_r;
}

Variable operator-(const float obj_l, const Variable &obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l - obj_r;
}

Variable operator*(const float obj_l, const Variable &obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l * obj_r;
}

Variable operator/(const float obj_l, const Variable &obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l / obj_r;
}

Variable operator+(const Variable &obj_l, const float obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l + val_r;
}

Variable operator-(const Variable &obj_l, const float obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l - val_r;
}

Variable operator*(const Variable &obj_l, const float obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l * val_r;
}

Variable operator/(const Variable &obj_l, const float obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l / val_r;
}

Variable Variable::exp() const {
    std::string name_ = "-(" + this->name + ")";
    auto [l_p, r_p] = get_that_vars(this, *this);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)
        throw std::runtime_error("Not implemented");
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return lft.exp(); }, name_};
}

Variable exp(const Variable &obj) {
    std::string name_ = "-(" + obj.name + ")";
    auto [l_p, r_p] = get_that_vars(&obj, obj);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)
        throw std::runtime_error("Not implemented");
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return Eigen::exp(lft.array()); }, name_};
}

Variable abs(const Variable &obj) {
    std::string name_ = "abs(" + obj.name + ")";
    auto [l_p, r_p] = get_that_vars(&obj, obj);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)
        throw std::runtime_error("Not implemented");
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return Eigen::abs(lft.array()); }, name_};
}

// TODO: think about general interface
float UpdatePolicies::CourantFriedrichsLewy(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
    auto &u = space_vars.at(0);
    auto &v = space_vars.at(1);
    auto &p = space_vars.at(2);
    auto &rho = space_vars.at(3);
    auto gamma = 5. / 3.;
    float dl = std::min({mesh->_dx, mesh->_dy, mesh->_dz});
    auto denom = dl * (((gamma * p.array()).cwiseQuotient(rho.array())).cwiseSqrt() +
                       (u.array() * u.array() + v.array() * v.array()).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return dt;
}

float UpdatePolicies::CourantFriedrichsLewy3D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
    auto &u = space_vars.at(0);
    auto &v = space_vars.at(1);
    auto &w = space_vars.at(2);
    auto &p = space_vars.at(3);
    auto &rho = space_vars.at(4);
    auto gamma = 5. / 3.;
    float dl = std::min({mesh->_dx, mesh->_dy, mesh->_dz});
    auto denom = dl * (((gamma * p.array()).cwiseQuotient(rho.array())).cwiseSqrt() +
                       (u.array() * u.array() + v.array() * v.array() +
                        w.array() * w.array()).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return dt;
}

float UpdatePolicies::CourantFriedrichsLewy3DCu(float CFL, std::vector<CudaDataMatrixD*> &space_vars, Mesh3D *mesh) {
    CudaDataMatrixD u = *(space_vars.at(0));
    CudaDataMatrixD v = *(space_vars.at(1));
    CudaDataMatrixD w = *(space_vars.at(2));
    CudaDataMatrixD p = *(space_vars.at(3));
    CudaDataMatrixD rho = *(space_vars.at(4));
    auto gamma = 5. / 3.;
    float dl = std::min(std::min(mesh->_dx, mesh->_dy), mesh->_dz);
    auto denom = cfl_cu(dl, gamma, p, rho, u, v, w);
    auto dt = CFL * denom;

    return dt;
}

float UpdatePolicies::CourantFriedrichsLewy1D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
    auto &u = space_vars.at(0);
    auto &p = space_vars.at(1);
    auto &rho = space_vars.at(2);
    auto gamma = 5. / 3.;
    float dl = std::min({mesh->_dx, mesh->_dy, mesh->_dz});
    auto denom = dl * (((gamma * p.array()).cwiseQuotient(rho.array())).cwiseSqrt() +
                       (u.array() * u.array()).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return dt;
}

float UpdatePolicies::CourantFriedrichsLewy1DCu(float CFL, std::vector<CudaDataMatrixD*> &space_vars, Mesh3D *mesh) {
    auto u = *space_vars.at(0);
    auto p = *space_vars.at(1);
    auto rho = *space_vars.at(2);
    auto gamma = 5. / 3.;
    float dl = std::min(std::min(mesh->_dx, mesh->_dy), mesh->_dz);
    auto denom = cfl_cu(dl, gamma, p, rho, u);
    auto dt = CFL * denom;

    return dt;
}


PointerVariable::PointerVariable(Mesh3D *mesh_, float *ptr) : _ptr{ptr} {
    name = "PointerVariable";
    mesh = mesh_;
}

MatrixX4dRB PointerVariable::evaluate() {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    ret.setConstant(*_ptr);
    return ret;
}

CudaDataMatrixD PointerVariable::evaluate_cu() {
    return CudaDataMatrixD{mesh->_num_nodes, *_ptr};
}

DT::DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       float CFL_, std::vector<Variable *> space_vars_) : update_fn{std::move(update_fn_)}, CFL{CFL_},
                                                           space_vars{std::move(space_vars_)} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;

    auto cu_mesh = dynamic_cast<CudaMesh3D*>(mesh);

    jit_declarations = {
            fmt::format("float dt_neigh[{0}] = {{ dt, dt, dt, dt, dt, dt }};", Mesh3D::n_faces),
    };
    jit_operations = {};
    jit_returns = { name };
}

DT::DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       std::function<float(float, std::vector<CudaDataMatrixD*> &, Mesh3D *mesh)> update_fn_cu_, float CFL_,
       std::vector<Variable *> space_vars_) : update_fn{std::move(update_fn_)}, update_fn_cu{std::move(update_fn_cu_)},
                                              CFL{CFL_}, space_vars{std::move(space_vars_)} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;
    has_update_fn_cu = true;

    jit_declarations = {
            fmt::format("float dt_neigh[{0}] = {{ dt, dt, dt, dt, dt, dt }};", Mesh3D::n_faces),
    };
    jit_operations = {};
    jit_returns = { name };
}

std::vector<std::string> DT::get_jit_declarations() {
    return jit_declarations;
}

std::vector<std::tuple<std::string, void*, TypeEnum>> DT::get_jit_inputs() {
    reset_input_pointers();

    std::vector<std::tuple<std::string, void*, TypeEnum>> ret = {};
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> DT::get_jit_operations() {
    return {};
}

void DT::reset_input_pointers() {
    auto cu_mesh = dynamic_cast<CudaMesh3D*>(mesh);
    jit_inputs = {
            {fmt::format("float dt"), &(_dt), TypeEnum::Double},
    };
}

void DT::update() {
    float dt_c = 0.0;
    if (has_update_fn_cu && CFDArcoGlobalInit::cuda_enabled) {
        std::vector<CudaDataMatrixD*> redist{};
        for (auto& var: space_vars) {
            redist.push_back(&var->current_cu);
        }
        dt_c = update_fn_cu(CFL, redist, mesh);
    } else {
        std::vector<Eigen::Matrix<float, -1, 1>> redist{};
        for (auto var: space_vars) {
            redist.push_back(var->current);
        }
        dt_c = update_fn(CFL, redist, mesh);
    }

//    MPI_Allreduce(&dt_c, &_dt, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    _dt = dt_c;

    _current_time_step_int++;
    _current_time_dbl += _dt;
}

MatrixX4dRB DT::evaluate() {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    ret.setConstant(_dt);
    return ret;
}

CudaDataMatrixD DT::evaluate_cu() {
    return CudaDataMatrixD{mesh->_num_nodes, _dt};
}


DtVar::DtVar(Variable *var_, int) {
    var = std::shared_ptr<Variable>{var_, [](Variable *) {}};

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        next_current_cu = CudaDataMatrixD{var->current_cu._size};

        jit_assign = fmt::format(
                "float temp_current_{0} = {0}; \n"
                "float next_current_{0}; \n"
                "next_current_{0} = dt * {{0}} + temp_current_{0}; \n"
                "next_current_{0}_ptr[idx] = next_current_{0}; \n",
                var->name);
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
            {fmt::format("float * __restrict__ next_current_{}_ptr", var->name), &next_current_cu, TypeEnum::DoublePointer},
            {fmt::format("float dt"), &(dt_), TypeEnum::Double},
    };
}

D2tVar::D2tVar(Variable *var_, int) {
    var = std::shared_ptr<Variable>{var_, [](Variable *) {}};
    var->is_dt2 = true;

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        pre_current_cu = CudaDataMatrixD{var->current_cu._size, 0};
        next_current_cu = CudaDataMatrixD{var->current_cu._size, 0};


        jit_assign = fmt::format(
                "float temp_current_{0} = {0}; \n"
                "float next_current_{0}; \n"
                "float pre_current_{0} = pre_current_{0}_ptr[idx]; \n"
                "if (d2t_second_stage) {{{{ \n"
                "next_current_{0} = dt * dt * {{0}} + 2 * temp_current_{0} - pre_current_{0}; \n"
                "}}}} else {{{{ \n"
                "next_current_{0} = dt * {{0}} + temp_current_{0}; \n"
                "}}}} \n"
                "next_current_{0}_ptr[idx] = next_current_{0}; \n"
                "pre_current_{0}_ptr[idx] = temp_current_{0}; \n"

                "\n", var->name);
    }
}

Eigen::Matrix<float, -1, 1> D2tVar::extract(Eigen::Matrix<float, -1, 1> &left_part, float dt) {
    if (var->history.size() > 1) {
        return dt * dt * left_part + 2 * var->current - var->history.at(var->history.size() - 2);
    }
    return dt * left_part + var->current;
}

CudaDataMatrixD D2tVar::extract_cu(CudaDataMatrixD &left_part, float dt) {
    if (var->history.size() > 1) {
        auto res = mul_mtrx(left_part, dt * dt) + mul_mtrx(var->current_cu, 2) -
                   CudaDataMatrixD::from_eigen(var->history.at(var->history.size() - 2));
        return res;
    }

    auto res = mul_mtrx(left_part, dt) + var->current_cu;
    return res;
}

void D2tVar::solve(Variable *equation, DT *dt) {
    EqSolver::solve_dt(equation, this, var.get(), dt);
}

void D2tVar::reset_input_pointers() {

    jit_inputs = {
            {fmt::format("float * __restrict__ next_current_{}_ptr", var->name), &next_current_cu, TypeEnum::DoublePointer},
            {fmt::format("float * __restrict__ pre_current_{}_ptr", var->name), &pre_current_cu, TypeEnum::DoublePointer},
            {fmt::format("float dt"), &(dt_), TypeEnum::Double},
            {fmt::format("size_t d2t_second_stage"), &second_stage, TypeEnum::SizeT},
    };
}


GradVar::GradVar(Variable *var_, bool clc_x_, bool clc_y_, bool clc_z_) : clc_x{clc_x_}, clc_y{clc_y_}, clc_z{clc_z_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    num_nodes = var->num_nodes;
    name = "Grad_" + std::to_string(clc_x_) + std::to_string(clc_y_) + std::to_string(clc_z_) + var_->name;
    is_subvariable = true;

    auto cu_mesh = dynamic_cast<CudaMesh3D*>(var_->mesh);

    jit_declarations = {
            fmt::format("float interpolation_ret_{}[{}];", var_->name, Mesh3D::n_faces),
            fmt::format("size_t face_neigh_ids[{}];", Mesh3D::n_faces),
            fmt::format("float face_area[{}];", Mesh3D::n_faces),
            fmt::format("float normals_alt_face[{}][{}];", Mesh3D::n_dims, Mesh3D::n_faces),
            fmt::format("const float* normals_alt_ptr[{}] = {{\n"
                        "            normals_alt_x_ptr,\n"
                        "            normals_alt_y_ptr,\n"
                        "            normals_alt_z_ptr\n"
                        "    }};", Mesh3D::n_dims),
            fmt::format("float grad_{}[{}];", name, Mesh3D::n_dims),
            fmt::format("float volume;"),
            fmt::format("float {} = 0;", name),
            fmt::format("float {}_neigh[{}] = {{ 0, 0, 0, 0, 0, 0 }};", name, Mesh3D::n_faces),
    };
    jit_operations = {
            fmt::format("volume = read_scalar_from_self(volume_ptr, idx, n);"),
            fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(ids_ptr, face_neigh_ids, idx, n);"),
            fmt::format("read_normals_from_self<Mesh3D>(normals_alt_ptr, normals_alt_face, idx, n);"),
            fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(face_area_ptr, face_area, idx, n);"),
            fmt::format("interpolate_to_face_linear_cu_k<Mesh3D>({0}, {0}_neigh, interpolation_ret_{0});", var_->name),
            fmt::format("gauss_grad_cu_k<Mesh3D>(interpolation_ret_{0}, normals_alt_face, face_area, volume, grad_{1});", var_->name, name),
            fmt::format("{{"
                        "        if ({1}) {0} += grad_{0}[0];\n"
                        "        if ({2}) {0} += grad_{0}[1];\n"
                        "        if ({3}) {0} += grad_{0}[2];\n"
                        "}}", name, clc_x_, clc_y_, clc_z_)
    };
    jit_returns = { name };
}

MatrixX4dRB GradVar::evaluate() {
    Eigen::Matrix<float, -1, 1> crr = var->evaluate();
    auto interpolated = interpolate_to_face_linear(var->mesh, &crr);
    auto grads = gauss_grad<Mesh3D, false>(var->mesh, &interpolated);

    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);

    std::vector<bool> fgs = {clc_x, clc_y, clc_z};
    for (int i = 0; i < fgs.size(); ++i) {
        if (fgs.at(i)) {
            res += grads.at(i);
        }
    }

    return res;
}

CudaDataMatrixD GradVar::evaluate_cu() {
    auto crr = var->evaluate_cu();
    auto gr = eval_grad(dynamic_cast<CudaMesh3D*>(var->mesh), crr, clc_x, clc_y, clc_z);
    return gr;
}

Grad2Var::Grad2Var(Variable *var_, bool clc_x_, bool clc_y_, bool clc_z_) : clc_x{clc_x_}, clc_y{clc_y_},
                                                                            clc_z{clc_z_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    name = "Lapl_" + std::to_string(clc_x_) + std::to_string(clc_y_) + std::to_string(clc_z_) + var_->name;
    is_subvariable = true;

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        auto cu_mesh = dynamic_cast<CudaMesh3D *>(var_->mesh);

        jit_declarations = {
                fmt::format("size_t face_neigh_ids[{}];", Mesh3D::n_faces),
                fmt::format("float face_area[{}];", Mesh3D::n_faces),
                fmt::format("float alpha_d[{}];", Mesh3D::n_faces),
                fmt::format("float normals_face[{}][{}];", Mesh3D::n_dims, Mesh3D::n_faces),
                fmt::format("const float* normals_ptr[{}] = {{\n"
                            "            normals_x_ptr,\n"
                            "            normals_y_ptr,\n"
                            "            normals_z_ptr\n"
                            "    }};", Mesh3D::n_dims),
                fmt::format("float corrected_surface_normal_grad_{}[{}];", var_->name, Mesh3D::n_faces),
                fmt::format("float lapl_{}[{}];", name, Mesh3D::n_dims),
                fmt::format("float volume;"),
                fmt::format("float {} = 0;", name),
                fmt::format("float {}_neigh[{}] = {{ 0, 0, 0, 0, 0, 0 }};", name, Mesh3D::n_faces),
        };
        jit_operations = {
                fmt::format("volume = read_scalar_from_self(volume_ptr, idx, n);"),
                fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(ids_ptr, face_neigh_ids, idx, n);"),
                fmt::format("read_normals_from_self<Mesh3D>(normals_ptr, normals_face, idx, n);"),
                fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(face_area_ptr, face_area, idx, n);"),
                fmt::format("read_n_vars_from_self<Mesh3D::n_faces>(alpha_d_ptr, alpha_d, idx, n);"),

                fmt::format(
                        "corrected_surface_normal_grad_cu_k<Mesh3D>({0}, {0}_neigh, alpha_d, corrected_surface_normal_grad_{0});",
                        var_->name),
                fmt::format(
                        "gauss_grad_cu_k<Mesh3D>(corrected_surface_normal_grad_{0}, normals_face, face_area, volume, lapl_{1});",
                        var_->name, name),
                fmt::format("{{"
                            "        if ({1}) {0} += lapl_{0}[0];\n"
                            "        if ({2}) {0} += lapl_{0}[1];\n"
                            "        if ({3}) {0} += lapl_{0}[2];\n"
                            "}}", name, clc_x_, clc_y_, clc_z_)
        };
        jit_returns = {name};
    }
}

MatrixX4dRB Grad2Var::evaluate() {
    Eigen::Matrix<float, -1, 1> crr = var->evaluate();
//    auto interpolated = interpolate_to_face_linear(var->mesh, &crr);
//    auto grads = gauss_grad(var->mesh, &interpolated);
    std::vector<Eigen::Matrix<float, -1, 1>> grads = {};

    MatrixX6dRB surface_normal_grads = corrected_surface_normal_grad(var->mesh, grads, &(var->current));
    auto lapl = gauss_grad<Mesh3D, true>(var->mesh, &surface_normal_grads);

    std::vector<bool> fgs = {clc_x, clc_y, clc_z};
    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>(mesh->_num_nodes);
    res.setConstant(0);

    for (int i = 0; i < fgs.size(); ++i) {
        if (fgs.at(i)) {
            res += lapl.at(i);
        }
    }

    return res;
}

CudaDataMatrixD Grad2Var::evaluate_cu() {
    auto crr = var->evaluate_cu();
    auto gr2 = eval_grad2(dynamic_cast<CudaMesh3D*>(var->mesh), crr, clc_x, clc_y, clc_z);
    return gr2;
}


StabVar::StabVar(Variable *var_, bool clc_x_, bool clc_y_, bool clc_z_) : clc_x{clc_x_}, clc_y{clc_y_}, clc_z{clc_z_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    is_subvariable = true;
}

MatrixX4dRB StabVar::evaluate() {
    std::vector<bool> fgs = {clc_x, clc_y, clc_z};
    Eigen::Matrix<float, -1, 1> crr = var->evaluate();

    auto interpolated = interpolate_to_face_linear(var->mesh, &crr);
    auto grads = gauss_grad<Mesh3D, false>(var->mesh, &interpolated);

    auto interpolated_upwing = interpolate_to_face_upwing(mesh, &crr, &grads);

    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);

    for (int i = 0; i < fgs.size(); ++i) {
        if (fgs.at(i)) {
            MatrixX6dRB collected = collect_vals_neigh_faces<Mesh3D, false>(mesh, &(interpolated_upwing.at(i)));
            MatrixX6dRB summed_faces = 0.5 * (interpolated_upwing.at(i) - collected);
            auto appliedStabVar = gauss_grad<Mesh3D, true>(var->mesh, &summed_faces);
            res += appliedStabVar.at(i);
        }
    }

    return res;
}

CudaDataMatrixD StabVar::evaluate_cu() {
    auto crr = var->evaluate_cu();
    auto gr2 = eval_stab(dynamic_cast<CudaMesh3D*>(var->mesh), crr, clc_x, clc_y, clc_z);
    return gr2;
}

void EqSolver::solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt) {
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        auto current_cu = equation->evaluate_cu();
        sync_device();
        auto extracted_cu = time_var->extract_cu(current_cu, dt->_dt);
        set_var->set_current(extracted_cu, 0);
        if (0) {
            std::cout << "At " << set_var->name << " Current sum = " << set_var->current.sum() << " Current max = "
                      << set_var->current.maxCoeff() << std::endl;
        }

    } else {
        Eigen::Matrix<float, -1, 1> current = equation->evaluate();
        auto extracted = time_var->extract(current, dt->_dt);
        set_var->set_current(extracted);

        if (1) {
            std::cout << "At " << set_var->name << " Current sum = " << extracted.sum() << " Current max = "
                      << extracted.maxCoeff() << std::endl;
        }
    }
}

Equation::Equation(size_t timesteps_) : timesteps{timesteps_} {}

void Equation::evaluate(std::vector<Variable *> &all_vars,
                        EquationTemplate &equation_system, DT *dt, bool visualize,
                        std::vector<Variable *> store_vars) const {
    indicators::ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PostfixText{"0.0 %"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };

    const bool use_kernel_builder = USE_GPU && (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled);

#if USE_GPU
    std::vector<KernelBuilder> krns{};

    if (use_kernel_builder) {
        for (auto &equation: equation_system) {
            if (std::get<3>(equation)) {
                krns.emplace_back(std::get<0>(equation), &std::get<2>(equation));
                std::get<0>(equation)->jit_post();
            } else {
                krns.emplace_back(nullptr, nullptr);
            }
        }
        for (int i = 0; i < krns.size(); ++i) {
            if (std::get<3>(equation_system.at(i))) {
                krns.at(i).build();
            }
        }
    }
#endif

    for (int t = 0; t < timesteps; ++t) {
        dt->update();

        if (!use_kernel_builder) {
            for (auto &equation: equation_system) {
                auto left_part = std::get<0>(equation);
                auto &right_part = std::get<2>(equation);
                left_part->solve(&right_part, dt);
            }
        } else {
#if USE_GPU
            for (int i = 0; i < krns.size(); ++i) {
                auto& eq_crr = equation_system.at(i);
                if (std::get<3>(eq_crr)) {
                    auto* left_part = krns.at(i).left_val;
                    left_part->set_dt(dt->_dt);
                    krns.at(i).run();
                    left_part->jit_post();
                } else {
                    auto left_part = std::get<0>(equation_system.at(i));
                    auto &right_part = std::get<2>(equation_system.at(i));
                    left_part->solve(&right_part, dt);
                }
                sync_device();
            }
#endif
        }

        for (auto& var: all_vars) {
            if (var->has_boundary_conditions_cu && (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)) {
                var->set_bound_cu(dt);
//                var->current = var->current_cu.to_eigen();
            } else {
                if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
                    var->current = var->current_cu.to_eigen();
                }
                var->set_bound(dt);
            }
//            std::cout << "At " << var->name << " Current sum = " << var->current.sum() << " Current max = "
//                      << var->current.maxCoeff() << std::endl;
            var->add_history();
        }

        if (visualize && CFDArcoGlobalInit::get_rank() == 0) {
            float progress = (static_cast<float>(t) / static_cast<float>(timesteps)) * 100;
            bar.set_progress(static_cast<size_t>(progress));
            bar.set_option(indicators::option::PostfixText{std::to_string(progress) + " %"});
        }

//        if (CFDArcoGlobalInit::store_stepping) store_history_stepping(store_vars, store_vars.at(0)->mesh, t);
    }

    for (auto var: all_vars) {
        if (var->has_boundary_conditions_cu && (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)) {
            var->current = var->current_cu.to_eigen();
        }
        std::cout << "At " << var->name << " Current sum = " << var->current.sum() << " Current max = "
                  << var->current.maxCoeff() << std::endl;
    }

    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << "Time progress = " << dt->_current_time_dbl << std::endl;
}

std::vector<std::string> Variable::get_jit_declarations() {
    if (!is_subvariable) {
        return jit_declarations;
    }

    std::vector<std::string> ret = {};
    auto val_l = left_operand->get_jit_declarations();
    auto val_r = right_operand->get_jit_declarations();

    ret.insert(ret.end(), val_l.begin(), val_l.end());
    ret.insert(ret.end(), val_r.begin(), val_r.end());
    ret.insert(ret.end(), jit_declarations.begin(), jit_declarations.end());

    return ret;
}

std::vector<std::tuple<std::string, void*, TypeEnum>> Variable::get_jit_inputs() {
    if (!is_subvariable) {
        reset_input_pointers();
        return jit_inputs;
    }

    std::vector<std::tuple<std::string, void*, TypeEnum>> ret = {};
    auto val_l = left_operand->get_jit_inputs();
    auto val_r = right_operand->get_jit_inputs();

    ret.insert(ret.end(), val_l.begin(), val_l.end());
    ret.insert(ret.end(), val_r.begin(), val_r.end());
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> Variable::get_jit_operations() {
    if (!is_subvariable) {
        return jit_operations;
    }

    std::vector<std::string> ret = {};
    auto val_l = left_operand->get_jit_operations();
    auto val_r = right_operand->get_jit_operations();

    ret.insert(ret.end(), val_l.begin(), val_l.end());
    ret.insert(ret.end(), val_r.begin(), val_r.end());
    ret.insert(ret.end(), jit_operations.begin(), jit_operations.end());

    return ret;
}

void Variable::reset_input_pointers() {
    if (is_constvar) return;
    auto cu_mesh = dynamic_cast<CudaMesh3D*>(mesh);
    jit_inputs = {
            {fmt::format("float * {}_ptr", name), &current_cu, TypeEnum::DoublePointer},
            {fmt::format("const size_t * __restrict__ ids_ptr"),       &cu_mesh->_ids_cu, TypeEnum::ConstSizeTPointer},
    };
}

std::vector<std::string> Grad2Var::get_jit_declarations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_declarations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_declarations.begin(), jit_declarations.end());

    return ret;
}

std::vector<std::tuple<std::string, void*, TypeEnum>> Grad2Var::get_jit_inputs() {
    reset_input_pointers();
    std::vector<std::tuple<std::string, void*, TypeEnum>> ret = {};
    auto val_var = var->get_jit_inputs();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> Grad2Var::get_jit_operations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_operations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_operations.begin(), jit_operations.end());

    return ret;
}

void Grad2Var::reset_input_pointers() {
    auto cu_mesh = dynamic_cast<CudaMesh3D*>(var->mesh);
    jit_inputs = {
            {fmt::format("const size_t * __restrict__ ids_ptr"),       &cu_mesh->_ids_cu, TypeEnum::ConstSizeTPointer},
            {fmt::format("const float * __restrict__ normals_x_ptr"), &cu_mesh->_normal_x_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ normals_y_ptr"), &cu_mesh->_normal_y_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ normals_z_ptr"), &cu_mesh->_normal_z_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ face_area_ptr"), &cu_mesh->_face_areas_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ alpha_d_ptr"),   &cu_mesh->_alpha_d_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ volume_ptr"),    &cu_mesh->_volumes_cu, TypeEnum::ConstDoublePointer},

    };
}

std::vector<std::string> GradVar::get_jit_declarations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_declarations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_declarations.begin(), jit_declarations.end());

    return ret;
}

std::vector<std::tuple<std::string, void*, TypeEnum>> GradVar::get_jit_inputs() {
    reset_input_pointers();
    std::vector<std::tuple<std::string, void*, TypeEnum>> ret = {};
    auto val_var = var->get_jit_inputs();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> GradVar::get_jit_operations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_operations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_operations.begin(), jit_operations.end());

    return ret;
}

void GradVar::reset_input_pointers() {
    auto cu_mesh = dynamic_cast<CudaMesh3D*>(var->mesh);
    jit_inputs = {
            {fmt::format("const size_t * __restrict__ ids_ptr"),       &cu_mesh->_ids_cu, TypeEnum::ConstSizeTPointer},
            {fmt::format("const float * __restrict__ normals_alt_x_ptr"), &cu_mesh->_normal_alt_x_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ normals_alt_y_ptr"), &cu_mesh->_normal_alt_y_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ normals_alt_z_ptr"), &cu_mesh->_normal_alt_z_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ face_area_ptr"), &cu_mesh->_face_areas_cu, TypeEnum::ConstDoublePointer},
            {fmt::format("const float * __restrict__ volume_ptr"),    &cu_mesh->_volumes_cu, TypeEnum::ConstDoublePointer},

    };
}

#if USE_GPU
void KernelBuilder::build() {
    std::string kernel_template = "#include <custom_cuda_functions.hpp>\n"
                                  "extern \"C\" __global__ void cfdARCHOKernel({} size_t n) {{\n"    // here goes inputs
                                      "{} \n"  // here goes declataions
                                      "auto idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;\n"
                                      "if (idx < n) {{\n"
                                          "{} \n"  // here goes operations
                                          "{} \n"  // here goes assign
                                      "}}\n"
                                  "}}";

    auto vec_jit_declarations = expression->get_jit_declarations();
    auto vec_jit_inputs = expression->get_jit_inputs();
    auto vec_jit_operations = expression->get_jit_operations();

    std::string full_jit_declarations;
    std::vector<std::string> all_decls = {};
    for (const auto & vec_jit_declaration : vec_jit_declarations) {
        if(std::find(all_decls.begin(), all_decls.end(), vec_jit_declaration) == all_decls.end()) {
            full_jit_declarations += vec_jit_declaration + "\n";
            all_decls.push_back(vec_jit_declaration);
        }
    }

    std::string full_jit_inputs;
    std::vector<std::string> all_ins = {};
    std::vector<std::tuple<std::string, void*, TypeEnum>> inputs_in_order = {};
    for (const auto & vec_jit_input : vec_jit_inputs) {
        if(std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    auto assign_inputs = left_val->get_jit_inputs();
    for (const auto & vec_jit_input : assign_inputs) {
        if(std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    std::string full_jit_operations;
    std::vector<std::string> all_ops = {};
    for (const auto & vec_jit_operation : vec_jit_operations) {
        if(std::find(all_ops.begin(), all_ops.end(), vec_jit_operation) == all_ops.end()) {
            full_jit_operations += vec_jit_operation + "\n";
            all_ops.push_back(vec_jit_operation);
        }
    }

    auto jit_assign = left_val->jit_assign;
    auto full_jit_assign =  fmt::format(jit_assign, expression->jit_returns.at(0));

    std::string formed_kernel = fmt::format(kernel_template,
                                            full_jit_inputs,
                                            full_jit_declarations,
                                            full_jit_operations,
                                            full_jit_assign);

//    std::cout << "Building kernel: " << std::endl << formed_kernel << std::endl;

    const char* include_path = CFDARCO_JIT_INCLUDE_PATH;
    const char* eigeninclude_path = EIGEN3_INCLUDE_DIRS;
    const char* rmm_include_path = EIGEN3_INCLUDE_DIRS;
    const char* env_name = "OCCA_INCLUDE_PATH";
#if defined(CFDARCHO_CUDA_ENABLE)
    occa::json kernelProps({
        {"okl/enabled", false},
        {"compiler_flags", fmt::format("-std=c++17 -I{} -I{} -DCFDARCHO_CUDA_ENABLE -DCFDARCO_SKIP_RMM -diag-suppress 20012", include_path, eigeninclude_path)}
    });
#elif defined(CFDARCHO_HIP_ENABLE)
    occa::json kernelProps({
           {"okl/enabled", false},
//           {"compiler_flags", fmt::format("-std=c++17 -I{} -I{} -DCFDARCHO_HIP_ENABLE -DCFDARCO_SKIP_RMM -march=native", include_path, eigeninclude_path)}
           {"compiler_flags", fmt::format("-std=c++17 -I{} -I{} -DCFDARCHO_HIP_ENABLE -DCFDARCO_SKIP_RMM -arch=sm_80", include_path, eigeninclude_path)}
   });
#else
    occa::json kernelProps({
           {"okl/enabled", false},
           {"compiler_flags", fmt::format("-std=c++17 -I{} -I{} -DCFDARCO_SKIP_RMM -diag-suppress 20012", include_path, eigeninclude_path)}
   });
#endif
    compute_kernel = occa::buildKernelFromString(formed_kernel, "cfdARCHOKernel", kernelProps);

    int blocksize = 64;
    int nblocks = std::ceil(static_cast<float>(expression->mesh->_num_nodes) / static_cast<float>(blocksize));
    compute_kernel.setRunDims(nblocks, blocksize);


    std::cout << "Kernel build" << std::endl;
}

void KernelBuilder::run() {
    auto vec_jit_inputs = expression->get_jit_inputs();

    std::string full_jit_inputs;
    std::vector<std::string> all_ins = {};
    std::vector<std::tuple<std::string, void*, TypeEnum>> inputs_in_order = {};
    for (const auto & vec_jit_input : vec_jit_inputs) {
        if(std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    auto assign_inputs = left_val->get_jit_inputs();
    for (const auto & vec_jit_input : assign_inputs) {
        if(std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    compute_kernel.clearArgs();


    std::vector<occa::memory> mems{};
    for (int i = 0; i < inputs_in_order.size(); ++i) {
        auto& [arg_name, arg_ptr, arg_type] = inputs_in_order.at(i);

        if (arg_type == TypeEnum::ConstDoublePointer) {
            const auto* mtrx_ptr = static_cast<CudaDataMatrixD*>(arg_ptr);
            const auto* raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a  = occa::wrapMemory<float>(raw_ptr, mtrx_ptr->_size);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::DoublePointer) {
            auto* mtrx_ptr = static_cast<CudaDataMatrixD*>(arg_ptr);
            auto* raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a  = occa::wrapMemory<float>(raw_ptr, mtrx_ptr->_size);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::ConstSizeTPointer) {
            const auto* mtrx_ptr = static_cast<CudaDataMatrix<size_t>*>(arg_ptr);
            const auto* raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a  = occa::wrapMemory<size_t>(raw_ptr, mtrx_ptr->_size);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::SizeT) {
            const auto* mtrx_ptr = static_cast<const size_t*>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }

        if (arg_type == TypeEnum::Double) {
            const auto* mtrx_ptr = static_cast<const float*>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }

        if (arg_type == TypeEnum::Boolean) {
            const auto* mtrx_ptr = static_cast<const bool*>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }


    }

    compute_kernel.pushArg(expression->mesh->_num_nodes);

    sync_device();
    compute_kernel.run();
    sync_device();
}
#endif
