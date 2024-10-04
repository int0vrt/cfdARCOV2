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


#include "fvm.hpp"
#include "cfdarcho_main.hpp"
#include "io_operators.hpp"

#include <utility>
#include <indicators/progress_bar.hpp>
#include <unsupported/Eigen/MatrixFunctions>

Variable::Variable(Mesh2D* mesh_, Eigen::VectorXd &initial_, BoundaryFN boundary_conditions_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
    if (CFDArcoGlobalInit::cuda_enabled)
        current_cu = CudaDataMatrix::from_eigen(current);
}

Variable::Variable(Mesh2D* mesh_, Eigen::VectorXd &initial_, BoundaryFN boundary_conditions_, BoundaryFNCU boundary_conditions_cu_, std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, boundary_conditions_cu{std::move(boundary_conditions_cu_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
    has_boundary_conditions_cu = true;
    if (CFDArcoGlobalInit::cuda_enabled)
        current_cu = CudaDataMatrix::from_eigen(current);
}

Variable::Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_,
                   std::function<MatrixX4dRB(MatrixX4dRB &, MatrixX4dRB &)> op_, std::string &name_) :
                    op{op_}, name{name_}  {
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

Variable::Variable(const std::shared_ptr<Variable> left_operand_, const std::shared_ptr<Variable> right_operand_,
                   std::function<CudaDataMatrix(CudaDataMatrix &, CudaDataMatrix &)> op_, std::string &name_) :
        op_cu{op_}, name{name_}  {
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

Variable::Variable(Mesh2D* mesh_, double value) : mesh{mesh_} {
    current = Eigen::VectorXd {mesh->_num_nodes};
    current.setConstant(value);
    if (CFDArcoGlobalInit::cuda_enabled)
        current_cu = CudaDataMatrix::from_eigen(current);
    num_nodes = mesh->_num_nodes;
    name = std::to_string(value);
    is_constvar = true;
    is_basically_created = false;
}

Variable::Variable() {
    current = {};
    boundary_conditions = {};
    name = "uninitialized";
    num_nodes = 0;
    is_basically_created = false;
}

Variable::Variable(Eigen::VectorXd& curr_) {
    current = curr_;
    if (CFDArcoGlobalInit::cuda_enabled)
        current_cu = CudaDataMatrix::from_eigen(current);
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

    if (copy_var.left_operand) {
        left_operand = std::shared_ptr<Variable> {copy_var.left_operand->clone()};
    }
    if (copy_var.right_operand) {
        right_operand = std::shared_ptr<Variable> {copy_var.right_operand->clone()};
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

    if (copy_var.left_operand) {
        left_operand = std::shared_ptr<Variable> {copy_var.left_operand->clone()};
    }
    if (copy_var.right_operand) {
        right_operand = std::shared_ptr<Variable> {copy_var.right_operand->clone()};
    }
}

std::shared_ptr<Variable> Variable::clone() const {
    if (is_basically_created) {
        return std::shared_ptr<Variable>{const_cast<Variable*>(this), [](Variable *) {}};
    }
    auto* new_obj = new Variable(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _GradEstimated::clone() const {
    auto* new_obj = new _GradEstimated(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> PointerVariable::clone() const {
    return std::shared_ptr<Variable>{const_cast<PointerVariable*>(this), [](Variable *) {}};
}

std::shared_ptr<Variable> DT::clone() const {
    return std::shared_ptr<Variable>{const_cast<DT*>(this), [](Variable *) {}};
}

std::shared_ptr<Variable> _DT::clone() const {
    auto* new_obj = new _DT(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _D2T::clone() const {
    auto* new_obj = new _D2T(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _Grad::clone() const {
    auto* new_obj = new _Grad(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _Grad2::clone() const {
    auto* new_obj = new _Grad2(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::shared_ptr<Variable> _Stab::clone() const  {
    auto* new_obj = new _Stab(*this);
    return std::shared_ptr<Variable>{new_obj};
}


void Variable::set_bound() {
    current = boundary_conditions(mesh, current);
    if (CFDArcoGlobalInit::cuda_enabled) {
        current_cu = CudaDataMatrix::from_eigen(current);
    }
}

void Variable::set_bound_cu() {
    current_cu = boundary_conditions_cu(mesh, current_cu);
    if (CFDArcoGlobalInit::get_size() > 1) {
        current = current_cu.to_eigen(num_nodes, 1);
    }
}

void Variable::add_history() {
    if (CFDArcoGlobalInit::cuda_enabled && CFDArcoGlobalInit::skip_history && CFDArcoGlobalInit::store_stepping) {
        current = current_cu.to_eigen(num_nodes, 1);
    }

    if (CFDArcoGlobalInit::skip_history && !is_dt2) {
        return;
    }

    if (CFDArcoGlobalInit::skip_history && is_dt2) {
        history.push_back({current});

        if (history.size() > 3) {
            history.erase(history.begin());
        }

        return;
    }

    history.push_back({current});

}

MatrixX4dRB* Variable::estimate_grads() {
    if (estimate_grid_cache_valid) {
        return &estimate_grid_cache;
    }

    estimate_grid_cache = Eigen::Matrix<double, -1, 2> { num_nodes, 2};
    estimate_grid_cache.setConstant(0);
    auto pre_ret = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto send_name = name + "current";
    current_redist = CFDArcoGlobalInit::get_redistributed(current, send_name);
    current_redist_mtrx = Eigen::Matrix<double, -1, 4> {num_nodes, 4};

    for (int j = 0; j < 4; ++j) {
        auto current_n2 = current_redist[j];
        current_redist_mtrx.col(j) = current_n2;
        pre_ret.col(j) = (current + current_n2) / 2;
    }
    auto pre_ret_x = pre_ret.cwiseProduct(mesh->_normal_x);
    auto pre_ret_y = pre_ret.cwiseProduct(mesh->_normal_y);

    auto grad_x = pre_ret_x.rowwise().sum();
    auto grad_y = pre_ret_y.rowwise().sum();

    auto grad_xd = grad_x.cwiseQuotient(mesh->_volumes);
    auto grad_yd = grad_y.cwiseQuotient(mesh->_volumes);

    estimate_grid_cache.col(0) = grad_xd;
    estimate_grid_cache.col(1) = grad_yd;
    estimate_grid_cache_valid = true;

    send_name = name + "grad";
    grad_redist = CFDArcoGlobalInit::get_redistributed(estimate_grid_cache, send_name);
    grad_redist_mtrx_x = Eigen::Matrix<double, -1, 4> {num_nodes, 4};
    grad_redist_mtrx_y = Eigen::Matrix<double, -1, 4> {num_nodes, 4};
    for (int j = 0; j < 4; ++j) {
        grad_redist_mtrx_x.col(j) = grad_redist[j].col(0);
        grad_redist_mtrx_y.col(j) = grad_redist[j].col(1);
    }

    return &estimate_grid_cache;
}

std::tuple<CudaDataMatrix, CudaDataMatrix> Variable::estimate_grads_cu() {
    if (estimate_grid_cache_valid) {
        return estimate_grid_cache_cu;
    }

    auto send_name = name + "current";
    if (CFDArcoGlobalInit::get_size() > 1) {
        current_redist = CFDArcoGlobalInit::get_redistributed(current, send_name);
        current_redist_cu = {};
        for (const auto &elem: current_redist) {
            current_redist_cu.push_back(CudaDataMatrix::from_eigen(elem));
        }
    } else {
        current_redist_cu = {};
        for (int i = 0; i < 4; ++i) {
            current_redist_cu.emplace_back(num_nodes);
        }
        from_indices(current_cu, mesh->_n2_ids_cu, current_redist_cu, num_nodes);
    }

    auto grad_xd = CudaDataMatrix(current_cu._size, 0);
    auto grad_yd = CudaDataMatrix(current_cu._size, 0);

    estimate_grads_kern(current_redist_cu, current_cu, mesh->_normal_x_cu, mesh->_normal_y_cu, mesh->_volumes_cu,
                        grad_xd, grad_yd);

    estimate_grid_cache_cu = {grad_xd, grad_yd};
    estimate_grid_cache_valid = true;

    send_name = name + "grad";

    if (CFDArcoGlobalInit::get_size() > 1) {
        auto grads_cu = from_multiple_cols({grad_xd, grad_yd});
        auto grads_redist = CFDArcoGlobalInit::get_redistributed(grads_cu.to_eigen(num_nodes, 2), send_name);
        grad_redist_cu = {};
        for (auto &grads_i_cu: grads_redist) {
            grad_redist_cu.emplace_back(CudaDataMatrix::from_eigen(grads_i_cu.col(0).eval()),
                                        CudaDataMatrix::from_eigen(grads_i_cu.col(1).eval()));
        }
    } else {
        auto grad_redist_cu1 = std::vector<CudaDataMatrix>{};
        auto grad_redist_cu2 = std::vector<CudaDataMatrix>{};
        for (int i = 0; i < 4; ++i) {
            grad_redist_cu1.emplace_back(num_nodes);
            grad_redist_cu2.emplace_back(num_nodes);
        }
        from_indices(grad_xd, mesh->_n2_ids_cu, grad_redist_cu1, num_nodes);
        from_indices(grad_yd, mesh->_n2_ids_cu, grad_redist_cu2, num_nodes);
        grad_redist_cu = {};
        for (int i=0; i < 4; ++i) {
            grad_redist_cu.emplace_back(grad_redist_cu1.at(i),
                                        grad_redist_cu2.at(i));
        }
    }
    return estimate_grid_cache_cu;
}

_GradEstimated Variable::dx() {
    return _GradEstimated{this, true, false};
}

_GradEstimated Variable::dy() {
    return _GradEstimated{this,  false, true};
}

Tup3* Variable::get_interface_vars_second_order() {
    if (is_subvariable) {
        auto var_l = left_operand->get_interface_vars_second_order();
        auto var_r = right_operand->get_interface_vars_second_order();

        get_second_order_cache = {op(var_l->el1, var_r->el1), op(var_l->el2, var_r->el2), op(var_l->el3, var_r->el3)};

        return &get_second_order_cache;
    }

    if (get_second_order_cache_valid) {
        return &get_second_order_cache;
    }

    auto grads = estimate_grads();

    auto cur_self = current.replicate(1, 4);

    auto ret_r = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_l = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto ret_sum = (-cur_self + current_redist_mtrx) / 0.002;

    ret_r.col(0) = current_redist_mtrx.col(0);
    ret_r.col(1) = cur_self.col(1);
    ret_r.col(2) = cur_self.col(2);
    ret_r.col(3) = current_redist_mtrx.col(3);

    ret_l.col(0) = cur_self.col(0);
    ret_l.col(1) = current_redist_mtrx.col(1);
    ret_l.col(2) = current_redist_mtrx.col(2);
    ret_l.col(3) = cur_self.col(3);

    get_second_order_cache = {ret_sum, ret_r, ret_l};
    get_second_order_cache_valid = true;
    return &get_second_order_cache;
}

std::tuple<CudaDataMatrix, CudaDataMatrix, CudaDataMatrix> Variable::get_interface_vars_second_order_cu() {
    if (is_subvariable) {
        auto [left_eval1, left_eval2, left_eval3] = left_operand->get_interface_vars_second_order_cu();
        auto [right_eval1, right_eval2, right_eval3] = right_operand->get_interface_vars_second_order_cu();

        return {op_cu(left_eval1, right_eval1), op_cu(left_eval2, right_eval2), op_cu(left_eval3, right_eval3)};
    }

    if (get_second_order_cache_valid) {
        return get_second_order_cache_cu;
    }

    auto grads = estimate_grads_cu();

    auto grads_x = std::get<0>(grads);
    auto grads_y = std::get<1>(grads);

    CudaDataMatrix ret_r_cu{current_cu._size * 4};
    CudaDataMatrix ret_l_cu{current_cu._size * 4};

    auto cur_self = from_multiple_cols({current_cu, current_cu, current_cu, current_cu});
    auto ret_sum_cu = (-cur_self + from_multiple_cols(current_redist_cu)) / CudaDataMatrix{current_cu._size * 4, 0.002};

    get_second_order_cache_cu = {ret_sum_cu, ret_r_cu, ret_l_cu};
    get_second_order_cache_valid = true;
    return get_second_order_cache_cu;
}

Tup3* Variable::get_interface_vars_first_order() {
    if (is_subvariable) {
        auto var_l = left_operand->get_interface_vars_first_order();
        auto var_r = right_operand->get_interface_vars_first_order();

        get_first_order_cache = {op(var_l->el1, var_r->el1), op(var_l->el2, var_r->el2), op(var_l->el3, var_r->el3)};

        return &get_first_order_cache;
    }

    if (get_first_order_cache_valid) {
        return &get_first_order_cache;
    }

    auto grads = estimate_grads();

    auto grads_self_x = grads->col(0).replicate(1, 4);
    auto grads_self_y = grads->col(1).replicate(1, 4);
    auto cur_self = current.replicate(1, 4);

    auto ret_r = Eigen::Matrix<double, -1, 4> { num_nodes, 4};
    auto ret_l = Eigen::Matrix<double, -1, 4> { num_nodes, 4};

    auto grads_self_xd = grads_self_x.cwiseProduct(mesh->_vec_in_edge_direction_x);
    auto grads_self_yd = grads_self_y.cwiseProduct(mesh->_vec_in_edge_direction_y);
    auto grads_neigh_xd = grad_redist_mtrx_x.cwiseProduct(mesh->_vec_in_edge_neigh_direction_x);
    auto grads_neigh_yd = grad_redist_mtrx_y.cwiseProduct(mesh->_vec_in_edge_neigh_direction_y);

    auto val_self = (grads_self_xd + grads_self_yd + cur_self).eval();
    auto val_neigh = (grads_neigh_xd + grads_neigh_yd + current_redist_mtrx).eval();

    auto ret_sum = (val_self + val_neigh) / 2;

    ret_r.col(0) = val_neigh.col(0);
    ret_r.col(1) = val_self.col(1);
    ret_r.col(2) = val_self.col(2);
    ret_r.col(3) = val_neigh.col(3);

    ret_l.col(0) = val_self.col(0);
    ret_l.col(1) = val_neigh.col(1);
    ret_l.col(2) = val_neigh.col(2);
    ret_l.col(3) = val_self.col(3);

    get_first_order_cache = {ret_sum, ret_r, ret_l};
    get_first_order_cache_valid = true;
    return &get_first_order_cache;
}

std::tuple<CudaDataMatrix, CudaDataMatrix, CudaDataMatrix> Variable::get_interface_vars_first_order_cu() {
    if (is_subvariable) {
        auto [left_eval1, left_eval2, left_eval3] = left_operand->get_interface_vars_first_order_cu();
        auto [right_eval1, right_eval2, right_eval3] = right_operand->get_interface_vars_first_order_cu();

        return {op_cu(left_eval1, right_eval1), op_cu(left_eval2, right_eval2), op_cu(left_eval3, right_eval3)};
    }

    if (get_first_order_cache_valid) {
        return get_first_order_cache_cu;
    }

    auto grads = estimate_grads_cu();

    auto grads_x = std::get<0>(grads);
    auto grads_y = std::get<1>(grads);

    CudaDataMatrix ret_sum_cu{current_cu._size * 4};
    CudaDataMatrix ret_r_cu{current_cu._size * 4};
    CudaDataMatrix ret_l_cu{current_cu._size * 4};

    get_interface_vars_first_order_kern(
            grads_x,
            grads_y,
            grad_redist_cu,
            current_cu,
            current_redist_cu,
            mesh->_vec_in_edge_direction_x_cu,
            mesh->_vec_in_edge_direction_y_cu,
            mesh->_vec_in_edge_neigh_direction_x_cu,
            mesh->_vec_in_edge_neigh_direction_y_cu,
            ret_sum_cu,
            ret_r_cu,
            ret_l_cu
    );

    get_first_order_cache_cu = {ret_sum_cu, ret_r_cu, ret_l_cu};
    get_first_order_cache_valid = true;
    return get_first_order_cache_cu;
}


Eigen::VectorXd Variable::extract(Eigen::VectorXd &left_part, double dt) {
    return left_part;
}

CudaDataMatrix Variable::extract_cu(CudaDataMatrix &left_part, double dt) {
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

CudaDataMatrix Variable::evaluate_cu() {
    if (!is_subvariable) {
        return current_cu;
    }

    auto val_l = left_operand->evaluate_cu();
    auto val_r = right_operand->evaluate_cu();
    return op_cu(val_l, val_r);
}

void Variable::set_current(Eigen::VectorXd &current_) {
    current = current_;
    estimate_grid_cache_valid = false;
    get_first_order_cache_valid = false;
    get_second_order_cache_valid = false;
    if (CFDArcoGlobalInit::cuda_enabled) {
        current_cu = CudaDataMatrix::from_eigen(current);
    }
}

void Variable::set_current(CudaDataMatrix& current_, bool copy_to_host) {
    estimate_grid_cache_valid = false;
    get_first_order_cache_valid = false;
    get_second_order_cache_valid = false;
    current_cu = current_;
    if (copy_to_host) {
        current = current_.to_eigen(num_nodes, 1);
    }
}

std::vector<Eigen::VectorXd> Variable::get_history() {
    return history;
}

void Variable::solve(Variable* equation, DT* dt) {
    EqSolver::solve_dt(equation, this, this, dt);
}

std::tuple<std::shared_ptr<Variable>, std::shared_ptr<Variable>> get_that_vars(const Variable* obj_l, const Variable &obj_r) {
    std::shared_ptr<Variable> l_p;
    std::shared_ptr<Variable> r_p;
    if (obj_l->is_subvariable || obj_l->is_constvar) {
        l_p = std::shared_ptr<Variable> {obj_l->clone()};
    } else {
        l_p = std::shared_ptr<Variable>(const_cast<Variable*>(obj_l), [](Variable *) {});
    }
    if (obj_r.is_subvariable || obj_r.is_constvar) {
        r_p = std::shared_ptr<Variable> {obj_r.clone()};
    } else {
        r_p = std::shared_ptr<Variable>(const_cast<Variable*>(&obj_r), [](Variable *) {});
    }

    return {l_p, r_p};
}

Variable Variable::operator+(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "+" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    if (CFDArcoGlobalInit::cuda_enabled)
        return {l_p, r_p, [](CudaDataMatrix& lft, CudaDataMatrix& rht){return lft + rht;}, name_};
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft + rht;}, name_};
}

Variable Variable::operator-(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "-" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    if (CFDArcoGlobalInit::cuda_enabled)
        return {l_p, r_p, [](CudaDataMatrix& lft, CudaDataMatrix& rht){return lft - rht;}, name_};
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft - rht;}, name_};
}

Variable Variable::operator*(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "*" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    if (CFDArcoGlobalInit::cuda_enabled)
        return {l_p, r_p, [](CudaDataMatrix& lft, CudaDataMatrix& rht){return lft * rht;}, name_};
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft.cwiseProduct(rht);}, name_};
}

Variable Variable::operator/(const Variable &obj_r) const {
    std::string name_ = "(" + this->name + "/" + obj_r.name + ")";
    auto [l_p, r_p] = get_that_vars(this, obj_r);
    if (CFDArcoGlobalInit::cuda_enabled)
        return {l_p, r_p, [](CudaDataMatrix& lft, CudaDataMatrix& rht){return lft / rht;}, name_};
    return {l_p, r_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft.cwiseQuotient(rht);}, name_};
}

Variable Variable::operator-() const {
    std::string name_ = "-(" + this->name + ")";
    auto [l_p, r_p] = get_that_vars(this, *this);
    if (CFDArcoGlobalInit::cuda_enabled)
        return {l_p, r_p, [](CudaDataMatrix& lft, CudaDataMatrix& rht){return -lft;}, name_};
    return {l_p, l_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return -lft;}, name_};
}

Variable operator+(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l + obj_r;
}

Variable operator-(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l - obj_r;
}

Variable operator*(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l * obj_r;
}

Variable operator/(const double obj_l, const Variable & obj_r) {
    auto val_l = Variable{obj_r.mesh, obj_l};
    return val_l / obj_r;
}

Variable operator+(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l + val_r;
}

Variable operator-(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l - val_r;
}

Variable operator*(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l * val_r;
}

Variable operator/(const Variable & obj_l, const double obj_r) {
    auto val_r = Variable{obj_l.mesh, obj_r};
    return obj_l / val_r;
}

Variable Variable::exp() const {
    std::string name_ = "-(" + this->name + ")";
    auto [l_p, r_p] = get_that_vars(this, *this);
    if (CFDArcoGlobalInit::cuda_enabled)
        throw std::runtime_error("Not implemented");
    return {l_p, l_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return lft.exp();}, name_};
}

Variable exp(const Variable & obj) {
    std::string name_ = "-(" + obj.name + ")";
    auto [l_p, r_p] = get_that_vars(&obj, obj);
    if (CFDArcoGlobalInit::cuda_enabled)
        throw std::runtime_error("Not implemented");
    return {l_p, l_p, [](MatrixX4dRB& lft, MatrixX4dRB& rht){return Eigen::exp(lft.array());}, name_};
}

_GradEstimated::_GradEstimated(Variable *var_, bool clc_x_, bool clc_y_) : var{var_}, clc_x{clc_x_}, clc_y{clc_y_} {
    mesh = var->mesh;
    name = "GradEstimated(" + var_->name + ")";
    is_constvar = true;
}

MatrixX4dRB _GradEstimated::evaluate() {
    auto grads = var->estimate_grads();
    if (clc_x && clc_y) {
        return grads->col(0) + grads->col(1);
    }
    if (clc_x) {
        return grads->col(0);
    }
    if (clc_y) {
        return grads->col(1);
    }
    return MatrixX4dRB{};
}

CudaDataMatrix _GradEstimated::evaluate_cu() {
    auto grads = var->estimate_grads_cu();
    if (clc_x && clc_y) {
        return std::get<0>(grads) + std::get<1>(grads);
    }
    if (clc_x) {
        return std::get<0>(grads);
    }
    if (clc_y) {
        return std::get<1>(grads);
    }
    return CudaDataMatrix{};
}

// TODO: think about general interface
double UpdatePolicies::CourantFriedrichsLewy(double CFL, std::vector<Eigen::VectorXd> &space_vars, Mesh2D* mesh) {
    auto u = space_vars.at(0);
    auto v = space_vars.at(1);
    auto p = space_vars.at(2);
    auto rho = space_vars.at(3);
    auto gamma = 5. / 3.;
    double dl = std::min(mesh->_dx, mesh->_dy);
    auto denom = dl * (((gamma * p.array()).cwiseQuotient(rho.array())).cwiseSqrt() + (u.array() * u.array() + v.array() * v.array()).cwiseSqrt()).cwiseInverse();

    auto dt = CFL * denom.minCoeff();

    return dt;
}

double UpdatePolicies::CourantFriedrichsLewyCu(double CFL, std::vector<CudaDataMatrix> &space_vars, Mesh2D* mesh) {
    auto u = space_vars.at(0);
    auto v = space_vars.at(1);
    auto p = space_vars.at(2);
    auto rho = space_vars.at(3);
    auto gamma = 5. / 3.;
    double dl = std::min(mesh->_dx, mesh->_dy);
    auto denom = cfl_cu(dl, gamma, p, rho, u, v);
    auto dt = CFL * denom;

    return dt;
}

PointerVariable::PointerVariable(Mesh2D* mesh_, double* ptr) : _ptr{ptr} {
    name = "PointerVariable";
    mesh = mesh_;
}

MatrixX4dRB PointerVariable::evaluate() {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    ret.setConstant(*_ptr);
    return ret;
}

CudaDataMatrix PointerVariable::evaluate_cu() {
    return CudaDataMatrix{mesh->_num_nodes, *_ptr};
}

DT::DT(Mesh2D* mesh_, std::function<double(double, std::vector<Eigen::VectorXd> &, Mesh2D* mesh)> update_fn_, double CFL_, std::vector<Variable*> space_vars_) : update_fn{update_fn_}, CFL{CFL_}, space_vars{space_vars_} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;
}

DT::DT(Mesh2D* mesh_, std::function<double(double, std::vector<Eigen::VectorXd> &, Mesh2D* mesh)> update_fn_,
       std::function<double(double, std::vector<CudaDataMatrix> &, Mesh2D* mesh)> update_fn_cu_, double CFL_,
       std::vector<Variable*> space_vars_) : update_fn{update_fn_}, update_fn_cu{update_fn_cu_}, CFL{CFL_}, space_vars{space_vars_} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;
    has_update_fn_cu = true;
}

void DT::update() {
    double dt_c = 0.0;
    if (has_update_fn_cu && CFDArcoGlobalInit::cuda_enabled) {
        std::vector<CudaDataMatrix> redist{};
        for (auto var : space_vars) {
            redist.push_back(var->current_cu);
        }
        dt_c = update_fn_cu(CFL, redist, mesh);
    } else {
        std::vector<Eigen::VectorXd> redist{};
        for (auto var : space_vars) {
            redist.push_back(var->current);
        }
        dt_c = update_fn(CFL, redist, mesh);
    }

    MPI_Allreduce(&dt_c, &_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    _current_time_step_int++;
    _current_time_dbl += _dt;
}

MatrixX4dRB DT::evaluate() {
    auto ret = Eigen::VectorXd{mesh->_num_nodes};
    ret.setConstant(_dt);
    return ret;
}

CudaDataMatrix DT::evaluate_cu() {
    return CudaDataMatrix{mesh->_num_nodes, _dt};
}


_DT::_DT(Variable *var_, int) {
    var = std::shared_ptr<Variable> {var_, [](Variable *) {}};
}

Eigen::VectorXd _DT::extract(Eigen::VectorXd& left_part, double dt) {
    return dt * left_part + var->current;
}

CudaDataMatrix _DT::extract_cu(CudaDataMatrix& left_part, double dt) {
    auto res = mul_mtrx(left_part, dt) + var->current_cu;
    return res;
}

void _DT::solve(Variable* equation, DT* dt) {
    EqSolver::solve_dt(equation, this, var.get(), dt);
}

_D2T::_D2T(Variable *var_, int) {
    var = std::shared_ptr<Variable> {var_, [](Variable *) {}};
    var->is_dt2 = true;
}

Eigen::VectorXd _D2T::extract(Eigen::VectorXd& left_part, double dt) {
    if (var->history.size() > 1) {
        return dt * dt * left_part + 2 * var->current - var->history.at(var->history.size() - 2);
    }
    return dt * left_part + var->current;
}

CudaDataMatrix _D2T::extract_cu(CudaDataMatrix& left_part, double dt) {
    if (var->history.size() > 1) {
        auto res = mul_mtrx(left_part, dt * dt) + mul_mtrx(var->current_cu, 2) - CudaDataMatrix::from_eigen(var->history.at(var->history.size() - 2));
        return res;
    }

    auto res = mul_mtrx(left_part, dt) + var->current_cu;
    return res;
}

void _D2T::solve(Variable* equation, DT* dt) {
    EqSolver::solve_dt(equation, this, var.get(), dt);
}


_Grad::_Grad(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    num_nodes = var->num_nodes;
    name = "Grad(" + var->name + ")";
}

MatrixX4dRB _Grad::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto& current_interface_gradients_star = current_interface_gradients->el1;

    if (clc_x && clc_y) {
        auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
        auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();
        return res_x + res_y;
    }
    if (clc_x) {
        auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
        return res_x;
    }
    if (clc_y) {
        auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();
        return res_y;
    }
    return MatrixX4dRB{};
}

CudaDataMatrix _Grad::evaluate_cu() {
    auto current_interface_gradients = var->get_interface_vars_first_order_cu();
    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);

    if (clc_x && clc_y) {
        auto res_x = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_x_cu, mesh->_num_nodes, 4);
        auto res_y = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_y_cu, mesh->_num_nodes, 4);
        return res_x + res_y;
    }
    if (clc_x) {
        auto res_x = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_x_cu, mesh->_num_nodes, 4);
        return res_x;
    }
    if (clc_y) {
        auto res_y = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_y_cu, mesh->_num_nodes, 4);
        return res_y;
    }
    return CudaDataMatrix{};
}

Tup3* _Grad::get_interface_vars_first_order() {
    current = var->evaluate();
    return Variable::get_interface_vars_first_order();
}

_Grad2::_Grad2(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
}

MatrixX4dRB _Grad2::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_second_order();
    auto& current_interface_gradients_star = current_interface_gradients->el1;

    MatrixX4dRB scaler {1, 4};
    scaler(0, 0) = -1;
    scaler(0, 1) = 1;
    scaler(0, 2) = 1;
    scaler(0, 3) = -1;
    MatrixX4dRB scaler_repl = scaler.replicate(var->num_nodes, 1);

    if (clc_x && clc_y) {
        auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x.cwiseProduct(scaler_repl))).rowwise().sum();
        auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y.cwiseProduct(scaler_repl))).rowwise().sum();
        return res_x + res_y;
    }

    if (clc_x) {
        auto scaled_norms = var->mesh->_normal_x.cwiseProduct(scaler_repl);
        auto res_x = (current_interface_gradients_star.cwiseProduct(scaled_norms)).rowwise().sum();
        return res_x;
    }
    if (clc_y) {
        auto scaled_norms = var->mesh->_normal_y.cwiseProduct(scaler_repl);
        auto res_y = (current_interface_gradients_star.cwiseProduct(scaled_norms)).rowwise().sum();
        return res_y;
    }
    return MatrixX4dRB{};
}

CudaDataMatrix _Grad2::evaluate_cu() {
    auto current_interface_gradients = var->get_interface_vars_second_order_cu();
    auto& current_interface_gradients_star = std::get<0>(current_interface_gradients);

    MatrixX4dRB scaler {1, 4};
    scaler(0, 0) = -1;
    scaler(0, 1) = 1;
    scaler(0, 2) = 1;
    scaler(0, 3) = -1;
    auto scaler_repl = CudaDataMatrix::from_eigen(scaler.replicate(var->num_nodes, 1));

    if (clc_x && clc_y) {
        auto res_x = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_x_cu * scaler_repl, mesh->_num_nodes, 4);
        auto res_y = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_y_cu * scaler_repl, mesh->_num_nodes, 4);
        return res_x + res_y;
    }
    if (clc_x) {
        auto res_x = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_x_cu * scaler_repl, mesh->_num_nodes, 4);
        return res_x;
    }
    if (clc_y) {
        auto res_y = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_y_cu * scaler_repl, mesh->_num_nodes, 4);
        return res_y;
    }
    return CudaDataMatrix{};
}


_Stab::_Stab(Variable *var_, bool clc_x_, bool clc_y_) : clc_x{clc_x_}, clc_y{clc_y_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
}

MatrixX4dRB _Stab::evaluate() {
    auto current_interface_gradients = var->get_interface_vars_first_order();
    auto current_interface_gradients_star = (current_interface_gradients->el3 - current_interface_gradients->el2) / 2;

    if (clc_x && clc_y) {
        auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
        auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();
        return res_x + res_y;
    }
    if (clc_x) {
        auto res_x = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_x)).rowwise().sum();
        return res_x;
    }
    if (clc_y) {
        auto res_y = (current_interface_gradients_star.cwiseProduct(var->mesh->_normal_y)).rowwise().sum();
        return res_y;
    }
    return MatrixX4dRB{};
}

CudaDataMatrix _Stab::evaluate_cu() {
    auto current_interface_gradients = var->get_interface_vars_first_order_cu();
    auto current_interface_gradients_star = div_const(std::get<2>(current_interface_gradients) - std::get<1>(current_interface_gradients), 2);

    if (clc_x && clc_y) {
        auto res_x = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_x_cu, mesh->_num_nodes, 4);
        auto res_y = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_y_cu, mesh->_num_nodes, 4);
        return res_x + res_y;
    }
    if (clc_x) {
        auto res_x = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_x_cu, mesh->_num_nodes, 4);
        return res_x;
    }
    if (clc_y) {
        auto res_y = rowwice_sum(current_interface_gradients_star * var->mesh->_normal_y_cu, mesh->_num_nodes, 4);
        return res_y;
    }
    return CudaDataMatrix{};
}

void EqSolver::solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt) {
    if (CFDArcoGlobalInit::cuda_enabled) {
        auto current_cu = equation->evaluate_cu();
        sync_device();
        auto extracted_cu = time_var->extract_cu(current_cu, dt->_dt);
        set_var->set_current(extracted_cu, CFDArcoGlobalInit::get_size() > 1);

    } else {
        Eigen::VectorXd current = equation->evaluate();
        auto other_var = to_grid_local(set_var->mesh, current);
        auto extracted = time_var->extract(current, dt->_dt);
        set_var->set_current(extracted);
    }
}

Equation::Equation(size_t timesteps_) : timesteps{timesteps_} {}

void Equation::evaluate(std::vector<Variable*> &all_vars,
                        EquationTemplate &equation_system, DT* dt, bool visualize,
                        std::vector<Variable*> store_vars) {
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

    for (int t = 0; t < timesteps; ++t) {
        dt->update();

        for (auto var : all_vars) {
            if (var->has_boundary_conditions_cu && CFDArcoGlobalInit::cuda_enabled) {
                var->set_bound_cu();
            } else {
                var->set_bound();
            }
        }

        for (auto& equation : equation_system) {
            auto left_part = std::get<0>(equation);
            auto& right_part = std::get<2>(equation);
            left_part->solve(&right_part, dt);
        }

        for (auto var : all_vars) {
            if (var->has_boundary_conditions_cu && CFDArcoGlobalInit::cuda_enabled) {
                var->set_bound_cu();
            } else {
                if (CFDArcoGlobalInit::cuda_enabled) {
                    var->current = var->current_cu.to_eigen(var->num_nodes, 1);
                }
                var->set_bound();
            }
            var->add_history();
        }

        if (visualize && CFDArcoGlobalInit::get_rank() == 0) {
            double progress = (static_cast<double>(t) / static_cast<double>(timesteps)) * 100;
            bar.set_progress(static_cast<size_t>(progress));
            bar.set_option(indicators::option::PostfixText{std::to_string(progress) + " %"});
        }

        if (CFDArcoGlobalInit::store_stepping) store_history_stepping(store_vars, store_vars.at(0)->mesh, t);
    }

    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << "Time progress = " << dt->_current_time_dbl << std::endl;
}

MatrixX4dRB to_grid(const Mesh2D* mesh, Eigen::VectorXd& values_half) {
    auto values = CFDArcoGlobalInit::recombine(values_half, "to_grid");
    MatrixX4dRB result = {mesh->_x, mesh->_y};
    for (auto& node : mesh->_nodes_tot) {
        double value = values(node->_id);
        size_t x_coord = node->_id / mesh->_x;
        size_t y_coord = node->_id % mesh->_x;
        result(x_coord, y_coord) = value;
    }
    return result;
}


MatrixX4dRB to_grid_local(const Mesh2D* mesh, Eigen::VectorXd& values) {
    MatrixX4dRB result = {mesh->_x, mesh->_y};
    for (auto& node : mesh->_nodes) {
        double value = values(node->_id);
        size_t x_coord = node->_id / mesh->_x;
        size_t y_coord = node->_id % mesh->_x;
        result(x_coord, y_coord) = value;
    }
    return result;
}


Eigen::VectorXd from_grid(const Mesh2D* mesh, MatrixX4dRB& grid) {
    Eigen::VectorXd result {mesh->_num_nodes};

    for (auto& node : mesh->_nodes) {
        size_t x_coord = node->_id / mesh->_x;
        size_t y_coord = node->_id % mesh->_x;
        double value = grid(x_coord, y_coord);
        result(node->_id) = value;
    }

    return result;
}
