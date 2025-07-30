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
#include "variables/variable.hpp"
#include "io_operators_tmpl.hpp"
#include "cuda_operators.hpp"
#include "equation.hpp"

#include <unsupported/Eigen/MatrixFunctions>

Variable::Variable(Mesh3D *mesh_, Eigen::Matrix<float, -1, 1> &initial_, BoundaryFN boundary_conditions_,
                   std::string name_) :
        mesh{mesh_}, current{initial_}, boundary_conditions{std::move(boundary_conditions_)}, name{std::move(name_)} {
    num_nodes = mesh->_num_nodes;
    is_basically_created = true;
    has_boundary_conditions_cu = false;
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
        auto cu_mesh = dynamic_cast<CudaMesh3D *>(mesh);
        jit_declarations = {
                fmt::format("__shared__ __align__(16) {} face_neigh_ids[pipeline_stages][BLOCKSIZE * {}];", cu_mesh->_ids_data_type, Mesh3D::n_faces),
                fmt::format("__shared__ __align__(16) float {}_shmem[pipeline_stages][BLOCKSIZE];", name),
                fmt::format("float {};", name),
                fmt::format("float {}_neigh[{}];", name, Mesh3D::n_faces)
        };
        jit_memops = {
                fmt::format(
                        "read_idxs_from_self_pipe<Mesh3D::n_faces>(ids_ptr, &(face_neigh_ids[pipe][0]), idx, n);"),
                fmt::format("read_n_vars_from_self_pipe<1>({0}_ptr, &({0}_shmem[pipe][0]), idx, n);", name),
        };
        jit_operations = {
                fmt::format("{0} = {0}_shmem[pipe][threadIdx.x];", name),
                fmt::format("read_n_vars_from_neigh_shmem<Mesh3D>({0}_ptr, &(face_neigh_ids[pipe][0]), {0}_neigh);",
                            name),
        };
        jit_returns = {name};
        jit_assign = {fmt::format("{}_ptr[idx] = {{}};", name)};
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
        auto cu_mesh = dynamic_cast<CudaMesh3D *>(mesh);
        jit_declarations = {
                fmt::format("__shared__ __align__(16) {} face_neigh_ids[pipeline_stages][BLOCKSIZE * {}];", cu_mesh->_ids_data_type, Mesh3D::n_faces),
                fmt::format("__shared__ __align__(16) float {}_shmem[pipeline_stages][BLOCKSIZE];", name),
                fmt::format("float {};", name),
                fmt::format("float {}_neigh[{}];", name, Mesh3D::n_faces)
        };
        jit_memops = {
                fmt::format(
                        "read_idxs_from_self_pipe<Mesh3D::n_faces>(ids_ptr, &(face_neigh_ids[pipe][0]), idx, n);"),
                fmt::format("read_n_vars_from_self_pipe<1>({0}_ptr, &({0}_shmem[pipe][0]), idx, n);", name),
        };
        jit_operations = {
                fmt::format("{0} = {0}_shmem[pipe][threadIdx.x];", name),
                fmt::format("read_n_vars_from_neigh_shmem<Mesh3D>({0}_ptr, &(face_neigh_ids[pipe][0]), {0}_neigh);",
                            name),
        };
        jit_returns = {name};
        jit_assign = {fmt::format("{}_ptr[idx] = {{}};", name)};
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

template<class To, class From>
std::enable_if_t<
        sizeof(To) == sizeof(From) &&
        std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
        To>
bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation additionally requires "
                  "destination type to be trivially constructible");

    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

Variable::Variable(const std::shared_ptr<Variable> &left_operand_, const std::shared_ptr<Variable> &right_operand_,
                   std::function<CudaDataMatrixD(CudaDataMatrixD &, CudaDataMatrixD &)> op_, std::string &name_,
                   std::vector<std::string> jit_declarations_,
                   std::vector<std::tuple<std::string, void *, TypeEnum>> jit_inputs_,
                   std::vector<std::string> jit_operations_, std::vector<std::string> jit_memops_,
                   std::vector<std::string> jit_returns_) :
        op_cu{std::move(op_)}, name{name_}, jit_declarations{std::move(jit_declarations_)},
        jit_inputs{std::move(jit_inputs_)},
        jit_operations{std::move(jit_operations_)}, jit_memops{std::move(jit_memops_)},
        jit_returns{std::move(jit_returns_)} {
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
    std::string var_name = fmt::format("constval_{:x}", bit_cast<uint32_t>(value));
    var_name.erase(std::remove(var_name.begin(), var_name.end(), '.'), var_name.end());
    name = var_name;
    is_constvar = true;
    is_basically_created = false;

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
        jit_declarations = {
                fmt::format("const float {} = {};", name, value),
                fmt::format("const float {0}_neigh[{1}] = {{ {2}, {2}, {2}, {2}, {2}, {2} }};", name, Mesh3D::n_faces,
                            value)
        };

        jit_inputs = {};
        jit_operations = {};
        jit_memops = {};
        jit_returns = {fmt::format("{}", name)};
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
    jit_memops = copy_var.jit_memops;
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
    jit_memops = copy_var.jit_memops;
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

void Variable::set_bound(const DT *dt_) {
    current = boundary_conditions(mesh, current, dt_);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        current_cu = CudaDataMatrixD::from_eigen(current);
    }
}

void Variable::set_bound_cu(const DT *dt_) {
    boundary_conditions_cu(mesh, current_cu, dt_);
    if (CFDArcoGlobalInit::get_size() > 1) {
        current = current_cu.to_eigen();
    }
}

void Variable::add_history() {
    if (!CFDArcoGlobalInit::skip_history) {
        if (history.size() % CFDArcoGlobalInit::store_n == 0) {
            if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
                current = current_cu.to_eigen();
            }
            if (!CFDArcoGlobalInit::store_stepping) {
                history.push_back({current});
            } else {
                history.push_back(Eigen::Matrix<float, -1, 1>{1});
                store_history_stepping(this, mesh, history.size());
            }
        } else {
            history.push_back(Eigen::Matrix<float, -1, 1>{1});
        }
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
        std::vector<std::string> jit_memops_{}; \
        std::vector<std::string> jit_returns_{ fmt::format("{}_" OPER_NAME "_{}", this->name, obj_r.name) }; \
        return {l_p, r_p, [](CudaDataMatrixD &lft, CudaDataMatrixD &rht) { return lft OPER_SiGN rht; }, name_, jit_declarations_, jit_inputs_, jit_operations_, jit_memops_, jit_returns_}; \
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
        std::vector<std::tuple<std::string, void *, TypeEnum>> jit_inputs_{};
        std::vector<std::string> jit_operations_{
                fmt::format("neg_{} = -{};", this->name, this->name),
                fmt::format("neg_{}_neigh[0] = -{}_neigh[0];", this->name, this->name),
                fmt::format("neg_{}_neigh[1] = -{}_neigh[1];", this->name, this->name),
                fmt::format("neg_{}_neigh[2] = -{}_neigh[2];", this->name, this->name),
                fmt::format("neg_{}_neigh[3] = -{}_neigh[3];", this->name, this->name),
                fmt::format("neg_{}_neigh[4] = -{}_neigh[4];", this->name, this->name),
                fmt::format("neg_{}_neigh[5] = -{}_neigh[5];", this->name, this->name),
        };
        std::vector<std::string> jit_memops_{};
        std::vector<std::string> jit_returns_{fmt::format("neg_{}", this->name)};

        return {l_p, r_p, [](CudaDataMatrixD &lft, CudaDataMatrixD &rht) { return -lft; }, name_, jit_declarations_,
                jit_inputs_, jit_operations_, jit_memops_, jit_returns_};
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
    std::string name_ = "exp_" + this->name;
    auto [l_p, r_p] = get_that_vars(this, *this);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        std::vector<std::string> jit_declarations_{
                fmt::format("float exp_{};", this->name),
                fmt::format("float exp_{}_neigh[{}];", this->name, Mesh3D::n_faces),
        };
        std::vector<std::tuple<std::string, void *, TypeEnum>> jit_inputs_{};
        std::vector<std::string> jit_operations_{
                fmt::format("exp_{} = std::exp({});", this->name, this->name),
                fmt::format("exp_{}_neigh[0] = std::exp({}_neigh[0]);", this->name, this->name),
                fmt::format("exp_{}_neigh[1] = std::exp({}_neigh[1]);", this->name, this->name),
                fmt::format("exp_{}_neigh[2] = std::exp({}_neigh[2]);", this->name, this->name),
                fmt::format("exp_{}_neigh[3] = std::exp({}_neigh[3]);", this->name, this->name),
                fmt::format("exp_{}_neigh[4] = std::exp({}_neigh[4]);", this->name, this->name),
                fmt::format("exp_{}_neigh[5] = std::exp({}_neigh[5]);", this->name, this->name),
        };
        std::vector<std::string> jit_memops_{};
        std::vector<std::string> jit_returns_{fmt::format("exp_{}", this->name)};

        return {l_p, r_p, [](CudaDataMatrixD &lft, CudaDataMatrixD &rht) { throw std::runtime_error("Not implemented"); return lft; }, name_, jit_declarations_,
                jit_inputs_, jit_operations_, jit_memops_, jit_returns_};
    }
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return lft.exp(); }, name_};
}

Variable exp(const Variable &obj) {
    std::string name_ = "exp_" + obj.name;
    auto [l_p, r_p] = get_that_vars(&obj, obj);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        std::vector<std::string> jit_declarations_{
                fmt::format("float exp_{};", obj.name),
                fmt::format("float exp_{}_neigh[{}];", obj.name, Mesh3D::n_faces),
        };
        std::vector<std::tuple<std::string, void *, TypeEnum>> jit_inputs_{};
        std::vector<std::string> jit_operations_{
                fmt::format("exp_{} = std::exp({});", obj.name, obj.name),
                fmt::format("exp_{}_neigh[0] = std::exp({}_neigh[0]);", obj.name, obj.name),
                fmt::format("exp_{}_neigh[1] = std::exp({}_neigh[1]);", obj.name, obj.name),
                fmt::format("exp_{}_neigh[2] = std::exp({}_neigh[2]);", obj.name, obj.name),
                fmt::format("exp_{}_neigh[3] = std::exp({}_neigh[3]);", obj.name, obj.name),
                fmt::format("exp_{}_neigh[4] = std::exp({}_neigh[4]);", obj.name, obj.name),
                fmt::format("exp_{}_neigh[5] = std::exp({}_neigh[5]);", obj.name, obj.name),
        };
        std::vector<std::string> jit_memops_{};
        std::vector<std::string> jit_returns_{fmt::format("exp_{}", obj.name)};

        return {l_p, r_p, [](CudaDataMatrixD &lft, CudaDataMatrixD &rht) { throw std::runtime_error("Not implemented"); return lft; }, name_, jit_declarations_,
                jit_inputs_, jit_operations_, jit_memops_, jit_returns_};
    }
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return Eigen::exp(lft.array()); }, name_};
}

Variable abs(const Variable &obj) {
    std::string name_ = "abs(" + obj.name + ")";
    auto [l_p, r_p] = get_that_vars(&obj, obj);
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)
        throw std::runtime_error("Not implemented");
    return {l_p, l_p, [](MatrixX4dRB &lft, MatrixX4dRB &rht) { return Eigen::abs(lft.array()); }, name_};
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

std::vector<std::tuple<std::string, void *, TypeEnum>> Variable::get_jit_inputs() {
    if (!is_subvariable) {
        reset_input_pointers();
        return jit_inputs;
    }

    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};
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

std::vector<std::string> Variable::get_jit_memops() {
    if (!is_subvariable) {
        return jit_memops;
    }

    std::vector<std::string> ret = {};
    auto val_l = left_operand->get_jit_memops();
    auto val_r = right_operand->get_jit_memops();

    ret.insert(ret.end(), val_l.begin(), val_l.end());
    ret.insert(ret.end(), val_r.begin(), val_r.end());
    ret.insert(ret.end(), jit_memops.begin(), jit_memops.end());

    return ret;
}

void Variable::reset_input_pointers() {
    if (is_constvar) return;
    auto cu_mesh = dynamic_cast<CudaMesh3D *>(mesh);
    jit_inputs = {
            {fmt::format("float * __restrict__ {}_ptr", name),   &current_cu,       TypeEnum::DoublePointer},
            {fmt::format("const {} * __restrict__ ids_ptr", cu_mesh->_ids_data_type), cu_mesh->_ids_mtrx_ptr, cu_mesh->_ids_data_type_en},
    };
}
