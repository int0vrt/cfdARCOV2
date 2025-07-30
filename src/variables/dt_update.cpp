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
#include "variables/dt_update.hpp"
#include "cuda_operators.hpp"

std::shared_ptr<Variable> CurrTime::clone() const {
    auto *new_obj = new CurrTime(*this);
    return std::shared_ptr<Variable>{new_obj};
}

CurrTime::CurrTime(Mesh3D *mesh_, DT *dt_) : _dt{dt_} {
    name = "curr_time";
    mesh = mesh_;

    jit_declarations = {
            fmt::format("float curr_time_neigh[{0}] = {{ curr_time, curr_time, curr_time, curr_time, curr_time, curr_time }};", Mesh3D::n_faces),
    };
    jit_operations = {};
    jit_memops = {};
    jit_returns = {name};
}

std::vector<std::string> CurrTime::get_jit_declarations() {
    return jit_declarations;
}

std::vector<std::tuple<std::string, void *, TypeEnum>> CurrTime::get_jit_inputs() {
    reset_input_pointers();

    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> CurrTime::get_jit_operations() {
    return {};
}

std::vector<std::string> CurrTime::get_jit_memops() {
    return {};
}

void CurrTime::reset_input_pointers() {
    jit_inputs = {
            {fmt::format("float curr_time"), &(_dt->_current_time_dbl), TypeEnum::Double},
    };
}

MatrixX4dRB CurrTime::evaluate() {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    ret.setConstant(_dt->_current_time_dbl);
    return ret;
}

CudaDataMatrixD CurrTime::evaluate_cu() {
    return CudaDataMatrixD{mesh->_num_nodes, _dt->_current_time_dbl};
}


std::shared_ptr<Variable> DT::clone() const {
    return std::shared_ptr<Variable>{const_cast<DT *>(this), [](Variable *) {}};
}

DT::DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       float CFL_, std::vector<Variable *> space_vars_) : update_fn{std::move(update_fn_)}, CFL{CFL_},
                                                          space_vars{std::move(space_vars_)} {
    name = "dt";
    mesh = mesh_;
    _dt = 0;

    jit_declarations = {
            fmt::format("float dt_neigh[{0}] = {{ dt, dt, dt, dt, dt, dt }};", Mesh3D::n_faces),
    };
    jit_operations = {};
    jit_memops = {};
    jit_returns = {name};
}

DT::DT(Mesh3D *mesh_, std::function<float(float, std::vector<Eigen::Matrix<float, -1, 1>> &, Mesh3D *mesh)> update_fn_,
       std::function<float(float, std::vector<CudaDataMatrixD *> &, Mesh3D *mesh)> update_fn_cu_, float CFL_,
       std::vector<Variable *> space_vars_) : update_fn{std::move(update_fn_)}, update_fn_cu{std::move(update_fn_cu_)},
                                              CFL{CFL_}, space_vars{std::move(space_vars_)} {
    name = "dt";
    mesh = mesh_;
    _dt = CFL_;
    has_update_fn_cu = true;

    jit_declarations = {
            fmt::format("float dt_neigh[{0}] = {{ dt, dt, dt, dt, dt, dt }};", Mesh3D::n_faces),
    };
    jit_operations = {};
    jit_memops = {};
    jit_returns = {name};
}

std::vector<std::string> DT::get_jit_declarations() {
    return jit_declarations;
}

std::vector<std::tuple<std::string, void *, TypeEnum>> DT::get_jit_inputs() {
    reset_input_pointers();

    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> DT::get_jit_operations() {
    return {};
}

std::vector<std::string> DT::get_jit_memops() {
    return {};
}

void DT::reset_input_pointers() {
    jit_inputs = {
            {fmt::format("float dt"), &(_dt), TypeEnum::Double},
    };
}

void DT::update() {
    float dt_c = 0.0;
    if (has_update_fn_cu && CFDArcoGlobalInit::cuda_enabled) {
        std::vector<CudaDataMatrixD *> redist{};
        for (auto &var: space_vars) {
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

// TODO: think about general interface
float
UpdatePolicies::CourantFriedrichsLewy(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
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

float
UpdatePolicies::CourantFriedrichsLewy3D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
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

float UpdatePolicies::CourantFriedrichsLewy3DCu(float CFL, std::vector<CudaDataMatrixD *> &space_vars, Mesh3D *mesh) {
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

float
UpdatePolicies::CourantFriedrichsLewy1D(float CFL, std::vector<Eigen::Matrix<float, -1, 1>> &space_vars, Mesh3D *mesh) {
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

float UpdatePolicies::CourantFriedrichsLewy1DCu(float CFL, std::vector<CudaDataMatrixD *> &space_vars, Mesh3D *mesh) {
    auto u = *space_vars.at(0);
    auto p = *space_vars.at(1);
    auto rho = *space_vars.at(2);
    auto gamma = 5. / 3.;
    float dl = std::min(std::min(mesh->_dx, mesh->_dy), mesh->_dz);
    auto denom = cfl_cu(dl, gamma, p, rho, u);
    auto dt = CFL * denom;

    return dt;
}
