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

#include <iostream>
//#include <matplot/matplot.h>
#include <chrono>
#include <thread>
#include <argparse/argparse.hpp>

#include "mesh3d.hpp"
#include "fvm3d.hpp"
#include "utils3d.hpp"


Eigen::Matrix<float, -1, 1> initial_rho(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.4 * mesh->_lx;
    float x_limit_upper = 0.6 * mesh->_lx;
    float y_limit_lower = 0.4 * mesh->_ly;
    float y_limit_upper = 0.6 * mesh->_ly;
    float z_limit_upper = 1 * mesh->_lz;

    for (auto& node : mesh->_nodes) {
        if ((y_limit_lower < node->y() && node->y() < y_limit_upper) && (x_limit_lower < node->x() && node->x() < x_limit_upper) && node->z() < z_limit_upper) {
            ret(i) = 5;
        } else {
            ret(i) = 1;
        }
        ++i;
    }
    return ret;
}

#define U_V_SPEED 0.1
#define W_SPEED 0.6

Eigen::Matrix<float, -1, 1> initial_u(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.48 * mesh->_lx;
    float x_limit_upper = 0.52 * mesh->_lx;
    float y_limit_lower = 0.48 * mesh->_ly;
    float y_limit_upper = 0.52 * mesh->_ly;
    float z_limit_upper = 1 * mesh->_lz;

    for (auto& node : mesh->_nodes) {
        if (node->z() < 0.1 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.2 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.3 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.4 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.5 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.6 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.7 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.8 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.9 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else {
            ret(node->_id) = U_V_SPEED;
        }
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_v(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.48 * mesh->_lx;
    float x_limit_upper = 0.52 * mesh->_lx;
    float y_limit_lower = 0.48 * mesh->_ly;
    float y_limit_upper = 0.52 * mesh->_ly;
    float z_limit_upper = 1 * mesh->_lz;

    for (auto& node : mesh->_nodes) {
        if (node->z() < 0.1 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.2 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.3 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.4 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.5 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.6 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.7 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else if (node->z() < 0.8 * mesh->_lz) {
            ret(node->_id) = U_V_SPEED;
        } else if (node->z() < 0.9 * mesh->_lz) {
            ret(node->_id) = -U_V_SPEED;
        } else {
            ret(node->_id) = U_V_SPEED;
        }
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> initial_w(Mesh3D* mesh) {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    int i = 0;

    float x_limit_lower = 0.45 * mesh->_lx;
    float x_limit_upper = 0.55 * mesh->_lx;
    float y_limit_lower = 0.45 * mesh->_ly;
    float y_limit_upper = 0.55 * mesh->_ly;

    for (auto& node : mesh->_nodes) {
        if ((y_limit_lower < node->y() && node->y() < y_limit_upper) && (x_limit_lower < node->x() && node->x() < x_limit_upper)) {
            ret(i) = W_SPEED;
        } else {
            ret(i) = W_SPEED;
        }
        ++i;
    }
    return ret;
}

Eigen::Matrix<float, -1, 1> boundary_none(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) {
    return arr;
}

Eigen::Matrix<float, -1, 1> _boundary_copy(Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const Eigen::Matrix<float, -1, 1>& copy_var) {
    auto arr1 = arr.cwiseProduct(mesh->_node_is_boundary_reverse);
    auto copy_var1 = copy_var.cwiseProduct(mesh->_node_is_boundary);
    return arr1 + copy_var1;
}

auto boundary_copy(const Eigen::Matrix<float, -1, 1>& copy_var) {
    return [copy_var] (Mesh3D* mesh, Eigen::Matrix<float, -1, 1>& arr, const DT* dt_) { return _boundary_copy(mesh, arr, copy_var); };
}

CudaDataMatrixD _boundary_copy_cu(Mesh3D* mesh_in, CudaDataMatrixD& arr, const CudaDataMatrixD& copy_var, const DT* dt_) {
    auto* mesh = dynamic_cast<CudaMesh3D*>(mesh_in);
    auto arr1 = arr * mesh->_node_is_boundary_reverse_cu;
    auto copy_var1 = copy_var * mesh->_node_is_boundary_cu;
    return arr1 + copy_var1;
}

inline auto boundary_copy_cu(const Eigen::Matrix<float, -1, 1>& copy_var) {
    CudaDataMatrixD cuda_copy_var;
    if (CFDArcoGlobalInit::cuda_enabled) {
        cuda_copy_var = CudaDataMatrixD::from_eigen(copy_var);
    }

    return [cuda_copy_var](Mesh3D *mesh, CudaDataMatrixD &arr, const DT* dt_) {
        if (CFDArcoGlobalInit::cuda_enabled)
            return _boundary_copy_cu(mesh, arr, cuda_copy_var, dt_);
        else
            return cuda_copy_var;
    };
}

inline CudaDataMatrixD boundary_none_cu(Mesh3D* mesh, CudaDataMatrixD& arr, const DT* dt_) {
    return arr;
}

int main(int argc, char **argv) {
    SingleLibInitializer3D initializer{argc, argv};
    auto mesh = initializer.mesh;
    auto timesteps = initializer.timesteps;

    auto rho_initial = initial_rho(mesh.get());
    auto rho = Variable(mesh.get(), rho_initial, boundary_copy(rho_initial), boundary_copy_cu(rho_initial), "rho");

    auto u_initial = initial_u(mesh.get());
    auto u = Variable(mesh.get(), u_initial, boundary_copy(u_initial), boundary_copy_cu(u_initial), "u");

    auto v_initial = initial_v(mesh.get());
    auto v = Variable(mesh.get(), v_initial, boundary_copy(v_initial), boundary_copy_cu(v_initial), "v");

    auto w_initial = initial_w(mesh.get());
    auto w = Variable(mesh.get(), w_initial, boundary_copy(w_initial), boundary_copy_cu(w_initial), "w");

    Eigen::Matrix<float, -1, 1> p_initial = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    p_initial.setConstant(2.5);
    auto p = Variable(mesh.get(), p_initial, boundary_copy(p_initial), boundary_copy_cu(p_initial), "p");

    Eigen::Matrix<float, -1, 1> mass_initial = rho.current.array() * mesh->_volumes.array();
    auto mass = Variable(mesh.get(), mass_initial, boundary_copy(mass_initial), boundary_copy_cu(mass_initial), "mass");

    Eigen::Matrix<float, -1, 1> rho_u_initial = rho.current.array() * u.current.array() * mesh->_volumes.array();
    auto rho_u = Variable(mesh.get(), rho_u_initial, boundary_copy(rho_u_initial), boundary_copy_cu(rho_u_initial), "rho_u");

    Eigen::Matrix<float, -1, 1> rho_v_initial = rho.current.array() * v.current.array() * mesh->_volumes.array();
    auto rho_v = Variable(mesh.get(), rho_v_initial, boundary_copy(rho_v_initial), boundary_copy_cu(rho_v_initial), "rho_v");

    Eigen::Matrix<float, -1, 1> rho_w_initial = rho.current.array() * w.current.array() * mesh->_volumes.array();
    auto rho_w = Variable(mesh.get(), rho_w_initial, boundary_copy(rho_w_initial), boundary_copy_cu(rho_w_initial), "rho_w");

    float gamma = 5. / 3.;

    auto E = p / (gamma - 1) + 0.5 * rho * ((u * u) + (v * v) + (w * w));
    Eigen::Matrix<float, -1, 1> E_initial = (p.current.array() / (gamma - 1) + 0.5 * rho.current.array() * (u.current.array() * u.current.array() + v.current.array() * v.current.array() + w.current.array() * w.current.array())) * mesh->_volumes.array();
    auto rho_e = Variable(mesh.get(), E_initial, boundary_copy(E_initial), boundary_copy_cu(E_initial), "rho_e");

    auto mass_tmp = Variable(mesh.get(), mass_initial,   boundary_none, boundary_none_cu, "mass_tmp");
    auto rho_u_tmp = Variable(mesh.get(), rho_u_initial, boundary_none, boundary_none_cu, "rho_u_tmp");
    auto rho_v_tmp = Variable(mesh.get(), rho_v_initial, boundary_none, boundary_none_cu, "rho_v_tmp");
    auto rho_w_tmp = Variable(mesh.get(), rho_w_initial, boundary_none, boundary_none_cu, "rho_w_tmp");
    auto rho_e_tmp = Variable(mesh.get(), E_initial,     boundary_none, boundary_none_cu, "rho_e_tmp");

    std::vector<Variable*> space_vars {&u, &v, &w, &p, &rho};
    auto dt = DT(mesh.get(), UpdatePolicies::CourantFriedrichsLewy3D, UpdatePolicies::CourantFriedrichsLewy3DCu, initializer.dt, space_vars);

    auto volumes_var = Variable(mesh.get(), mesh->_volumes, boundary_none, boundary_none_cu, "volumes_var");

    float dissip = 2;
    EquationTemplate equation_system = {
            {&rho,        '=', mass / volumes_var, true},
            {&u,          '=', rho_u / rho / volumes_var, true},
            {&v,          '=', rho_v / rho / volumes_var, true},
            {&w,          '=', rho_w / rho / volumes_var, true},
            {&p,          '=', (rho_e / volumes_var - 0.5 * rho * (u * u + v * v + w * w)) * (gamma - 1), true},

            {&rho,    '=', rho - 0.5 * dt * (u * d1dx(rho) + rho * d1dx(u) + v * d1dy(rho) + rho * d1dy(v) + w * d1dz(rho) + rho * d1dz(w)), true},
            {&u,      '=', u   - 0.5 * dt * (u * d1dx(u)   + v * d1dy(u)   + w * d1dz(u)   + (1 / rho) * d1dx(p)), true},
            {&v,      '=', v   - 0.5 * dt * (u * d1dx(v)   + v * d1dy(v)   + w * d1dz(v)   + (1 / rho) * d1dy(p)), true},
            {&w,      '=', w   - 0.5 * dt * (u * d1dx(w)   + v * d1dy(w)   + w * d1dz(w)   + (1 / rho) * d1dz(p)), true},
            {&p,      '=', p   - 0.5 * dt * (gamma * p * (d1dx(u) + d1dy(v) + d1dz(w)) + u * d1dx(p) + v * d1dy(p) + w * d1dz(p)), true},

            {&mass_tmp,  '=', 0 * mass_tmp -  (d1dx(rho * u) + d1dy(rho * v) + d1dz(rho * w)), true},
            {&rho_u_tmp, '=', 0 * rho_u_tmp - (d1dx(rho * u * u + p) + d1dy(rho * v * u) + d1dz(rho * w * u)), true},
            {&rho_v_tmp, '=', 0 * rho_v_tmp - (d1dx(rho * v * u) + d1dy(rho * v * v + p) + d1dz(rho * v * w)), true},
            {&rho_w_tmp, '=', 0 * rho_w_tmp - (d1dx(rho * w * u) + d1dy(rho * w * v) + d1dz(rho * w * w + p)), true},
            {&rho_e_tmp, '=', 0 * rho_e_tmp - (d1dx((E + p) * u) + d1dy((E + p) * v) + d1dz((E + p) * w)), true},

            {d1t(mass),  '=', mass_tmp -  stab_tot(rho) * dissip, false},
            {d1t(rho_u), '=', rho_u_tmp - stab_tot(rho * u) * dissip, false},
            {d1t(rho_v), '=', rho_v_tmp - stab_tot(rho * v) * dissip, false},
            {d1t(rho_w), '=', rho_w_tmp - stab_tot(rho * w) * dissip, false},
            {d1t(rho_e), '=', rho_e_tmp - stab_tot(E) * dissip, false},

    };

    auto equation = Equation(timesteps);
    initializer.init_store({&rho});

    std::vector<Variable*> all_vars {&rho, &u, &v, &w, &p, &mass, &rho_u, &rho_v, &rho_w, &rho_e};

    auto begin = std::chrono::steady_clock::now();
    equation.evaluate(all_vars, equation_system, &dt, initializer.visualize, {&rho});
    auto end = std::chrono::steady_clock::now();
    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << std::endl << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]" << std::endl;

    initializer.finalize();
    return 0;
}