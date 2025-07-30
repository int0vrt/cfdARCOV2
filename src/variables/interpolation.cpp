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
#include "variables/interpolation.hpp"
#include "grad_utils.hpp"

std::shared_ptr<Variable> InterpVar::clone() const {
    auto *new_obj = new InterpVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}


InterpVar::InterpVar(Variable *var_, bool clc_x_, bool clc_y_, bool clc_z_, bool inv_) : clc_x{clc_x_}, clc_y{clc_y_},
                                                                                         clc_z{clc_z_}, inv{inv_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    num_nodes = var->num_nodes;
    name = "InterpVar_" + std::to_string(clc_x_) + std::to_string(clc_y_) + std::to_string(clc_z_) +
           std::to_string(inv_) + var_->name;
    is_subvariable = true;

    auto cu_mesh = dynamic_cast<CudaMesh3D *>(var_->mesh);

    jit_declarations = {
            fmt::format("float interpolation_ret_{}[{}];", var_->name, Mesh3D::n_faces),
            fmt::format("float {} = 0;", name),
            fmt::format("float {}_neigh[{}] = {{ 0, 0, 0, 0, 0, 0 }};", name, Mesh3D::n_faces),
    };
    jit_memops = {};
    jit_operations = {
            fmt::format("interpolate_to_face_linear_cu_k<Mesh3D>({0}, {0}_neigh, interpolation_ret_{0});", var_->name),

            fmt::format("{{\n"
                        "        {0} = 0;\n"
                        "        if (!{5}) {{\n"
                        "           if ({1}) {0} += interpolation_ret_{4}[0];\n"
                        "           if ({2}) {0} += interpolation_ret_{4}[2];\n"
                        "           if ({3}) {0} += interpolation_ret_{4}[4];\n"
                        "        }} else {{\n"
                        "           if ({1}) {0} += interpolation_ret_{4}[1];\n"
                        "           if ({2}) {0} += interpolation_ret_{4}[3];\n"
                        "           if ({3}) {0} += interpolation_ret_{4}[5];\n"
                        "        }}\n"
                        "        {0} /= ((int) {1} + (int) {2} + (int) {3});  \n"
                        "}}", name, clc_x_, clc_y_, clc_z_, var_->name, inv_),
    };
    jit_returns = {name};
}

MatrixX4dRB InterpVar::evaluate() {
    Eigen::Matrix<float, -1, 1> crr = var->evaluate();
    auto interpolated = interpolate_to_face_linear(var->mesh, &crr);

    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);

    std::vector<bool> fgs = {clc_x, clc_y, clc_z};
    if (!inv) {
        if (clc_x) {
            res += interpolated(Eigen::all, 0);
        }
        if (clc_y) {
            res += interpolated(Eigen::all, 2);
        }
        if (clc_z) {
            res += interpolated(Eigen::all, 4);
        }
    } else {
        if (clc_x) {
            res += interpolated(Eigen::all, 1);
        }
        if (clc_y) {
            res += interpolated(Eigen::all, 3);
        }
        if (clc_z) {
            res += interpolated(Eigen::all, 5);
        }
    }


    res /= ((int) clc_x + (int) clc_y + (int) clc_z);

    return res;
}

CudaDataMatrixD InterpVar::evaluate_cu() {
    auto crr = var->evaluate_cu();
    auto gr = eval_interp(dynamic_cast<CudaMesh3D *>(var->mesh), crr, clc_x, clc_y, clc_z, inv);
    return gr;
}

std::vector<std::string> InterpVar::get_jit_declarations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_declarations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_declarations.begin(), jit_declarations.end());

    return ret;
}

std::vector<std::tuple<std::string, void *, TypeEnum>> InterpVar::get_jit_inputs() {
    reset_input_pointers();
    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};
    auto val_var = var->get_jit_inputs();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> InterpVar::get_jit_operations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_operations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_operations.begin(), jit_operations.end());

    return ret;
}

std::vector<std::string> InterpVar::get_jit_memops() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_memops();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_memops.begin(), jit_memops.end());

    return ret;
}

void InterpVar::reset_input_pointers() {
    auto cu_mesh = dynamic_cast<CudaMesh3D *>(var->mesh);
    jit_inputs = {
            {fmt::format("const {} * __restrict__ ids_ptr", cu_mesh->_ids_data_type), cu_mesh->_ids_mtrx_ptr, cu_mesh->_ids_data_type_en},
    };
}
