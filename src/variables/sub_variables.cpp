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
#include "variables/sub_variables.hpp"

std::shared_ptr<Variable> PointerVariable::clone() const {
    return std::shared_ptr<Variable>{const_cast<PointerVariable *>(this), [](Variable *) {}};
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

std::shared_ptr<Variable> NEntryVarReal::clone() const {
    auto *new_obj = new NEntryVarReal(*this);
    return std::shared_ptr<Variable>{new_obj};
}

std::vector<std::string> NEntryVarReal::get_jit_declarations() {
    std::vector<std::string> ret = {};
    for (auto var_enrty: left_vars_list) {
        auto val_var = var_enrty->get_jit_declarations();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    for (auto &var_enrty: right_vars_list) {
        auto val_var = var_enrty.get_jit_declarations();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }

    return ret;
}

std::vector<std::tuple<std::string, void *, TypeEnum>> NEntryVarReal::get_jit_inputs() {
    reset_input_pointers();
    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};

    for (auto var_enrty: left_vars_list) {
        auto val_var = var_enrty->get_jit_inputs();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    for (auto &var_enrty: right_vars_list) {
        auto val_var = var_enrty.get_jit_inputs();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }

    return ret;
}

std::vector<std::string> NEntryVarReal::get_jit_operations() {
    std::vector<std::string> ret = {};
    for (auto var_enrty: left_vars_list) {
        auto val_var = var_enrty->get_jit_operations();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    for (auto &var_enrty: right_vars_list) {
        auto val_var = var_enrty.get_jit_operations();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    return ret;
}

std::vector<std::string> NEntryVarReal::get_jit_memops() {
    std::vector<std::string> ret = {};
    for (auto var_enrty: left_vars_list) {
        auto val_var = var_enrty->get_jit_memops();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    for (auto &var_enrty: right_vars_list) {
        auto val_var = var_enrty.get_jit_memops();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    return ret;
}

void NEntryVarReal::reset_input_pointers() {
    for (auto var_enrty: left_vars_list) {
        var_enrty->reset_input_pointers();
    }
    for (auto &var_enrty: right_vars_list) {
        var_enrty.reset_input_pointers();
    }
}

std::vector<std::string> NEntryVarReal::get_jit_returns() {
    std::vector<std::string> ret = {};
    for (auto var_enrty: left_vars_list) {
        auto val_var = var_enrty->get_jit_returns();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    for (auto &var_enrty: right_vars_list) {
        auto val_var = var_enrty.get_jit_returns();
        ret.insert(ret.end(), val_var.begin(), val_var.end());
    }
    return ret;
}

NEntryVarReal::NEntryVarReal(EquationTemplate &init_vars_list, bool is_left_part) {
    is_left_part_ = is_left_part;
    is_subvariable = false;
    is_constvar = true;

    if (is_left_part) {
        for (auto &eq: init_vars_list) {
            left_vars_list.push_back(std::get<0>(eq));
            jit_assign.push_back(left_vars_list.at(left_vars_list.size() - 1)->jit_assign.at(0));

            mesh = left_vars_list.at(left_vars_list.size() - 1)->mesh;
            num_nodes = left_vars_list.at(left_vars_list.size() - 1)->num_nodes;
        }
    } else {
        for (auto &eq: init_vars_list) {
            right_vars_list.push_back(std::move(std::get<2>(eq)));
            mesh = right_vars_list.at(right_vars_list.size() - 1).mesh;
            num_nodes = right_vars_list.at(right_vars_list.size() - 1).num_nodes;
        }
    }
}

SingleEquationTemplate NEntryVar::restruct() {
    auto *left_eqs = new NEntryVarReal{vars_list, true};
    auto right_eqs = NEntryVarReal{vars_list, false};

    bool use_kern = true;
    for (auto &eq: vars_list) {
        use_kern &= std::get<3>(eq);
    }

    if (use_kern) {
        left_eqs->jit_declarations = left_eqs->get_jit_declarations();
        left_eqs->jit_memops = left_eqs->get_jit_memops();
        left_eqs->jit_operations = left_eqs->get_jit_operations();
        left_eqs->jit_returns = left_eqs->get_jit_returns();
        left_eqs->jit_inputs = left_eqs->get_jit_inputs();

        right_eqs.jit_declarations = right_eqs.get_jit_declarations();
        right_eqs.jit_memops = right_eqs.get_jit_memops();
        right_eqs.jit_operations = right_eqs.get_jit_operations();
        right_eqs.jit_returns = right_eqs.get_jit_returns();
        right_eqs.jit_inputs = right_eqs.get_jit_inputs();
    }

    return {left_eqs, '=', right_eqs, use_kern};
}
