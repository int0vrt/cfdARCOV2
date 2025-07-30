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

#include "variables/variable.hpp"

class NEntryVar;

using SingleEquationTemplate = std::tuple<Variable *, char, Variable, bool>;
using EquationTemplate = std::vector<SingleEquationTemplate>;
using NEquationTemplate = std::vector<NEntryVar>;

class PointerVariable : public Variable {
public:
    PointerVariable(Mesh3D *mesh_, float *ptr);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    float *_ptr = nullptr;
};

class NEntryVar : public Variable {
public:
    NEntryVar(EquationTemplate vars_list_) : vars_list{std::move(vars_list_)} {};

    NEntryVar(SingleEquationTemplate vars_list_) : vars_list{EquationTemplate{std::move(vars_list_)}} {};

    NEntryVar(std::initializer_list<SingleEquationTemplate> vars_list_) : vars_list{std::move(vars_list_)} {};

    EquationTemplate vars_list;

    SingleEquationTemplate restruct();
};

class NEntryVarReal : public Variable {
public:
    NEntryVarReal(EquationTemplate &init_vars_list, bool is_left_part);

    virtual std::vector<std::string> get_jit_declarations() override;

    virtual std::vector<std::tuple<std::string, void *, TypeEnum>> get_jit_inputs() override;

    virtual std::vector<std::string> get_jit_operations() override;

    virtual std::vector<std::string> get_jit_memops() override;

    virtual void reset_input_pointers() override;

    virtual std::vector<std::string> get_jit_returns() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    bool is_left_part_;
    std::vector<Variable *> left_vars_list{};
    std::vector<Variable> right_vars_list{};
};
