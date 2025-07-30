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
#include "variables/sub_variables.hpp"

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
