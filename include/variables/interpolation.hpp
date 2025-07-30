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
#include "cuda_operators.hpp"

class InterpVar : public Variable {
public:
    explicit InterpVar(Variable *var_, bool clc_x_ = true, bool clc_y_ = true, bool clc_z_ = true, bool inv_ = false);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;

    std::vector<std::tuple<std::string, void *, TypeEnum>> get_jit_inputs() override;

    std::vector<std::string> get_jit_operations() override;

    std::vector<std::string> get_jit_memops() override;

    void reset_input_pointers() override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
    bool clc_z;
    bool inv;
};


inline auto interp_x(Variable &var, bool inv = false) {
    return InterpVar(&var, true, false, false, inv);
}

inline auto interp_x(Variable &&var, bool inv = false) {
    return InterpVar(&var, true, false, false, inv);
}

inline auto interp_y(Variable &var, bool inv = false) {
    return InterpVar(&var, false, true, false, inv);
}

inline auto interp_y(Variable &&var, bool inv = false) {
    return InterpVar(&var, false, true, false, inv);
}

inline auto interp_z(Variable &var, bool inv = false) {
    return InterpVar(&var, false, false, true, inv);
}

inline auto interp_z(Variable &&var, bool inv = false) {
    return InterpVar(&var, false, false, true, inv);
}

inline auto interp(Variable &var, bool inv = false) {
    return InterpVar(&var, true, true, true, inv);
}

inline auto interp(Variable &&var, bool inv = false) {
    return InterpVar(&var, true, true, true, inv);
}

template<bool x, bool y, bool z>
inline auto interp_d(Variable &var, bool inv = false) {
    return InterpVar(&var, x, y, z, inv);
}

template<bool x, bool y, bool z>
inline auto interp_d(Variable &&var, bool inv = false) {
    return InterpVar(&var, x, y, z, inv);
}
