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

class StabVar : public Variable {
public:
    explicit StabVar(Variable *var_, bool clc_x_ = true, bool clc_y_ = true, bool clc_z_ = true);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::shared_ptr<Variable> var;
    bool clc_x;
    bool clc_y;
    bool clc_z;
};


inline auto stab_tot(Variable &var) {
    return StabVar(&var, true, true, true);
}

inline auto stab_tot(Variable &&var) {
    return StabVar(&var, true, true, true);
}

inline auto stabx(Variable &var) {
    return StabVar(&var, true, false, false);
}

inline auto stabx(Variable &&var) {
    return StabVar(&var, true, false, false);
}

inline auto staby(Variable &var) {
    return StabVar(&var, false, true, false);
}

inline auto staby(Variable &&var) {
    return StabVar(&var, false, true, false);
}

inline auto stabz(Variable &var) {
    return StabVar(&var, false, false, true);
}

inline auto stabz(Variable &&var) {
    return StabVar(&var, false, false, true);
}
