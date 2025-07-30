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
#include "variables/dt_update.hpp"

#include <functional>

using SourceFillFn = std::function<float(float current_time)>;

class PointSource : public Variable {
public:
    PointSource(Mesh3D *mesh_, DT *dt_, int timesteps_, float x_, float y_, float z_, float radius, SourceFillFn fill_fn_, std::string name_suff);

    MatrixX4dRB evaluate() override;

    CudaDataMatrixD evaluate_cu() override;

    [[nodiscard]] std::shared_ptr<Variable> clone() const override;

    std::vector<std::string> get_jit_declarations() override;

    std::vector<std::tuple<std::string, void *, TypeEnum>> get_jit_inputs() override;

    std::vector<std::string> get_jit_operations() override;

    std::vector<std::string> get_jit_memops() override;

    void reset_input_pointers() override;

    DT *_dt;
    int _timesteps;
    float _x;
    float _y;
    float _z;
    size_t _base_idx;
    SourceFillFn _fill_fn;

    std::vector<float> values;
    CudaDataMatrixD values_cu;

    std::vector<uint32_t> mask;
    CudaDataMatrix<uint32_t> mask_cu;
};
