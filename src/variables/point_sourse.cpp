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
#include "variables/point_sourse.hpp"
#include "cuda_operators.hpp"

std::shared_ptr<Variable> PointSource::clone() const {
    auto *new_obj = new PointSource(*this);
    new_obj->mask = this->mask;
    new_obj->mask_cu = this->mask_cu;
    new_obj->values = this->values;
    new_obj->values_cu = this->values_cu;
    return std::shared_ptr<Variable>{new_obj};
}


PointSource::PointSource(Mesh3D *mesh_, DT *dt_, int timesteps_, float x_, float y_, float z_, float radius, SourceFillFn fill_fn_, std::string name_suff) {
    name = "point_source_" + name_suff;
    mesh = mesh_;
    _dt = dt_;
    _x = x_;
    _y = y_;
    _z = z_;
    _timesteps = timesteps_;
    _fill_fn = fill_fn_;

    for (int i = 0; i < mesh->_num_nodes; ++i) {
        auto& node = mesh->_nodes[i];
        float x = node.x();
        float y = node.y();
        float z = node.z();

        float r1 = std::sqrt((x - x_) * (x - x_) + (y - y_) * (y - y_) + (z - z_) * (z - z_));

        if (r1 < radius) {
            mask.push_back(1);
        } else {
            mask.push_back(0);
        }
    }

    for (int i = 0; i < timesteps_; ++i) {
        values.push_back(fill_fn_(static_cast<float>(i) * dt_->_dt));
    }

    values_cu = CudaDataMatrixD{values.data(), static_cast<size_t>(timesteps_)};
    mask_cu = CudaDataMatrix<uint32_t>{mask.data(), mesh->_num_nodes};

    jit_declarations = {
            fmt::format("float {0} = 0.0;", name),
            fmt::format("float {0}_neigh[{1}];", name, Mesh3D::n_faces),
    };
    jit_operations = {
            fmt::format("if ({0}_mask_ptr[idx]) {{ {0} = {0}_ptr[current_timestep]; }} else {{ {0} = 0.0f; }}", name),

            fmt::format("{0}_neigh[0] = {0};\n"
                        "{0}_neigh[1] = {0};\n"
                        "{0}_neigh[2] = {0};\n"
                        "{0}_neigh[3] = {0};\n"
                        "{0}_neigh[4] = {0};\n"
                        "{0}_neigh[5] = {0};\n", name),
    };
    jit_memops = {};
    jit_returns = {name};
}

std::vector<std::string> PointSource::get_jit_declarations() {
    return jit_declarations;
}

std::vector<std::tuple<std::string, void *, TypeEnum>> PointSource::get_jit_inputs() {
    reset_input_pointers();

    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> PointSource::get_jit_operations() {
    return jit_operations;
}

std::vector<std::string> PointSource::get_jit_memops() {
    return jit_memops;
}



void PointSource::reset_input_pointers() {
    jit_inputs = {
            {fmt::format("uint32_t * __restrict__ {0}_mask_ptr", name), &mask_cu, TypeEnum::ConstUint32TPointer},
            {fmt::format("float * __restrict__ {0}_ptr", name), &values_cu, TypeEnum::DoublePointer},
            {fmt::format("size_t current_timestep"), &(_dt->_current_time_step_int), TypeEnum::SizeT},
    };
}

MatrixX4dRB PointSource::evaluate() {
    auto ret = Eigen::Matrix<float, -1, 1>{mesh->_num_nodes};
    throw std::runtime_error{"Not implemented"};
    return ret;
}

CudaDataMatrixD PointSource::evaluate_cu() {
    throw std::runtime_error{"Not implemented"};
    return CudaDataMatrixD{};
}

