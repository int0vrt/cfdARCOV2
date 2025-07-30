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
#include "variables/dissipation.hpp"
#include "grad_utils.hpp"


std::shared_ptr<Variable> StabVar::clone() const {
    auto *new_obj = new StabVar(*this);
    return std::shared_ptr<Variable>{new_obj};
}

StabVar::StabVar(Variable *var_, bool clc_x_, bool clc_y_, bool clc_z_) : clc_x{clc_x_}, clc_y{clc_y_}, clc_z{clc_z_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    is_subvariable = true;
}

MatrixX4dRB StabVar::evaluate() {
    std::vector<bool> fgs = {clc_x, clc_y, clc_z};
    Eigen::Matrix<float, -1, 1> crr = var->evaluate();

    auto interpolated = interpolate_to_face_linear(var->mesh, &crr);
    auto grads = gauss_grad<Mesh3D, false>(var->mesh, &interpolated);

    auto interpolated_upwing = interpolate_to_face_upwing(mesh, &crr, &grads);

    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);

    for (int i = 0; i < fgs.size(); ++i) {
        if (fgs.at(i)) {
            MatrixX6dRB collected = collect_vals_neigh_faces<Mesh3D, false>(mesh, &(interpolated_upwing.at(i)));
            MatrixX6dRB summed_faces = 0.5 * (interpolated_upwing.at(i) - collected);
            auto appliedStabVar = gauss_grad<Mesh3D, true>(var->mesh, &summed_faces);
            res += appliedStabVar.at(i);
        }
    }

    return res;
}

CudaDataMatrixD StabVar::evaluate_cu() {
    auto crr = var->evaluate_cu();
    auto gr2 = eval_stab(dynamic_cast<CudaMesh3D *>(var->mesh), crr, clc_x, clc_y, clc_z);
    return gr2;
}
