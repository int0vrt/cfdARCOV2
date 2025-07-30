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
#include "variables/spatial_second_derivative.hpp"
#include "grad_utils.hpp"

std::shared_ptr<Variable> Grad2Var::clone() const {
    auto *new_obj = new Grad2Var(*this);
    return std::shared_ptr<Variable>{new_obj};
}

Grad2Var::Grad2Var(Variable *var_, bool clc_x_, bool clc_y_, bool clc_z_) : clc_x{clc_x_}, clc_y{clc_y_},
                                                                            clc_z{clc_z_} {
    var = std::shared_ptr<Variable>{var_->clone()};
    mesh = var_->mesh;
    name = "Lapl_" + std::to_string(clc_x_) + std::to_string(clc_y_) + std::to_string(clc_z_) + var_->name;
    is_subvariable = true;

    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        auto cu_mesh = dynamic_cast<CudaMesh3D *>(var_->mesh);

        jit_declarations = {
                fmt::format("__shared__ __align__(16) {} face_neigh_ids[pipeline_stages][BLOCKSIZE * {}];", cu_mesh->_ids_data_type, Mesh3D::n_faces),
                fmt::format("__shared__ __align__(16) float volume_shmem[pipeline_stages][BLOCKSIZE];"),
                fmt::format("__shared__ __align__(16) float face_area_shmem[pipeline_stages][BLOCKSIZE * {}];",
                            Mesh3D::n_faces),
                fmt::format("__shared__ __align__(16) float alpha_d_shmem[pipeline_stages][BLOCKSIZE * {}];",
                            Mesh3D::n_faces),
                fmt::format("__shared__ __align__(16) float normals_face_shmem[pipeline_stages][BLOCKSIZE * {} * {}];",
                            Mesh3D::n_dims, Mesh3D::n_faces),
                fmt::format("float face_area[{}];", Mesh3D::n_faces),
                fmt::format("float alpha_d[{}];", Mesh3D::n_faces),
                fmt::format("float normals_face[{}][{}];", Mesh3D::n_dims, Mesh3D::n_faces),
                fmt::format("const float* normals_ptr[{}] = {{\n"
                            "            normals_x_ptr,\n"
                            "            normals_y_ptr,\n"
                            "            normals_z_ptr\n"
                            "    }};", Mesh3D::n_dims),
                fmt::format("float corrected_surface_normal_grad_{}[{}];", var_->name, Mesh3D::n_faces),
                fmt::format("float lapl_{}[{}];", name, Mesh3D::n_dims),
                fmt::format("float volume;"),
                fmt::format("float {} = 0;", name),
                fmt::format("float {}_neigh[{}] = {{ 0, 0, 0, 0, 0, 0 }};", name, Mesh3D::n_faces),
        };
        jit_memops = {
                fmt::format("read_n_vars_from_self_pipe<1>(volume_ptr, &(volume_shmem[pipe][0]), idx, n);"),
                fmt::format(
                        "read_idxs_from_self_pipe<Mesh3D::n_faces>(ids_ptr, &(face_neigh_ids[pipe][0]), idx, n);"),
                fmt::format(
                        "read_n_vars_from_self_pipe<Mesh3D::n_faces>(face_area_ptr, &(face_area_shmem[pipe][0]), idx, n);"),
                fmt::format(
                        "read_n_vars_from_self_pipe<Mesh3D::n_faces>(alpha_d_ptr, &(alpha_d_shmem[pipe][0]), idx, n);"),
                fmt::format(
                        "read_normals_from_self_pipe<Mesh3D>(normals_ptr, &(normals_face_shmem[pipe][0]), idx, n);"),
        };
        jit_operations = {
                fmt::format("volume = volume_shmem[pipe][threadIdx.x];"),
                fmt::format("read_n_vars_from_shmem_to_reg<Mesh3D::n_faces>(&(face_area_shmem[pipe][0]), face_area);"),
                fmt::format("read_n_vars_from_shmem_to_reg<Mesh3D::n_faces>(&(alpha_d_shmem[pipe][0]), alpha_d);"),
                fmt::format("read_normals_from_shared_to_neigh<Mesh3D>(&(normals_face_shmem[pipe][0]), normals_face);"),

                fmt::format(
                        "corrected_surface_normal_grad_cu_k<Mesh3D>({0}, {0}_neigh, alpha_d, corrected_surface_normal_grad_{0});",
                        var_->name),
                fmt::format(
                        "gauss_grad_cu_k<Mesh3D>(corrected_surface_normal_grad_{0}, normals_face, face_area, volume, lapl_{1});",
                        var_->name, name),
                fmt::format("{{"
                            "        {0} = 0;\n"
                            "        if ({1}) {0} += lapl_{0}[0];\n"
                            "        if ({2}) {0} += lapl_{0}[1];\n"
                            "        if ({3}) {0} += lapl_{0}[2];\n"
                            "        lapl_{0}[0] = 0;\n"
                            "        lapl_{0}[1] = 0;\n"
                            "        lapl_{0}[2] = 0;\n"
                            "}}", name, clc_x_, clc_y_, clc_z_)
        };
        jit_returns = {name};
    }
}

MatrixX4dRB Grad2Var::evaluate() {
    Eigen::Matrix<float, -1, 1> crr = var->evaluate();
    std::vector<Eigen::Matrix<float, -1, 1>> grads = {};

    MatrixX6dRB surface_normal_grads = corrected_surface_normal_grad(var->mesh, grads, &(var->current));
    auto lapl = gauss_grad<Mesh3D, true>(var->mesh, &surface_normal_grads);

    std::vector<bool> fgs = {clc_x, clc_y, clc_z};
    Eigen::Matrix<float, -1, 1> res = Eigen::Matrix<float, -1, 1>(mesh->_num_nodes);
    res.setConstant(0);

    for (int i = 0; i < fgs.size(); ++i) {
        if (fgs.at(i)) {
            res += lapl.at(i);
        }
    }

    return res;
}

CudaDataMatrixD Grad2Var::evaluate_cu() {
    auto crr = var->evaluate_cu();
    auto gr2 = eval_grad2(dynamic_cast<CudaMesh3D *>(var->mesh), crr, clc_x, clc_y, clc_z);
    return gr2;
}

std::vector<std::string> Grad2Var::get_jit_declarations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_declarations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_declarations.begin(), jit_declarations.end());

    return ret;
}

std::vector<std::tuple<std::string, void *, TypeEnum>> Grad2Var::get_jit_inputs() {
    reset_input_pointers();
    std::vector<std::tuple<std::string, void *, TypeEnum>> ret = {};
    auto val_var = var->get_jit_inputs();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_inputs.begin(), jit_inputs.end());

    return ret;
}

std::vector<std::string> Grad2Var::get_jit_operations() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_operations();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_operations.begin(), jit_operations.end());

    return ret;
}

std::vector<std::string> Grad2Var::get_jit_memops() {
    std::vector<std::string> ret = {};
    auto val_var = var->get_jit_memops();

    ret.insert(ret.end(), val_var.begin(), val_var.end());
    ret.insert(ret.end(), jit_memops.begin(), jit_memops.end());

    return ret;
}

void Grad2Var::reset_input_pointers() {
    auto cu_mesh = dynamic_cast<CudaMesh3D *>(var->mesh);
    jit_inputs = {
            {fmt::format("const {} * __restrict__ ids_ptr", cu_mesh->_ids_data_type), cu_mesh->_ids_mtrx_ptr, cu_mesh->_ids_data_type_en},
            {fmt::format(
                    "const float * __restrict__ normals_x_ptr"),      &cu_mesh->_normal_x_cu,   TypeEnum::ConstDoublePointer},
            {fmt::format(
                    "const float * __restrict__ normals_y_ptr"),      &cu_mesh->_normal_y_cu,   TypeEnum::ConstDoublePointer},
            {fmt::format(
                    "const float * __restrict__ normals_z_ptr"),      &cu_mesh->_normal_z_cu,   TypeEnum::ConstDoublePointer},
            {fmt::format(
                    "const float * __restrict__ face_area_ptr"),      &cu_mesh->_face_areas_cu, TypeEnum::ConstDoublePointer},
            {fmt::format(
                    "const float * __restrict__ alpha_d_ptr"),        &cu_mesh->_alpha_d_cu,    TypeEnum::ConstDoublePointer},
            {fmt::format(
                    "const float * __restrict__ volume_ptr"),         &cu_mesh->_volumes_cu,    TypeEnum::ConstDoublePointer},

    };
}
