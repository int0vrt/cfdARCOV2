/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2024 cfdARCHO

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
#ifndef CFDARCO_MESH3D_HPP
#define CFDARCO_MESH3D_HPP

#include "abstract_mesh.hpp"
#include "decls.hpp"
#include "cuda_data_matrix.hpp"
#include <unordered_map>
#include <memory>
#include <cstddef>


class Vertex3D;

class Face3D;

class Quadrangle3D;

class Mesh3D;


class Vertex3D : AbstractVertex {
public:
    Vertex3D(float x, float y, float z, size_t id) : _id{id}, _coords(x, y, z) {};

    void compute() override {};

    [[nodiscard]] Eigen::Matrix<float, -1, 1> coordinates() const override;

    [[nodiscard]] float x() const;

    [[nodiscard]] float y() const;

    [[nodiscard]] float z() const;

    Mesh3D *_mesh = nullptr;
    size_t _id;
    Eigen::Matrix<float, 3, 1> _coords;
};

class Face3D : AbstractFace {
public:
    Face3D(size_t vrtx_1_id, size_t vrtx_2_id, size_t vrtx_3_id, size_t vrtx_4_id, size_t id) : _id{id},
                                                                                                _vertexes_id{vrtx_1_id,
                                                                                                             vrtx_2_id,
                                                                                                             vrtx_3_id,
                                                                                                             vrtx_4_id} {};

    void compute() override;

    [[nodiscard]] bool is_boundary() const override;

    Mesh3D *_mesh = nullptr;
    size_t _id;
    bool _set = false;
    std::array<size_t, 4> _vertexes_id;
    std::vector<size_t> _nodes_id{};
    Eigen::Matrix<float, 3, 1> _normal = {0, 0, 0};
    Eigen::Matrix<float, 3, 1> _normal_alt = {0, 0, 0};
    Eigen::Matrix<float, 3, 1> _center_coords = {0, 0, 0};
    float _area = 0;
};

class Quadrangle3D : AbstractCell {
public:
    Quadrangle3D(size_t f1, size_t f2, size_t f3, size_t f4, size_t f5, size_t f6,
                 size_t v1, size_t v2, size_t v3, size_t v4, size_t v5, size_t v6, size_t v7, size_t v8,
                 size_t id,
                 size_t x_, size_t y_, size_t z_) :
            _id{id}, _faces_id{f1, f2, f3, f4, f5, f6}, _vertexes_id{v1, v2, v3, v4, v5, v6, v7, v8},
            _coord_idx{x_, y_, z_} {};

    void compute() override;

    [[nodiscard]] Eigen::Matrix<float, -1, 1> center_coords() const override;

    [[nodiscard]] bool is_boundary() const override;

    [[nodiscard]] bool is_boundary_x() const;
    [[nodiscard]] bool is_boundary_y() const;
    [[nodiscard]] bool is_boundary_z() const;

    [[nodiscard]] float x() const;

    [[nodiscard]] float y() const;

    [[nodiscard]] float z() const;

    [[nodiscard]] static size_t opposite_face_id(size_t face_id);

    Mesh3D *_mesh = nullptr;
    size_t _id;
    std::array<size_t, 6> _faces_id;
    std::array<size_t, 8> _vertexes_id;
    Eigen::Matrix<float, 3, 1> _center_coords = {0, 0, 0};
    std::array<size_t, 3> _coord_idx = {0, 0, 0};
    Eigen::Matrix<float, 6, 3> _normals = {};
    Eigen::Matrix<float, 6, 3> _normals_alt = {};
    float _volume = 0;
};

class Mesh3D : AbstractMesh {
public:
    Mesh3D(size_t x, size_t y, size_t z, float dx, float dy, float dz) : _num_nodes{x * y * z}, _x{x}, _y{y}, _z{z},
                                                                            _lx{dx * static_cast<float>(x)},
                                                                            _ly{dy * static_cast<float>(y)},
                                                                            _lz{dz * static_cast<float>(z)}, _dx{dx},
                                                                            _dy{dy}, _dz{dz} {};

    void compute() override;

    void init_basic_internals();


    [[nodiscard]] size_t square_node_coord_to_idx(size_t x, size_t y, size_t z) const;

    [[nodiscard]] std::array<size_t, 6> square_node_coord_to_face_idx(size_t x, size_t y, size_t z) const;

    [[nodiscard]] std::array<size_t, 8> square_node_coord_to_vertex_idx(size_t x, size_t y, size_t z) const;

    size_t _num_nodes;

    size_t _x;
    size_t _y;
    size_t _z;

    float _lx;
    float _ly;
    float _lz;

    float _dx;
    float _dy;
    float _dz;

    std::vector<std::shared_ptr<Vertex3D>> _vertexes{};
    std::vector<std::shared_ptr<Face3D>> _faces{};
    std::vector<std::shared_ptr<Quadrangle3D>> _nodes{};

    MatrixX6dRB _normal_x{};
    MatrixX6dRB _normal_y{};
    MatrixX6dRB _normal_z{};
    std::vector<MatrixX6dRB *> _normals_all{};

    MatrixX6dRB _normal_alt_x{};
    MatrixX6dRB _normal_alt_y{};
    MatrixX6dRB _normal_alt_z{};
    std::vector<MatrixX6dRB *> _normals_alt_all{};

    MatrixX6Idx _ids{};
    MatrixX6SignIdx _ids_bound_free{};

    MatrixX6dRB _alpha_d{};
    std::vector<MatrixX6dRB> _n_min_alpha_d{};
    MatrixX6dRB _len_node_center_to_face{};

    Eigen::Matrix<float, -1, 1> _volumes{};
    MatrixX6dRB _face_areas{};

    Eigen::Matrix<float, -1, 1> _node_is_boundary{};
    Eigen::Matrix<float, -1, 1> _node_is_boundary_reverse{};

    static constexpr int n_dims = 3;
    static constexpr int n_faces = 6;

};

class CudaMesh3D : public Mesh3D {
public:

    using Mesh3D::Mesh3D;

    CudaDataMatrixD _normal_x_cu{};
    CudaDataMatrixD _normal_y_cu{};
    CudaDataMatrixD _normal_z_cu{};

    CudaDataMatrixD _normal_alt_x_cu{};
    CudaDataMatrixD _normal_alt_y_cu{};
    CudaDataMatrixD _normal_alt_z_cu{};

    CudaDataMatrix<size_t> _ids_cu{};
    CudaDataMatrix<ptrdiff_t> _ids_bound_free_cu{};

    CudaDataMatrixD _alpha_d_cu{};
    std::array<CudaDataMatrixD, n_dims> _n_min_alpha_d_cu{};
    CudaDataMatrixD _len_node_center_to_face_cu{};

    CudaDataMatrixD _volumes_cu{};
    CudaDataMatrixD _face_areas_cu{};

    CudaDataMatrixD _node_is_boundary_cu{};
    CudaDataMatrixD _node_is_boundary_reverse_cu{};

    inline void host_to_device() {
        _normal_x_cu = CudaDataMatrixD::from_eigen(_normal_x);
        _normal_y_cu = CudaDataMatrixD::from_eigen(_normal_y);
        _normal_z_cu = CudaDataMatrixD::from_eigen(_normal_z);

        _normal_alt_x_cu = CudaDataMatrixD::from_eigen(_normal_alt_x);
        _normal_alt_y_cu = CudaDataMatrixD::from_eigen(_normal_alt_y);
        _normal_alt_z_cu = CudaDataMatrixD::from_eigen(_normal_alt_z);

        _ids_cu = CudaDataMatrix<size_t>::from_eigen(_ids);
        _ids_bound_free_cu = CudaDataMatrix<ptrdiff_t>::from_eigen(_ids_bound_free);

        _alpha_d_cu = CudaDataMatrixD::from_eigen(_alpha_d);
        _n_min_alpha_d_cu = {
                CudaDataMatrixD::from_eigen(_n_min_alpha_d.at(0)),
                CudaDataMatrixD::from_eigen(_n_min_alpha_d.at(1)),
                CudaDataMatrixD::from_eigen(_n_min_alpha_d.at(2))
        };
        _len_node_center_to_face_cu = CudaDataMatrixD::from_eigen(_len_node_center_to_face);

        _volumes_cu = CudaDataMatrixD::from_eigen(_volumes);
        _face_areas_cu = CudaDataMatrixD::from_eigen(_face_areas);

        _node_is_boundary_cu = CudaDataMatrixD::from_eigen(_node_is_boundary);
        _node_is_boundary_reverse_cu = CudaDataMatrixD::from_eigen(_node_is_boundary_reverse);

//        std::cout << "Mesh data moved to GPU" << std::endl;
    }
};

#endif //CFDARCO_MESH3D_HPP
