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
// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com


#include <iostream>
#include <memory>
#include <indicators/progress_bar.hpp>
#include "mesh3d.hpp"

Eigen::Matrix<float, -1, 1> Vertex3D::coordinates() const {
    return _coords;
}

float Vertex3D::x() const {
    return _coords(0);
}

float Vertex3D::y() const {
    return _coords(1);
}

float Vertex3D::z() const {
    return _coords(2);
}

void Face3D::compute() {
    std::vector<Eigen::Matrix<float, 3, 1>> coords{};
    _center_coords = {0, 0, 0};
    for (unsigned long i: _vertexes_id) {
        coords.push_back(_mesh->_vertexes.at(i)->_coords);
        _center_coords += _mesh->_vertexes.at(i)->_coords;
    }

    auto AB = coords.at(1) - coords.at(0);
    auto BC = coords.at(2) - coords.at(1);
    auto CD = coords.at(3) - coords.at(2);
    auto AD = coords.at(3) - coords.at(0);
    auto CB = coords.at(1) - coords.at(2);

    _normal = AB.cross(BC);
    _normal = _normal.normalized();
    _center_coords /= _vertexes_id.size();

    _area = 0.5 * (std::abs(AB.cross(AD).sum()) + std::abs(CB.cross(CD).sum()));
}

bool Face3D::is_boundary() const {
    return _nodes_id.size() == 1;
}


void Quadrangle3D::compute() {
    std::vector<Eigen::Matrix<float, 3, 1>> coords{};
    _center_coords = {0, 0, 0};
    for (unsigned long i: _vertexes_id) {
        coords.push_back(_mesh->_vertexes.at(i)->_coords);
        _center_coords += _mesh->_vertexes.at(i)->_coords;
    }
    _center_coords /= _vertexes_id.size();

    _volume = 0;
    for (int i = 0; i < _faces_id.size(); ++i) {
        auto normal = _mesh->_faces.at(_faces_id.at(i))->_normal;
        auto face_center = _mesh->_faces.at(_faces_id.at(i))->_center_coords;
        auto face_area = _mesh->_faces.at(_faces_id.at(i))->_area;

        Eigen::Matrix<float, 1, 3> good_norm;
        if (normal.dot(face_center - _center_coords) >= 0) {
            good_norm = normal.transpose();
        } else {
            good_norm = -normal.transpose();
        }


        if (i == 4 || i == 5 || i == 0 || i == 1) {
            _normals(i, Eigen::all) = -normal.transpose();
        } else {
            _normals(i, Eigen::all) = normal.transpose();
        }

        _normals_alt(i, Eigen::all) = good_norm;


        _volume += face_center.dot(good_norm) * face_area;
    }

    _volume /= 3;
}

Eigen::Matrix<float, -1, 1> Quadrangle3D::center_coords() const {
    return _center_coords;
}

bool Quadrangle3D::is_boundary() const {
    return std::any_of(_faces_id.begin(), _faces_id.end(),
                       [&](size_t face_id) {
                           return _mesh->_faces.at(face_id)->is_boundary();
                       });
}

bool Quadrangle3D::is_boundary_x() const {
    return _mesh->_faces.at(4)->is_boundary() || _mesh->_faces.at(5)->is_boundary();
}
bool Quadrangle3D::is_boundary_y() const {
    return _mesh->_faces.at(2)->is_boundary() || _mesh->_faces.at(3)->is_boundary();
}
bool Quadrangle3D::is_boundary_z() const {
    return _mesh->_faces.at(0)->is_boundary() || _mesh->_faces.at(1)->is_boundary();
}

float Quadrangle3D::x() const {
    return _center_coords(0);
}

float Quadrangle3D::y() const {
    return _center_coords(1);
}

float Quadrangle3D::z() const {
    return _center_coords(2);
}

size_t Quadrangle3D::opposite_face_id(size_t face_id) {
    if (face_id == 0) return 1;
    if (face_id == 1) return 0;
    if (face_id == 2) return 3;
    if (face_id == 3) return 2;
    if (face_id == 4) return 5;
    if (face_id == 5) return 4;
    return 0;
}


void Mesh3D::compute() {
#pragma omp parallel for
    for (auto &entry: _vertexes) {
        entry->_mesh = this;
        entry->compute();
    }

#pragma omp parallel for
    for (auto &entry: _faces) {
        entry->_mesh = this;
        entry->compute();
    }

#pragma omp parallel for
    for (auto &entry: _nodes) {
        entry->_mesh = this;
        entry->compute();
    }

    std::cout << "NUM NODES = " << _num_nodes << std::endl;

    _node_is_boundary = Eigen::Matrix<float, -1, 1>{_num_nodes};
    _node_is_boundary_reverse = Eigen::Matrix<float, -1, 1>{_num_nodes};

    _volumes = Eigen::Matrix<float, -1, 1>{_num_nodes};
#pragma omp parallel for
    for (const auto &node: _nodes) {
        _volumes(node->_id) = node->_volume;
    }

    _normal_x = MatrixX6dRB{_num_nodes, 6};
    _normal_y = MatrixX6dRB{_num_nodes, 6};
    _normal_z = MatrixX6dRB{_num_nodes, 6};

    _normal_alt_x = MatrixX6dRB{_num_nodes, 6};
    _normal_alt_y = MatrixX6dRB{_num_nodes, 6};
    _normal_alt_z = MatrixX6dRB{_num_nodes, 6};

    _ids = MatrixX6Idx{_num_nodes, 6};
    _ids_bound_free = MatrixX6SignIdx{_num_nodes, 6};

    _alpha_d = MatrixX6dRB{_num_nodes, 6};
    _len_node_center_to_face = MatrixX6dRB{_num_nodes, 6};
    _n_min_alpha_d = std::vector<MatrixX6dRB>{
            MatrixX6dRB{_num_nodes, 6},
            MatrixX6dRB{_num_nodes, 6},
            MatrixX6dRB{_num_nodes, 6}
    };

    _face_areas = MatrixX6dRB{_num_nodes, 6};

#pragma omp parallel for
    for (size_t i = 0; i < _num_nodes; ++i) {
        auto &node = _nodes.at(i);

        auto norm_v_x{node->_normals.block<6, 1>(0, 0)};
        auto norm_v_y{node->_normals.block<6, 1>(0, 1)};
        auto norm_v_z{node->_normals.block<6, 1>(0, 2)};

        auto norm_alt_v_x{node->_normals_alt.block<6, 1>(0, 0)};
        auto norm_alt_v_y{node->_normals_alt.block<6, 1>(0, 1)};
        auto norm_alt_v_z{node->_normals_alt.block<6, 1>(0, 2)};

        _normal_x.block<1, 6>(node->_id, 0) = norm_v_x;
        _normal_y.block<1, 6>(node->_id, 0) = norm_v_y;
        _normal_z.block<1, 6>(node->_id, 0) = norm_v_z;

        _normal_alt_x.block<1, 6>(node->_id, 0) = norm_alt_v_x;
        _normal_alt_y.block<1, 6>(node->_id, 0) = norm_alt_v_y;
        _normal_alt_z.block<1, 6>(node->_id, 0) = norm_alt_v_z;

        if (node->is_boundary()) {
            _node_is_boundary(i) = 1.;
            _node_is_boundary_reverse(i) = 0.;
        } else {
            _node_is_boundary(i) = 0.;
            _node_is_boundary_reverse(i) = 1.;
        }

        for (int j = 0; j < node->_faces_id.size(); ++j) {
            auto face_id = node->_faces_id.at(j);
            auto &face = _faces.at(face_id);
            auto opposite_face_local_id = node->opposite_face_id(j);
            auto opposite_face_id = node->_faces_id.at(opposite_face_local_id);
            auto &opposite_face = _faces.at(opposite_face_id);

            _face_areas(i, j) = face->_area;

            auto &n1 = node, n2 = node;
            if (face->_nodes_id.size() > 1) {
                auto &n1_ = _nodes.at(face->_nodes_id.at(0));
                auto &n2_ = _nodes.at(face->_nodes_id.at(1));
                if (n1_->_id == i) {
                    n2 = n2_;
                } else {
                    n2 = n1_;
                }

                _ids(i, j) = n2->_id;
                _ids_bound_free(i, j) = n2->_id;

                Eigen::Matrix<float, -1, 1> face_normal = node->_normals.block<1, 3>(j, 0);
                Eigen::Matrix<float, -1, 1> node_center_connect = (n2->center_coords()) - (n1->center_coords());
                float cos_phi =
                        (face_normal.dot(node_center_connect)) / (face_normal.norm() * node_center_connect.norm());
                float alpha = 1 / cos_phi;

                _alpha_d(i, j) = std::abs(alpha / node_center_connect.norm());
                _len_node_center_to_face(i, j) = std::abs(((face->_center_coords) - (n1->center_coords())).norm());

                auto n_min_alpha_d = face_normal - (alpha * node_center_connect);
                _n_min_alpha_d.at(0)(i, j) = n_min_alpha_d(0);
                _n_min_alpha_d.at(1)(i, j) = n_min_alpha_d(1);
                _n_min_alpha_d.at(2)(i, j) = n_min_alpha_d(2);

            } else {
                _ids(i, j) = n1->_id;
                _ids_bound_free(i, j) = -1;

//                auto& n1_ = _nodes.at(face->_nodes_id.at(0));
//
//                auto node_x = n1_->_coord_idx.at(0);
//                auto node_y = n1_->_coord_idx.at(1);
//                auto node_z = n1_->_coord_idx.at(2);
//
//                if (j == 0) {
//                    node_z = _z - 1;
//                };
//                if (j == 1) {
//                    node_z = 0;
//                };
//                if (j == 2) {
//                    node_y = _y - 1;
//                };
//                if (j == 3) {
//                    node_y = 0;
//                };
//                if (j == 4) {
//                    node_x = _x - 1;
//                };
//                if (j == 5) {
//                    node_x = 0;
//                };
//
//                auto& opposite_node = _nodes.at(square_node_coord_to_idx(node_x, node_y, node_z));
//                _ids(i, j) = opposite_node->_id;
//                _ids_bound_free(i, j) = opposite_node->_id;

                Eigen::Matrix<float, -1, 1> face_normal = node->_normals.block<1, 3>(j, 0);
                Eigen::Matrix<float, -1, 1> node_center_connect = (face->_center_coords) - (n1->center_coords());
                float cos_phi =
                        (face_normal.dot(node_center_connect)) / (face_normal.norm() * node_center_connect.norm());
                float alpha = 1 / cos_phi;

                _alpha_d(i, j) = std::abs(alpha / (node_center_connect.norm() * 2));
                _len_node_center_to_face(i, j) = std::abs(((face->_center_coords) - (n1->center_coords())).norm());

                Eigen::Matrix<float, -1, 1> n_min_alpha_d = face_normal - (alpha * node_center_connect);
                _n_min_alpha_d.at(0)(i, j) = n_min_alpha_d(0);
                _n_min_alpha_d.at(1)(i, j) = n_min_alpha_d(1);
                _n_min_alpha_d.at(2)(i, j) = n_min_alpha_d(2);
            }

        }
    }

    _normals_all = {&_normal_x, &_normal_y, &_normal_z};
    _normals_alt_all = {&_normal_alt_x, &_normal_alt_y, &_normal_alt_z};

    std::cout << "Done mesh compute" << std::endl;
}

void Mesh3D::init_basic_internals() {
    size_t vrtx_id = 0;
    for (int x_ = 0; x_ < (_x + 1); ++x_) {
        for (int y_ = 0; y_ < (_y + 1); ++y_) {
            for (int z_ = 0; z_ < (_z + 1); ++z_) {
                auto vrtx = std::make_shared<Vertex3D>(x_ * _dx, y_ * _dy, z_ * _dz, vrtx_id);
                _vertexes.push_back(vrtx);
                vrtx_id++;
            }
        }
    }

    for (size_t i = 0; i < ((_x * _y * (_z + 1)) + (_z * _y * (_x + 1)) + (_x * _z * (_y + 1))); ++i) {
        auto face = std::make_shared<Face3D>(0, 0, 0, 0, i);
        _faces.push_back(face);
    }

    for (size_t i = 0; i < (_x * _y * _z); ++i) {
        auto node = std::make_shared<Quadrangle3D>(0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0, 0, 0, 0,
                                                   i,
                                                   0, 0, 0);
        _nodes.push_back(node);
    }

    for (size_t x_ = 0; x_ < _x; ++x_) {
        for (size_t y_ = 0; y_ < _y; ++y_) {
            for (size_t z_ = 0; z_ < _z; ++z_) {
                auto node_id = square_node_coord_to_idx(x_, y_, z_);
                auto vtrx_ids = square_node_coord_to_vertex_idx(x_, y_, z_);

                auto face_ids = square_node_coord_to_face_idx(x_, y_, z_);

                auto node = _nodes.at(node_id);
                node->_faces_id = face_ids;
                node->_vertexes_id = vtrx_ids;
                node->_coord_idx = {x_, y_, z_};

                _faces.at(face_ids.at(0))->_vertexes_id = {vtrx_ids.at(0), vtrx_ids.at(2), vtrx_ids.at(6),
                                                           vtrx_ids.at(4)};
                _faces.at(face_ids.at(1))->_vertexes_id = {vtrx_ids.at(1), vtrx_ids.at(3), vtrx_ids.at(7),
                                                           vtrx_ids.at(5)};

                _faces.at(face_ids.at(2))->_vertexes_id = {vtrx_ids.at(0), vtrx_ids.at(1), vtrx_ids.at(5),
                                                           vtrx_ids.at(4)};
                _faces.at(face_ids.at(3))->_vertexes_id = {vtrx_ids.at(2), vtrx_ids.at(3), vtrx_ids.at(7),
                                                           vtrx_ids.at(6)};

                _faces.at(face_ids.at(4))->_vertexes_id = {vtrx_ids.at(1), vtrx_ids.at(3), vtrx_ids.at(2),
                                                           vtrx_ids.at(0)};
                _faces.at(face_ids.at(5))->_vertexes_id = {vtrx_ids.at(5), vtrx_ids.at(7), vtrx_ids.at(6),
                                                           vtrx_ids.at(4)};

                for (auto face_id: face_ids) {
                    _faces.at(face_id)->_nodes_id.push_back(node_id);
                    _faces.at(face_id)->_set = true;
                }
            }
        }
    }

    std::cout << "Nodes set" << std::endl;
}

size_t Mesh3D::square_node_coord_to_idx(size_t x, size_t y, size_t z) const {
    return _z * _y * x + _z * y + z;
}

std::array<size_t, 6> Mesh3D::square_node_coord_to_face_idx(size_t x, size_t y, size_t z) const {
    return {
            (_z + 1) * _y * x + (_z + 1) * y + z,
            (_z + 1) * _y * x + (_z + 1) * y + (z + 1),

            (_x * _y * (_z + 1)) + (_x * (_y + 1) * z + (_y + 1) * x + y),
            (_x * _y * (_z + 1)) + (_x * (_y + 1) * z + (_y + 1) * x + (y + 1)),

            (_x * _y * (_z + 1)) + (_x * (_y + 1) * _z) + ((_x + 1) * _z * y + (_x + 1) * z + x),
            (_x * _y * (_z + 1)) + (_x * (_y + 1) * _z) + ((_x + 1) * _z * y + (_x + 1) * z + (x + 1)),

    };
}

std::array<size_t, 8> Mesh3D::square_node_coord_to_vertex_idx(size_t x, size_t y, size_t z) const {
    return {
            (x + 0) * (_z + 1) * (_y + 1) + (y + 0) * (_z + 1) + (z + 0),
            (x + 0) * (_z + 1) * (_y + 1) + (y + 0) * (_z + 1) + (z + 1),
            (x + 0) * (_z + 1) * (_y + 1) + (y + 1) * (_z + 1) + (z + 0),
            (x + 0) * (_z + 1) * (_y + 1) + (y + 1) * (_z + 1) + (z + 1),
            (x + 1) * (_z + 1) * (_y + 1) + (y + 0) * (_z + 1) + (z + 0),
            (x + 1) * (_z + 1) * (_y + 1) + (y + 0) * (_z + 1) + (z + 1),
            (x + 1) * (_z + 1) * (_y + 1) + (y + 1) * (_z + 1) + (z + 0),
            (x + 1) * (_z + 1) * (_y + 1) + (y + 1) * (_z + 1) + (z + 1)
    };
}
