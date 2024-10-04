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
#include "mesh2d.hpp"

Eigen::VectorXd Vertex2D::coordinates() const {
    return _coords;
}

double Vertex2D::x() const {
    return _coords(0);
}

double Vertex2D::y() const {
    return _coords(1);
}

void Edge2D::compute() {
    auto p1_coords = _mesh->_vertexes.at(_vertexes_id.at(0))->_coords;
    auto p2_coords = _mesh->_vertexes.at(_vertexes_id.at(1))->_coords;
    _center_coords = (p1_coords + p2_coords) / 2;
    _normal = Eigen::Vector2d {};
    _normal << -(p2_coords(1) - p1_coords(1)), (p2_coords(0) - p1_coords(0));
    _normal = _normal / _normal.norm();
    _area = (p2_coords - p1_coords).norm();
}

bool Edge2D::is_boundary() const {
//    return _nodes_id.at(0) == _nodes_id.at(1);
    return _nodes_id.size() == 1 || _nodes_id.at(0) == _nodes_id.at(1);
}

void Quadrangle2D::compute() {
    auto p1_coords = _mesh->_vertexes.at(_vertexes_id.at(0))->_coords;
    auto p2_coords = _mesh->_vertexes.at(_vertexes_id.at(1))->_coords;
    auto p3_coords = _mesh->_vertexes.at(_vertexes_id.at(2))->_coords;
    auto p4_coords = _mesh->_vertexes.at(_vertexes_id.at(3))->_coords;
    _center_coords = (p1_coords + p2_coords + p3_coords + p4_coords) / 4;
    for (int i=0; i < 4; ++i){
        auto edge = _mesh->_edges.at(_edges_id.at(i));
        auto direction_vector = edge->_center_coords - _center_coords;
        _vectors_in_edges_directions.block<1, 2>(i, 0) = direction_vector;
        _vectors_in_edges_directions_by_id[edge->_id] = direction_vector;
    }

    _volume = 0.5 * (
            p1_coords(0) * p2_coords(1) + p2_coords(0) * p3_coords(1) + p3_coords(0) * p4_coords(1) + p4_coords(0) * p1_coords(1)
            -
            (p1_coords(1) * p2_coords(0) + p2_coords(1) * p3_coords(0) + p3_coords(1) * p4_coords(0) + p4_coords(1) * p1_coords(0) )
    );

    for (size_t i = 0; i < _vertexes_id.size() - 1; ++i) {
        auto v0_id = _vertexes_id.at(i);
        auto v1_id = _vertexes_id.at(i+1);

        auto v0 = _mesh->_vertexes.at(v0_id);
        auto v1 = _mesh->_vertexes.at(v1_id);

        auto dx = v1->x() - v0->x();
        auto dy = v1->y() - v0->y();
        _normals.block<1, 2>(static_cast<int>(i), 0) = Eigen::Vector2d {dy, -dx};
    }

    auto v0_id = _vertexes_id.at(_vertexes_id.size() - 1);
    auto v1_id = _vertexes_id.at(0);

    auto v0 = _mesh->_vertexes.at(v0_id);
    auto v1 = _mesh->_vertexes.at(v1_id);

    auto dx = v1->x() - v0->x();
    auto dy = v1->y() - v0->y();
    _normals.block<1, 2>(static_cast<int>(_vertexes_id.size()) - 1, 0) = Eigen::Vector2d {dy, -dx};
}

Eigen::VectorXd Quadrangle2D::center_coords() const {
    return _center_coords;
}

bool Quadrangle2D::is_boundary() const {
    return std::any_of(_edges_id.begin(), _edges_id.end(),
                       [&](size_t edge_id) { return _mesh->_edges.at(edge_id)->is_boundary(); });
}

double Quadrangle2D::x() const {
    return _center_coords(0);
}

double Quadrangle2D::y() const {
    return _center_coords(1);
}


void Mesh2D::compute() {
    for (auto& entry : _vertexes) {
        entry->_mesh = this;
        entry->compute();
    }
    for (auto& entry : _edges) {
        entry->_mesh = this;
        entry->compute();
    }
    for (auto& entry : _nodes) {
        entry->_mesh = this;
        entry->compute();
    }

    _volumes_tot = Eigen::VectorXd {_num_nodes_tot};
    for (const auto& node : _nodes) {
        _volumes_tot(node->_id) = node->_volume;
    }

    _n2_ids_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _normal_x_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _normal_y_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_direction_x_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_direction_y_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_neigh_direction_x_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _vec_in_edge_neigh_direction_y_tot = Eigen::MatrixX4d {_num_nodes_tot, 4};
    _node_is_boundary_tot = Eigen::MatrixXd {_num_nodes_tot, 1 };
    _node_is_boundary_reverce_tot = Eigen::MatrixXd {_num_nodes_tot, 1};
    for (size_t i = 0; i < _num_nodes_tot; ++i) {
        auto& node = _nodes.at(i);
        Eigen::Vector4d norm_v_x {node->_normals.block<4, 1>(0, 0)};
        Eigen::Vector4d norm_v_y {node->_normals.block<4, 1>(0, 1)};
        _normal_x_tot.block<1, 4>(node->_id, 0) = norm_v_x;
        _normal_y_tot.block<1, 4>(node->_id, 0) = norm_v_y;
        if (node->is_boundary()) {
            _node_is_boundary_tot(i, 0) = 1.;
            _node_is_boundary_reverce_tot(i, 0) = 0.;
        } else {
            _node_is_boundary_tot(i, 0) = 0.;
            _node_is_boundary_reverce_tot(i, 0) = 1.;
        }

        for (int j = 0; j < node->_edges_id.size(); ++j) {
            auto edge_id = node->_edges_id.at(j);
            auto& edge = _edges.at(edge_id);

            auto& n1 = node, n2 = node;
            if (edge->_nodes_id.size() > 1) {
                auto& n1_ = _nodes.at(edge->_nodes_id.at(0));
                auto& n2_ = _nodes.at(edge->_nodes_id.at(1));
                if (n1_->_id == i) {
                    n2 = n2_;
                } else {
                    n2 = n1_;
                }
            }

            _vec_in_edge_direction_x_tot(node->_id, j) = n1->_vectors_in_edges_directions(j, 0);
            _vec_in_edge_direction_y_tot(node->_id, j) = n1->_vectors_in_edges_directions(j, 1);
            _vec_in_edge_neigh_direction_x_tot(node->_id, j) = n2->_vectors_in_edges_directions_by_id.at(edge_id)(0, 0);
            _vec_in_edge_neigh_direction_y_tot(node->_id, j) = n2->_vectors_in_edges_directions_by_id.at(edge_id)(0, 1);

            _n2_ids_tot(n1->_id, j) = n2->_id;
        }
    }

    _volumes = _volumes_tot;
    _normal_x = _normal_x_tot;
    _normal_y = _normal_y_tot;
    _vec_in_edge_direction_x = _vec_in_edge_direction_x_tot;
    _vec_in_edge_direction_y = _vec_in_edge_direction_y_tot;
    _vec_in_edge_neigh_direction_x = _vec_in_edge_neigh_direction_x_tot;
    _vec_in_edge_neigh_direction_y = _vec_in_edge_neigh_direction_y_tot;
    _n2_ids = _n2_ids_tot;
    _node_is_boundary = _node_is_boundary_tot;
    _node_is_boundary_reverce = _node_is_boundary_reverce_tot;
}

void Mesh2D::init_basic_internals() {
    for (int x_ = 0; x_ < _x; ++x_) {
        for (int y_ = 0; y_ < _y; ++y_) {
            auto i = coord_fo_idx(x_, y_);

            if (i == 0) {
                auto v1 = std::make_shared<Vertex2D>(0, 0, 0);
                auto v2 = std::make_shared<Vertex2D>(_dx, 0, 1);
                auto v3 = std::make_shared<Vertex2D>(_dx, _dy, 2);
                auto v4 = std::make_shared<Vertex2D>(0, _dy, 3);

                auto e1 = std::make_shared<Edge2D>(v1->_id, v2->_id, 0);
                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, 1);
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, 2);
                auto e4 = std::make_shared<Edge2D>(v4->_id, v1->_id, 3);

                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, 0);

                e1->_nodes_id.push_back(n->_id);
                e2->_nodes_id.push_back(n->_id);
                e3->_nodes_id.push_back(n->_id);
                e4->_nodes_id.push_back(n->_id);

                _vertexes.push_back(v1);
                _vertexes.push_back(v2);
                _vertexes.push_back(v3);
                _vertexes.push_back(v4);
                _nodes.push_back(n);
                _edges.push_back(e1);
                _edges.push_back(e2);
                _edges.push_back(e3);
                _edges.push_back(e4);

            } else if (y_ == 0) {
                auto v2 = std::make_shared<Vertex2D>((x_ + 1) * _dx, 0, _vertexes.size());
                auto v3 = std::make_shared<Vertex2D>((x_ + 1) * _dx, _dy, _vertexes.size() + 1);

                auto node_left_id = coord_fo_idx(x_ - 1, y_);
                auto& v1 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(1));
                auto& v4 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(2));

                auto e1 = std::make_shared<Edge2D>(v1->_id, v2->_id, _edges.size());
                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, _edges.size() + 1);
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, _edges.size() + 2);

                auto& e4 = _edges.at(_nodes.at(node_left_id)->_edges_id.at(1));
                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, _nodes.size());

                if (std::find(e1->_nodes_id.begin(), e1->_nodes_id.end(), n->_id) == e1->_nodes_id.end()) {
                    e1->_nodes_id.push_back(n->_id);
                }
                if (std::find(e2->_nodes_id.begin(), e2->_nodes_id.end(), n->_id) == e2->_nodes_id.end()) {
                    e2->_nodes_id.push_back(n->_id);
                }
                if (std::find(e3->_nodes_id.begin(), e3->_nodes_id.end(), n->_id) == e3->_nodes_id.end()) {
                    e3->_nodes_id.push_back(n->_id);
                }
                if (std::find(e4->_nodes_id.begin(), e4->_nodes_id.end(), n->_id) == e4->_nodes_id.end()) {
                    e4->_nodes_id.push_back(n->_id);
                }

                _vertexes.push_back(v2);
                _vertexes.push_back(v3);
                _nodes.push_back(n);
                _edges.push_back(e1);
                _edges.push_back(e2);
                _edges.push_back(e3);

            } else if (x_ == 0) {
                auto v3 = std::make_shared<Vertex2D>(_dx, (y_ + 1) * _dy, _vertexes.size());
                auto v4 = std::make_shared<Vertex2D>(0, (y_ + 1) * _dy, _vertexes.size() + 1);

                auto node_bottom_id = coord_fo_idx(x_, y_ - 1);
                auto& v1 = _vertexes.at(_nodes.at(node_bottom_id)->_vertexes_id.at(3));
                auto& v2 = _vertexes.at(_nodes.at(node_bottom_id)->_vertexes_id.at(2));

                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, _edges.size());
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, _edges.size() + 1);
                auto e4 = std::make_shared<Edge2D>(v4->_id, v1->_id, _edges.size() + 2);

                auto& e1 = _edges.at(_nodes.at(node_bottom_id)->_edges_id.at(2));
                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, _nodes.size());

                if (std::find(e1->_nodes_id.begin(), e1->_nodes_id.end(), n->_id) == e1->_nodes_id.end()) {
                    e1->_nodes_id.push_back(n->_id);
                }
                if (std::find(e2->_nodes_id.begin(), e2->_nodes_id.end(), n->_id) == e2->_nodes_id.end()) {
                    e2->_nodes_id.push_back(n->_id);
                }
                if (std::find(e3->_nodes_id.begin(), e3->_nodes_id.end(), n->_id) == e3->_nodes_id.end()) {
                    e3->_nodes_id.push_back(n->_id);
                }
                if (std::find(e4->_nodes_id.begin(), e4->_nodes_id.end(), n->_id) == e4->_nodes_id.end()) {
                    e4->_nodes_id.push_back(n->_id);
                }

                _vertexes.push_back(v3);
                _vertexes.push_back(v4);
                _nodes.push_back(n);
                _edges.push_back(e2);
                _edges.push_back(e3);
                _edges.push_back(e4);

            } else {
                auto v3 = std::make_shared<Vertex2D>((x_ + 1)*_dx, (y_ + 1) * _dy, _vertexes.size());

                auto node_left_id = coord_fo_idx(x_ - 1, y_ );
                auto node_bottom_id = coord_fo_idx(x_, y_ - 1);
                auto& v1 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(1));
                auto& v2 = _vertexes.at(_nodes.at(node_bottom_id)->_vertexes_id.at(2));
                auto& v4 = _vertexes.at(_nodes.at(node_left_id)->_vertexes_id.at(2));

                auto e2 = std::make_shared<Edge2D>(v2->_id, v3->_id, _edges.size());
                auto e3 = std::make_shared<Edge2D>(v3->_id, v4->_id, _edges.size() + 1);

                auto& e1 = _edges.at(_nodes.at(node_bottom_id)->_edges_id.at(2));
                auto& e4 = _edges.at(_nodes.at(node_left_id)->_edges_id.at(1));
                auto n = std::make_shared<Quadrangle2D>(e1->_id, e2->_id, e3->_id, e4->_id, v1->_id, v2->_id, v3->_id, v4->_id, _nodes.size());

                if (std::find(e1->_nodes_id.begin(), e1->_nodes_id.end(), n->_id) == e1->_nodes_id.end()) {
                    e1->_nodes_id.push_back(n->_id);
                }
                if (std::find(e2->_nodes_id.begin(), e2->_nodes_id.end(), n->_id) == e2->_nodes_id.end()) {
                    e2->_nodes_id.push_back(n->_id);
                }
                if (std::find(e3->_nodes_id.begin(), e3->_nodes_id.end(), n->_id) == e3->_nodes_id.end()) {
                    e3->_nodes_id.push_back(n->_id);
                }
                if (std::find(e4->_nodes_id.begin(), e4->_nodes_id.end(), n->_id) == e4->_nodes_id.end()) {
                    e4->_nodes_id.push_back(n->_id);
                }

                _vertexes.push_back(v3);
                _nodes.push_back(n);
                _edges.push_back(e2);
                _edges.push_back(e3);
            }


        }
    }
}

size_t Mesh2D::coord_fo_idx(size_t x, size_t y) const {
    return x * _x + y;
}

std::vector<size_t> Mesh2D::get_ids_of_neightbours(size_t node_id) {
    std::vector<size_t> ret {};

    auto& node = _nodes.at(node_id);

    for (int j = 0; j < node->_edges_id.size(); ++j) {
        auto edge_id = node->_edges_id.at(j);
        auto &edge = _edges.at(edge_id);

        if (edge->_nodes_id.size() > 1) {
            auto &n1_ = _nodes.at(edge->_nodes_id.at(0));
            auto &n2_ = _nodes.at(edge->_nodes_id.at(1));
            if (n1_->_id != node_id) {
                ret.push_back(n1_->_id);
            }
            if (n2_->_id != node_id) {
                ret.push_back(n2_->_id);
            }
        }
    }

    return ret;
}

void Mesh2D::delete_node(int id) {
    auto node = _nodes.at(id);
    _nodes.erase (_nodes.begin() + id);

    for (auto edge_id : node->_edges_id) {
        auto edge = _edges.at(edge_id);
        edge->_nodes_id.erase(std::remove(edge->_nodes_id.begin(), edge->_nodes_id.end(), node->_id), edge->_nodes_id.end());
    }
    for (const auto& node_iter : _nodes) {
        if (node_iter->_id >= node->_id) {
            node_iter->_id = node_iter->_id - 1;
        }

        std::unordered_map<size_t, Eigen::Matrix<double, 1, 2>> new_vectors_in_edges_directions_by_id{};
        for (auto& it : node_iter->_vectors_in_edges_directions_by_id) {
            if (it.first >= node->_id) {
                new_vectors_in_edges_directions_by_id[it.first - 1] = it.second;
            } else {
                new_vectors_in_edges_directions_by_id[it.first] = it.second;
            }
        }
        node_iter->_vectors_in_edges_directions_by_id = new_vectors_in_edges_directions_by_id;
    }
    for (const auto& edge_iter : _edges) {
        std::vector<size_t> new_nodes_id {};
        for (auto& it : edge_iter->_nodes_id) {
            if (it >= node->_id) {
                new_nodes_id.push_back(it - 1);
            } else {
                new_nodes_id.push_back(it);
            }
        }
        edge_iter->_nodes_id = new_nodes_id;
    }

    _num_nodes_tot--;
    _num_nodes--;
}

void Mesh2D::make_strange_internals() {
    int st_x = 0.4 * static_cast<double>(_x);
    int en_x = 0.6 * static_cast<double>(_x);
    int st_y = 0.4 * static_cast<double>(_y);
    int en_y = 0.6 * static_cast<double>(_y);

    std::vector<size_t> to_remove{};
    for (int x_ = 0; x_ < _x; ++x_) {
        for (int y_ = 0; y_ < _y; ++y_) {
            if (x_ > st_x && x_ < en_x && y_ > st_y && y_ < en_y) {
                auto i = coord_fo_idx(x_, y_);
                to_remove.push_back(i);
            }
        }
    }

//    st_x = 0.2 * static_cast<double>(_x);
//    en_x = 0.4 * static_cast<double>(_x);
//    st_y = 0.6 * static_cast<double>(_y);
//    en_y = 0.8 * static_cast<double>(_y);
//
//    for (int x_ = 0; x_ < _x; ++x_) {
//        for (int y_ = 0; y_ < _y; ++y_) {
//            if (x_ > st_x && x_ < en_x && y_ > st_y && y_ < en_y) {
//                auto i = coord_fo_idx(x_, y_);
//                to_remove.push_back(i);
//            }
//        }
//    }
//
//    st_x = 0.6 * static_cast<double>(_x);
//    en_x = 0.8 * static_cast<double>(_x);
//    st_y = 0.6 * static_cast<double>(_y);
//    en_y = 0.8 * static_cast<double>(_y);
//
//    for (int x_ = 0; x_ < _x; ++x_) {
//        for (int y_ = 0; y_ < _y; ++y_) {
//            if (x_ > st_x && x_ < en_x && y_ > st_y && y_ < en_y) {
//                auto i = coord_fo_idx(x_, y_);
//                to_remove.push_back(i);
//            }
//        }
//    }
//
//    st_x = 0.6 * static_cast<double>(_x);
//    en_x = 0.8 * static_cast<double>(_x);
//    st_y = 0.2 * static_cast<double>(_y);
//    en_y = 0.4 * static_cast<double>(_y);
//
//    for (int x_ = 0; x_ < _x; ++x_) {
//        for (int y_ = 0; y_ < _y; ++y_) {
//            if (x_ > st_x && x_ < en_x && y_ > st_y && y_ < en_y) {
//                auto i = coord_fo_idx(x_, y_);
//                to_remove.push_back(i);
//            }
//        }
//    }


//  make circle
//    std::vector<size_t> to_remove{};
//    for (int x_ = 0; x_ < _x; ++x_) {
//        for (int y_ = 0; y_ < _y; ++y_) {
//            auto i = coord_fo_idx(x_, y_);
//            auto x_rl = static_cast<double>(x_) / static_cast<double>(_x);
//            auto y_rl = static_cast<double>(y_) / static_cast<double>(_y);
//
//            auto origin_dist = x_rl * x_rl + y_rl * y_rl;
//            if (origin_dist <= 0.6) {
//                to_remove.push_back(i);
//            }
//        }
//    }


    std::sort(to_remove.begin(), to_remove.end(), std::greater<>());

    indicators::ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PostfixText{"0.0 %"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ForegroundColor{indicators::Color::blue},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };

    int i = 0;
    int tot_remove = to_remove.size();
    for (auto node_id : to_remove) {
        delete_node(node_id);

        double progress = (static_cast<double>(i) / static_cast<double>(tot_remove)) * 100;
        bar.set_progress(static_cast<size_t>(progress));
        bar.set_option(indicators::option::PostfixText{std::to_string(progress) + " %"});
        i++;
    }
}
