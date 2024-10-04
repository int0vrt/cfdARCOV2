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
#include "io_operators.hpp"
#include "cfdarcho_main.hpp"
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

void store_history(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, const fs::path& store_path) {
    if (CFDArcoGlobalInit::get_rank() == 0) {
        fs::create_directories(CFDArcoGlobalInit::store_dir);
    }
    for(auto var : vars_to_store) {

        auto store_var_dir = CFDArcoGlobalInit::store_dir / var->name;
        fs::create_directories(store_var_dir);

        for (int i = 0; i < var->history.size() - 1; ++i) {
            auto grid_hist = CFDArcoGlobalInit::recombine(var->history[i], "to_grid");

            if (CFDArcoGlobalInit::get_rank() == 0) {
                std::fstream file;
                file.open(store_var_dir / (std::to_string(i) + ".bin"), std::ios_base::out | std::ios_base::binary);

                if (!file.is_open()) {
                    std::cerr << "Unable to open the file" << std::endl;
                    return;
                }


                file.write(reinterpret_cast<char *>(grid_hist.data()),
                           grid_hist.rows() * grid_hist.cols() * sizeof(double));
                file.close();
            }
        }
    }

    if (CFDArcoGlobalInit::get_rank() == 0) {
        store_mesh(mesh, CFDArcoGlobalInit::store_dir);
        auto store_dir_latest = fs::absolute(store_path / "run_latest");
        fs::remove_all(store_dir_latest);
        fs::create_symlink(fs::absolute(CFDArcoGlobalInit::store_dir), store_dir_latest);

    }
}


void init_store_history_stepping(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, const fs::path& store_path) {
    CFDArcoGlobalInit::store_stepping = true;
    if (CFDArcoGlobalInit::get_rank() == 0) {
        fs::create_directories(CFDArcoGlobalInit::store_dir);

        for(auto var : vars_to_store) {
            auto store_var_dir = CFDArcoGlobalInit::store_dir / var->name;
            fs::create_directories(store_var_dir);
        }

        store_mesh(mesh, CFDArcoGlobalInit::store_dir);
    }
}

void finalize_history_stepping(const fs::path& store_path) {
    if (CFDArcoGlobalInit::get_rank() == 0) {
        auto store_dir_latest = fs::absolute(store_path / "run_latest");
        fs::remove_all(store_dir_latest);
        fs::create_symlink(fs::absolute(CFDArcoGlobalInit::store_dir), store_dir_latest);
    }
}


void store_history_stepping(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, int i) {
    if (CFDArcoGlobalInit::get_rank() == 0) {
        fs::create_directories(CFDArcoGlobalInit::store_dir);
    }
    for (auto var: vars_to_store) {

        auto store_var_dir = CFDArcoGlobalInit::store_dir / var->name;
        fs::create_directories(store_var_dir);

        auto grid_hist = CFDArcoGlobalInit::recombine(var->current, "to_grid");

        if (CFDArcoGlobalInit::get_rank() == 0) {
            std::fstream file;
            file.open(store_var_dir / (std::to_string(i) + ".bin"), std::ios_base::out | std::ios_base::binary);

            if (!file.is_open()) {
                std::cerr << "Unable to open the file" << std::endl;
                return;
            }


            file.write(reinterpret_cast<char *>(grid_hist.data()),
                       grid_hist.rows() * grid_hist.cols() * sizeof(double));
            file.close();
        }
    }
}


void store_mesh(const Mesh2D* mesh, const fs::path& store_path) {
    json mesh_json;

    auto vertexes = json::array();
    auto edges = json::array();
    auto nodes = json::array();

    for (auto vrtx : mesh->_vertexes) {
        vertexes.push_back(json::array({vrtx->x(), vrtx->y()}));
    }
    for (auto edge : mesh->_edges) {
        auto vertexes_id = json::array({edge->_vertexes_id.at(0), edge->_vertexes_id.at(1)});
        auto nodes_id = json::array();
        for (auto node_id : edge->_nodes_id) {
            nodes_id.push_back(node_id);
        }
        edges.push_back({
            {"vertexes_id", vertexes_id},
            {"nodes_id", nodes_id},
        });
    }
    for (auto node : mesh->_nodes_tot) {
        auto node_edges_arr = json::array();
        for (auto edge_id : node->_edges_id) {
            node_edges_arr.push_back(edge_id);
        }
        auto node_vrtx_arr = json::array();
        for (auto vrtx_id : node->_vertexes_id) {
            node_vrtx_arr.push_back(vrtx_id);
        }
        nodes.push_back({
            {"vertexes", node_vrtx_arr},
            {"edges", node_edges_arr},
        });
    }

    mesh_json["vertexes"] = vertexes;
    mesh_json["edges"] = edges;
    mesh_json["nodes"] = nodes;

    mesh_json["x"] = mesh->_x;
    mesh_json["y"] = mesh->_y;
    mesh_json["lx"] = mesh->_lx;
    mesh_json["ly"] = mesh->_ly;

    std::ofstream o(store_path / "mesh.json");
    o << std::setw(4) << mesh_json << std::endl;
}


std::shared_ptr<Mesh2D> read_mesh(const fs::path& store_path) {
    std::ifstream f(store_path);
    json data = json::parse(f);

    auto mesh = std::make_shared<Mesh2D>(data["x"], data["y"], data["lx"], data["ly"]);
    mesh->_num_nodes = data["nodes"].size();
    mesh->_num_nodes_tot = data["nodes"].size();
    for (int i = 0; i < data["vertexes"].size(); ++i) {
        auto vrtx = std::make_shared<Vertex2D>(data["vertexes"][i][0], data["vertexes"][i][1], i);
        mesh->_vertexes.push_back(vrtx);
    }
    for (int i = 0; i < data["edges"].size(); ++i) {
        auto edge_json = data["edges"][i];
        auto edge = std::make_shared<Edge2D>(edge_json["vertexes_id"][0], edge_json["vertexes_id"][1], i);
        for (auto node_id : edge_json["nodes_id"]) {
            edge->_nodes_id.push_back(node_id);
        }
        mesh->_edges.push_back(edge);
    }
    for (int i = 0; i < data["nodes"].size(); ++i) {
        auto node_edges_arr = data["nodes"][i]["edges"];
        auto node_vrtx_arr = data["nodes"][i]["vertexes"];
        auto node = std::make_shared<Quadrangle2D>( node_edges_arr[0], node_edges_arr[1], node_edges_arr[2], node_edges_arr[3],
                                        node_vrtx_arr[0], node_vrtx_arr[1], node_vrtx_arr[2], node_vrtx_arr[3], i);

        mesh->_nodes.push_back(node);
    }

    mesh->compute();

    return mesh;
}


std::pair<std::shared_ptr<Mesh2D>, std::vector<Variable>> read_history(const fs::path& store_path) {
    auto mesh = read_mesh(store_path / "mesh.json");
    std::vector<Variable> vars;

    for (const auto & entry : fs::directory_iterator(store_path)) {
        if (entry.is_directory()) {
            Eigen::VectorXd initial = Eigen::VectorXd::Zero(mesh->_num_nodes);
            auto var_name = entry.path().filename().string();
            BoundaryFN bound = [] (Mesh2D* mesh, Eigen::VectorXd& arr) { throw std::runtime_error{"Using uninitialized Variable"}; return Eigen::VectorXd{}; };
            auto var = Variable(mesh.get(),
                                initial,
                                bound,
                                var_name);

            auto num_entries = std::distance(fs::directory_iterator(entry), fs::directory_iterator{});
            for (int i = 0; i < num_entries; ++i) {
                std::ifstream file( entry.path() / (std::to_string(i) + ".bin"), std::ios::binary );
                if(!file.is_open())
                {
                    throw std::runtime_error{"Unable to open the file"};
                }

                Eigen::VectorXd grid {mesh->_num_nodes};
                file.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(double));
                file.close();

                if(!file.good()) {
                    throw std::runtime_error{"Error occurred at reading time!"};
                }

                var.history.push_back(grid);
            }

            vars.push_back(var);
        }
    }

    return {mesh, vars};
}


std::pair<std::vector<Variable>, int> init_read_history_stepping(Mesh2D* mesh, const fs::path& store_path) {
    std::vector<Variable> vars;
    int num_entries;

    for (const auto & entry : fs::directory_iterator(store_path)) {
        if (entry.is_directory()) {
            Eigen::VectorXd initial = Eigen::VectorXd::Zero(mesh->_num_nodes);
            auto var_name = entry.path().filename().string();
            BoundaryFN bound = [] (Mesh2D* mesh, Eigen::VectorXd& arr) { throw std::runtime_error{"Using uninitialized Variable"}; return Eigen::VectorXd{}; };
            auto var = Variable(mesh,
                                initial,
                                bound,
                                var_name);

            num_entries = std::distance(fs::directory_iterator(entry), fs::directory_iterator{});
            vars.push_back(var);
        }
    }

    return {vars, num_entries};
}

void read_history_stepping(Mesh2D* mesh, std::vector<Variable*> vars, int q, const fs::path& store_path) {
    int i = 0;
    for (const auto & entry : fs::directory_iterator(store_path)) {
        if (entry.is_directory()) {

            std::ifstream file( entry.path() / (std::to_string(q) + ".bin"), std::ios::binary );
            if(!file.is_open())
            {
                throw std::runtime_error{"Unable to open the file " + (entry.path() / (std::to_string(q) + ".bin")).string()};
            }

            Eigen::VectorXd grid {mesh->_num_nodes};
            file.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(double));
            file.close();

            if(!file.good()) {
                throw std::runtime_error{"Error occurred at reading time!"};
            }

            vars.at(i)->current = grid;
            i++;
        }
    }
}