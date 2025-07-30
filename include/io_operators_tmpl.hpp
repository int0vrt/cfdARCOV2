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
#ifndef CFDARCO_IO_OPERATORS_HPP
#define CFDARCO_IO_OPERATORS_HPP

#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

#include <vtkActor.h>
#include <vtkImageData.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

template<typename MeshClass>
void write_single_vti_file(const MeshClass *mesh, Eigen::Matrix<float, -1, 1> &grid_hist, const fs::path& store_path) {
    vtkNew<vtkImageData> imageData;
    imageData->SetDimensions(mesh->_x, mesh->_y, mesh->_z);
    imageData->AllocateScalars(VTK_FLOAT, 0);
    int *dims = imageData->GetDimensions();

#pragma omp parallel for collapse(3)
    for (int z = 0; z < dims[2]; z++) {
        for (int y = 0; y < dims[1]; y++) {
            for (int x = 0; x < dims[0]; x++) {
                float *pixel = static_cast<float *>(imageData->GetScalarPointer(x, y, z));
                float mesh_val = grid_hist(mesh->square_node_coord_to_idx(x, y, z));
                pixel[0] = mesh_val;
            }
        }
    }

    vtkNew<vtkXMLImageDataWriter> writer;
    writer->SetFileName(store_path.c_str());
    writer->SetInputData(imageData);
    writer->Write();
}

inline void start_base_store(const std::vector<Variable *> &vars_to_store) {
    if (CFDArcoGlobalInit::get_rank() == 0) {
        fs::create_directories(CFDArcoGlobalInit::store_dir);

        for (auto var: vars_to_store) {
            auto store_var_dir = CFDArcoGlobalInit::store_dir / var->name;
            fs::create_directories(store_var_dir);
        }
    }
}

inline void end_base_store() {
    if (CFDArcoGlobalInit::get_rank() == 0) {
        auto store_dir_latest = fs::absolute(CFDArcoGlobalInit::store_dir / ".." / "run_latest");
        fs::remove_all(store_dir_latest);
        fs::create_symlink(fs::absolute(CFDArcoGlobalInit::store_dir), store_dir_latest);
    }
}

template<typename Variable, typename MeshClass>
void store_history(const std::vector<Variable *> &vars_to_store, const MeshClass *mesh) {
    start_base_store(vars_to_store);

    for (auto var: vars_to_store) {

        auto store_var_dir = CFDArcoGlobalInit::store_dir / var->name;

        for (int i = 0; i < var->history.size() - 1; ++i) {
            Eigen::Matrix<float, -1, 1> &grid_hist = var->history[i];
            if ((i % CFDArcoGlobalInit::store_n) != 0) {
                continue;
            }
            fs::path store_path = fs::path{store_var_dir / ("res_" + std::to_string(i) + ".vti")};
            write_single_vti_file(mesh, grid_hist, store_path);
        }
    }
    end_base_store();

}

void thread_function(int aaa) {
    StoreInstance store_val;
    while (1) {
        bool success_pop = CFDArcoGlobalInit::store_queue.try_pop(store_val);
        if (success_pop) {
            if (store_val.finalize_) {
                return;
            }

            auto store_var_dir = CFDArcoGlobalInit::store_dir / store_val.name_;
            fs::path store_path = fs::path{store_var_dir / ("res_" + std::to_string(store_val.i_) + ".vti")};
            write_single_vti_file(static_cast<Mesh3D*>(store_val.mesh_ptr_), store_val.data_, store_path);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }
}

inline void init_store_history_stepping(const std::vector<Variable *> &vars_to_store) {
    CFDArcoGlobalInit::store_stepping = true;
    start_base_store(vars_to_store);
    CFDArcoGlobalInit::store_queue = oneapi::tbb::concurrent_queue<StoreInstance>{};

    CFDArcoGlobalInit::store_thread.join();
    CFDArcoGlobalInit::store_thread = std::thread(thread_function, 10);
}

inline void finalize_history_stepping() {
    Eigen::Matrix<float, -1, 1> tmp_m = Eigen::Matrix<float, -1, 1>{1};
    std::string tmp_n = "";
    Mesh3D* tmp_mesh = nullptr;

    CFDArcoGlobalInit::store_queue.push(StoreInstance{tmp_m, tmp_mesh, tmp_n, 0, true});
    CFDArcoGlobalInit::store_thread.join();
    end_base_store();
}


template<typename Variable, typename MeshClass>
void store_history_stepping(Variable * var, MeshClass *mesh, int i) {
//    auto store_var_dir = CFDArcoGlobalInit::store_dir / var->name;
//    fs::path store_path = fs::path{store_var_dir / ("res_" + std::to_string(i) + ".vti")};
//    Eigen::Matrix<float, -1, 1> &grid_hist = var->current;
//    write_single_vti_file(mesh, grid_hist, store_path);

    CFDArcoGlobalInit::store_queue.push(StoreInstance{var->current, mesh, var->name, i, false});
}


//
//template<typename MeshClass>
//void store_mesh(const MeshClass *mesh, const fs::path &store_path) {
//    json mesh_json;
//
//    auto vertexes = json::array();
//    auto faces = json::array();
//    auto nodes = json::array();
//
//    for (auto vrtx: mesh->_vertexes) {
//        vertexes.push_back(json::array({vrtx->x(), vrtx->y(), vrtx->z()}));
//    }
//    for (auto face: mesh->_faces) {
//        auto vertexes_id = json::array({face->_vertexes_id.at(0), face->_vertexes_id.at(1), face->_vertexes_id.at(2),
//                                        face->_vertexes_id.at(3)});
//        auto nodes_id = json::array();
//        for (auto node_id: face->_nodes_id) {
//            nodes_id.push_back(node_id);
//        }
//        faces.push_back({
//                                {"vertexes_id", vertexes_id},
//                                {"nodes_id",    nodes_id},
//                        });
//    }
//    for (auto node: mesh->_nodes) {
//        auto node_faces_arr = json::array();
//        for (auto face_id: node->_faces_id) {
//            node_faces_arr.push_back(face_id);
//        }
//        auto node_vrtx_arr = json::array();
//        for (auto vrtx_id: node->_vertexes_id) {
//            node_vrtx_arr.push_back(vrtx_id);
//        }
//        nodes.push_back({
//                                {"vertexes", node_vrtx_arr},
//                                {"face",     node_faces_arr},
//                        });
//    }
//
//    mesh_json["vertexes"] = vertexes;
//    mesh_json["face"] = faces;
//    mesh_json["nodes"] = nodes;
//
//    mesh_json["x"] = mesh->_x;
//    mesh_json["y"] = mesh->_y;
//    mesh_json["z"] = mesh->_z;
//    mesh_json["lx"] = mesh->_lx;
//    mesh_json["ly"] = mesh->_ly;
//    mesh_json["lz"] = mesh->_lz;
//
//    std::ofstream o(store_path / "mesh.json");
//    o << std::setw(4) << mesh_json << std::endl;
//}


//template<typename Variable, typename MeshClass>
//std::shared_ptr<MeshClass> read_mesh(const fs::path& store_path) {
//    std::ifstream f(store_path);
//    json data = json::parse(f);
//
//    auto mesh = std::make_shared<MeshClass>(data["x"], data["y"], data["lx"], data["ly"]);
//    mesh->_num_nodes = data["nodes"].size();
//    mesh->_num_nodes_tot = data["nodes"].size();
//    for (int i = 0; i < data["vertexes"].size(); ++i) {
//        auto vrtx = std::make_shared<Vertex2D>(data["vertexes"][i][0], data["vertexes"][i][1], i);
//        mesh->_vertexes.push_back(vrtx);
//    }
//    for (int i = 0; i < data["edges"].size(); ++i) {
//        auto edge_json = data["edges"][i];
//        auto edge = std::make_shared<Edge2D>(edge_json["vertexes_id"][0], edge_json["vertexes_id"][1], i);
//        for (auto node_id : edge_json["nodes_id"]) {
//            edge->_nodes_id.push_back(node_id);
//        }
//        mesh->_edges.push_back(edge);
//    }
//    for (int i = 0; i < data["nodes"].size(); ++i) {
//        auto node_edges_arr = data["nodes"][i]["edges"];
//        auto node_vrtx_arr = data["nodes"][i]["vertexes"];
//        auto node = std::make_shared<Quadrangle2D>( node_edges_arr[0], node_edges_arr[1], node_edges_arr[2], node_edges_arr[3],
//                                                    node_vrtx_arr[0], node_vrtx_arr[1], node_vrtx_arr[2], node_vrtx_arr[3], i);
//
//        mesh->_nodes.push_back(node);
//    }
//
//    mesh->compute();
//
//    return mesh;
//}
//
//
//template<typename Variable, typename MeshClass>
//std::pair<std::shared_ptr<MeshClass>, std::vector<Variable>> read_history(const fs::path& store_path) {
//    auto mesh = read_mesh(store_path / "mesh.json");
//    std::vector<Variable> vars;
//
//    for (const auto & entry : fs::directory_iterator(store_path)) {
//        if (entry.is_directory()) {
//            Eigen::Matrix<float, -1, 1> initial = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
//            auto var_name = entry.path().filename().string();
//            BoundaryFN bound = [] (MeshClass* mesh, Eigen::Matrix<float, -1, 1>& arr) { throw std::runtime_error{"Using uninitialized Variable"}; return Eigen::Matrix<float, -1, 1>{}; };
//            auto var = Variable(mesh.get(),
//                                initial,
//                                bound,
//                                var_name);
//
//            auto num_entries = std::distance(fs::directory_iterator(entry), fs::directory_iterator{});
//            for (int i = 0; i < num_entries; ++i) {
//                std::ifstream file( entry.path() / (std::to_string(i) + ".bin"), std::ios::binary );
//                if(!file.is_open())
//                {
//                    throw std::runtime_error{"Unable to open the file"};
//                }
//
//                Eigen::Matrix<float, -1, 1> grid {mesh->_num_nodes};
//                file.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(float));
//                file.close();
//
//                if(!file.good()) {
//                    throw std::runtime_error{"Error occurred at reading time!"};
//                }
//
//                var.history.push_back(grid);
//            }
//
//            vars.push_back(var);
//        }
//    }
//
//    return {mesh, vars};
//}
//
//
//template<typename Variable, typename MeshClass>
//std::pair<std::vector<Variable>, int> init_read_history_stepping(MeshClass* mesh, const fs::path& store_path) {
//    std::vector<Variable> vars;
//    int num_entries;
//
//    for (const auto & entry : fs::directory_iterator(store_path)) {
//        if (entry.is_directory()) {
//            Eigen::Matrix<float, -1, 1> initial = Eigen::Matrix<float, -1, 1>::Zero(mesh->_num_nodes);
//            auto var_name = entry.path().filename().string();
//            BoundaryFN bound = [] (MeshClass* mesh, Eigen::Matrix<float, -1, 1>& arr) { throw std::runtime_error{"Using uninitialized Variable"}; return Eigen::Matrix<float, -1, 1>{}; };
//            auto var = Variable(mesh,
//                                initial,
//                                bound,
//                                var_name);
//
//            num_entries = std::distance(fs::directory_iterator(entry), fs::directory_iterator{});
//            vars.push_back(var);
//        }
//    }
//
//    return {vars, num_entries};
//}
//
//template<typename Variable, typename MeshClass>
//void read_history_stepping(MeshClass* mesh, std::vector<Variable*> vars, int q, const fs::path& store_path) {
//    int i = 0;
//    for (const auto & entry : fs::directory_iterator(store_path)) {
//        if (entry.is_directory()) {
//
//            std::ifstream file( entry.path() / (std::to_string(q) + ".bin"), std::ios::binary );
//            if(!file.is_open())
//            {
//                throw std::runtime_error{"Unable to open the file " + (entry.path() / (std::to_string(q) + ".bin")).string()};
//            }
//
//            Eigen::Matrix<float, -1, 1> grid {mesh->_num_nodes};
//            file.read(reinterpret_cast<char*>(grid.data()), grid.size() * sizeof(float));
//            file.close();
//
//            if(!file.good()) {
//                throw std::runtime_error{"Error occurred at reading time!"};
//            }
//
//            vars.at(i)->current = grid;
//            i++;
//        }
//    }
//}


#endif //CFDARCO_IO_OPERATORS_HPP
