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
#ifndef CFDARCO_IO_OPERATORS_HPP
#define CFDARCO_IO_OPERATORS_HPP

#include "mesh2d.hpp"
#include "fvm.hpp"

#include <filesystem>

namespace fs = std::filesystem;

void store_history(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, const fs::path& store_path = "./dumps");
void init_store_history_stepping(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, const fs::path& store_path = "./dumps");
void store_history_stepping(const std::vector<Variable*>& vars_to_store, const Mesh2D* mesh, int i);
void finalize_history_stepping(const fs::path& store_path = "./dumps");
void store_mesh(const Mesh2D* mesh, const fs::path& store_path = "./dumps");

std::pair<std::vector<Variable>, int> init_read_history_stepping(Mesh2D* mesh, const fs::path& store_path = "./dumps/run_latest/");
void read_history_stepping(Mesh2D* mesh, std::vector<Variable*> vars, int q, const fs::path& store_path = "./dumps/run_latest/");
std::pair<std::shared_ptr<Mesh2D>, std::vector<Variable>> read_history(const fs::path& store_path = "./dumps/run_latest/");
std::shared_ptr<Mesh2D> read_mesh(const fs::path& store_path = "./dumps/run_latest/mesh.json");

#endif //CFDARCO_IO_OPERATORS_HPP
