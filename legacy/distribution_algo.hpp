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
#ifndef CFDARCO_DISTRIBUTION_ALGO_HPP
#define CFDARCO_DISTRIBUTION_ALGO_HPP

#include "mesh2d.hpp"

std::vector<int> linear_distribution(Mesh2D* mesh, size_t num_proc, std::vector<size_t>& priorities);
std::vector<int> cluster_distribution(Mesh2D* mesh, size_t num_proc, std::vector<size_t>& priorities);

#endif //CFDARCO_DISTRIBUTION_ALGO_HPP
