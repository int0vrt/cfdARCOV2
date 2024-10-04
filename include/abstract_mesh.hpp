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
#ifndef CFDARCO_ABSTRACT_MESH_HPP
#define CFDARCO_ABSTRACT_MESH_HPP


#include <Eigen/Core>

class AbstractVertex {
public:
    virtual void compute() = 0;

    [[nodiscard]] virtual Eigen::Matrix<float, -1, 1> coordinates() const = 0;
};

class AbstractEdge {
public:
    virtual void compute() = 0;

    [[nodiscard]] virtual bool is_boundary() const = 0;
};

class AbstractFace {
public:
    virtual void compute() = 0;

    [[nodiscard]] virtual bool is_boundary() const = 0;
};

class AbstractCell {
public:
    virtual void compute() = 0;

    [[nodiscard]] virtual Eigen::Matrix<float, -1, 1> center_coords() const = 0;

    [[nodiscard]] virtual bool is_boundary() const = 0;
};

class AbstractMesh {
public:
    virtual void compute() = 0;
};

#endif //CFDARCO_ABSTRACT_MESH_HPP
