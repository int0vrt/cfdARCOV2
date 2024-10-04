"""
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
"""
import json
import os

import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def mesh_variable_to_grid(var_value, Lx, Ly):
    grid = np.zeros((Lx, Ly), dtype=np.float64)
    for idx, elem in enumerate(var_value):
        x_coord = int(idx / Lx)
        y_coord = int(idx % Lx)
        grid[x_coord, y_coord] = elem
    return grid


def make_heatmap(T_history, Lx, Ly):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
    ax.pcolormesh(X, Y, T_history, vmax=100, vmin=-100)
    plt.show()


def read_var(var_path, Lx, Ly):
    var_ = []
    with open(var_path) as filee:
        for line in filee.readlines():
            var_.append(float(line[:-1]))

    return mesh_variable_to_grid(np.asarray(var_), Lx, Ly) - 273.15


if __name__ == '__main__':
    T_path = "../dumps/T"

    Lx = 100
    Ly = 100

    T_val = read_var(T_path, Lx, Ly)
    make_heatmap(T_val, Lx, Ly)
