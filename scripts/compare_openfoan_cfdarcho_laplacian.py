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


def mesh_variable_to_grid_cfdarcho(var_value, Lx, Ly):
    grid = np.zeros((Lx, Ly), dtype=np.float64)
    for idx, elem in enumerate(var_value):
        x_coord = int(idx % Lx)
        y_coord = int(idx / Lx)
        grid[x_coord, y_coord] = elem
    return grid


def make_heatmap(T, Lx, Ly):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
    ax.pcolormesh(X, Y, T, vmax=100, vmin=-100)


def read_var_cfdarcho(var_path, Lx, Ly):
    var_history = []
    filepathes = []
    for filename in tqdm.tqdm(os.listdir(var_path)):
        f = os.path.join(var_path, filename)
        if os.path.isfile(f):
            filepathes.append(f)

    for i in range(len(filepathes)):
        var = np.fromfile(var_path + "/" + str(i) + ".bin", dtype="float64")
        var_history.append(mesh_variable_to_grid_cfdarcho(var, Lx, Ly))

    return var_history[-1]


def mesh_variable_to_grid_openfoam(var_value, Lx, Ly):
    grid = np.zeros((Lx, Ly), dtype=np.float64)
    for idx, elem in enumerate(var_value):
        x_coord = int(idx / Lx)
        y_coord = int(idx % Lx)
        grid[x_coord, y_coord] = elem
    return grid


def read_var_openfoam(var_path, Lx, Ly):
    var_ = []
    with open(var_path) as filee:
        for line in filee.readlines():
            var_.append(float(line[:-1]))

    return mesh_variable_to_grid_openfoam(np.asarray(var_), Lx, Ly) - 273


def percent_error(predictions, targets):
    return np.mean(np.abs(np.abs(predictions)-np.abs(targets))/np.abs(targets)) * 100


if __name__ == '__main__':
    base_dir_cfdarco = "../dumps/run_latest/"
    base_dir_openfoam = "../dumps/T"

    with open(base_dir_cfdarco + "/mesh.json") as filee:
        mesh_json = json.load(filee)
    mesh = []

    for node in mesh_json["nodes"]:
        node_repr = []
        for v_id in node["vertexes"]:
            node_repr.append(mesh_json["vertexes"][v_id])
        mesh.append(node_repr)

    cfdarco_T = read_var_cfdarcho(base_dir_cfdarco + "/T/", mesh_json["x"], mesh_json["y"])
    openfoam_T = read_var_openfoam(base_dir_openfoam, mesh_json["x"], mesh_json["y"])

    make_heatmap(cfdarco_T, mesh_json["x"], mesh_json["y"])
    make_heatmap(openfoam_T, mesh_json["x"], mesh_json["y"])

    print("error = ", percent_error(cfdarco_T, openfoam_T), "%")

    plt.show()
