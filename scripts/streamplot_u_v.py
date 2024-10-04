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


def make_streamplot(u_history, v_history, Lx, Ly):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))

    # def animate(i):
    #     print(i)
    #     u_curr_history = u_history[i]
    #     v_curr_history = v_history[i]
    #     ax.clear()
    #     ax.streamplot(X, Y, u_curr_history, v_curr_history)
    #
    # anim = animation.FuncAnimation(fig, animate, frames=len(u_history), repeat=False)

    u_curr_history = u_history[-1]
    v_curr_history = v_history[-1]
    ax.streamplot(X, Y, u_curr_history, v_curr_history)
    plt.show()


def read_var(var_path, Lx, Ly):
    var_history = []
    filepathes = []
    for filename in tqdm.tqdm(os.listdir(var_path)):
        f = os.path.join(var_path, filename)
        if os.path.isfile(f):
            filepathes.append(f)

    for i in range(len(filepathes) - 1, len(filepathes)):
        var = np.fromfile(var_path + "/" + str(i) + ".bin", dtype="float64")
        var_history.append(mesh_variable_to_grid(var, Lx, Ly))

    return var_history


if __name__ == '__main__':
    base_dir = "../dumps/run_latest/"

    with open(base_dir + "/mesh.json") as filee:
        mesh_json = json.load(filee)
    mesh = []

    for node in mesh_json["nodes"]:
        node_repr = []
        for v_id in node["vertexes"]:
            node_repr.append(mesh_json["vertexes"][v_id])
        mesh.append(node_repr)

    u_history = read_var(base_dir + "/u/", mesh_json["x"], mesh_json["y"])
    v_history = read_var(base_dir + "/v/", mesh_json["x"], mesh_json["y"])
    make_streamplot(u_history, v_history, mesh_json["x"], mesh_json["y"])
