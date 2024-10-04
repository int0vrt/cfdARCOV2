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
import time

import tqdm
import pygame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np


def mesh_variable_to_grid(var_value, Lx, Ly):
    grid = np.zeros((Lx, Ly), dtype=np.float64)
    for idx, elem in enumerate(var_value):
        x_coord = int(idx % Lx)
        y_coord = int(idx / Lx)
        grid[x_coord, y_coord] = elem
    return grid


def make_heatmap(T_history, Lx, Ly):
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # T_history = [T_history[-1]]

    if len(T_history) > 1:
        print("Animating")
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
        def animate(i):
            print(i)
            data = T_history[i * 40]
            ax.cla()
            ax.plot_surface(X, Y, data, vmax=15, vmin=-15)
        anim = animation.FuncAnimation(fig, animate, frames=int(len(T_history) / 40), repeat=False, interval=1)
    else:
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(np.linspace(0, Lx, Lx), np.linspace(0, Ly, Ly))
        print("Last show")
        # ax.pcolormesh(X, Y, T_history[0], vmax=100, vmin=-5)
        ax.pcolormesh(X, Y, T_history[0])
        plt.axis('off')
        plt.savefig('mesh_sim.pdf')
    plt.show()

    # pygame.init()
    # display = pygame.display.set_mode((Lx, Ly))
    # pygame.display.set_caption("Solving the 2d Wave Equation")
    #
    #
    # i = 0
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             return
    #
    #     pixeldata = np.zeros((Lx, Ly, 3), dtype=np.uint8)
    #     pixeldata[:, :, 0] = np.clip(T_history[i] + 128, 0, 255)
    #     pixeldata[:, :, 1] = 0
    #     pixeldata[:, :, 2] = 0
    #
    #     surf = pygame.surfarray.make_surface(pixeldata)
    #     display.blit(pygame.transform.scale(surf, (Lx, Ly)), (0, 0))
    #     pygame.display.update()
    #
    #     i+=1
    #     time.sleep(0.1)


def read_var(var_path, Lx, Ly):
    var_history = []
    filepathes = []
    for filename in tqdm.tqdm(os.listdir(var_path)):
        f = os.path.join(var_path, filename)
        if os.path.isfile(f):
            filepathes.append(f)

    for i in range(len(filepathes)):
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

    T_history = read_var(base_dir + "/h/", mesh_json["x"], mesh_json["y"])
    print(mesh_json.keys())
    make_heatmap(T_history, mesh_json["x"], mesh_json["y"])
