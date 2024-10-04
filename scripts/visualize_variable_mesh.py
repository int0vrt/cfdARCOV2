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
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import subprocess
import json
import tqdm


def plot_mesh(mesh_nodes, values_in_history):
    # set terminal gif animate size 400,400
    # set output "cfd.gif"

    template = """
    set terminal pdf
    set output "mesh_sim.pdf"
    
    set palette maxcolors 1024
    set style fill transparent solid 0.9 noborder
    set xrange [0:1]
    set yrange [0:1]
    set cbrange [0:1024]
    set size ratio 1
    unset key
    
    {}
    """
    sub_template = """
    {}
    plot 0
    """

    cmap_name = "autumn_r"
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)

    def get_rect_value(x1, x2, x3, x4, y1, y2, y3, y4, value):
        rgb = scalarMap.to_rgba(value)
        r = "{:02x}".format(int(rgb[0] * 255))
        g = "{:02x}".format(int(rgb[1] * 255))
        b = "{:02x}".format(int(rgb[2] * 255))
        # rect_template = f'set object polygon from {x1},{y1} to {x2},{y2} to {x3},{y3} to {x4},{y4} to {x1},{y1} fc rgb "#{r}{g}{b}" fillstyle solid 1.0 border lt -1'
        rect_template = f'set object polygon from {x1},{y1} to {x2},{y2} to {x3},{y3} to {x4},{y4} to {x1},{y1} fc rgb "#{r}{g}{b}" '
        return rect_template

    all_history = []
    for curr_values in tqdm.tqdm([values_in_history[-1]]):
        all_polys = []

        i = 0
        for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) in mesh_nodes:
            all_polys.append(get_rect_value(x1, x2, x3, x4, y1, y2, y3, y4, curr_values[i]))
            i += 1

        sub_template_curr = sub_template.replace("{}", "\n".join(all_polys))
        all_history.append(sub_template_curr)

    template = template.replace("{}", "\n".join(all_history))
    with open("poly.gnuplot", "w") as filee:
        filee.write(template)

    subprocess.run(["gnuplot", "poly.gnuplot"])


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

    var_history = []
    filepathes = []
    for filename in tqdm.tqdm(os.listdir(base_dir + "/rho/")):
        f = os.path.join(base_dir + "/rho/", filename)
        if os.path.isfile(f):
            filepathes.append(f)

    for i in range(len(filepathes)):
        var = np.fromfile(base_dir + "/rho/" + str(i) + ".bin", dtype="float64")
        var_history.append(var)

    plot_mesh(mesh, var_history)
