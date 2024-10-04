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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess
import json


def plot_mesh(mesh_nodes, values, out_name):
    template = """
    set terminal pdf
    set output "{out_name}.pdf"
    
    set palette maxcolors 1024
    set style fill transparent solid 0.9 noborder
    set xrange [0:1]
    set yrange [0:1]
    set cbrange [0:1024]
    set size ratio 1
    unset key
    
    {}
    
    plot 0
    """
    template = template.replace("{out_name}", out_name)

    cmap_name = "autumn_r"
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)


    def get_rect_value(x1, x2, x3, x4, y1, y2, y3, y4, value):
        rgb = scalarMap.to_rgba(value)
        r = "{:02x}".format(int(rgb[0] * 255))
        g = "{:02x}".format(int(rgb[1] * 255))
        b = "{:02x}".format(int(rgb[2] * 255))
        rect_template = f'set object polygon from {x1},{y1} to {x2},{y2} to {x3},{y3} to {x4},{y4} to {x1},{y1} fc rgb "#{r}{g}{b}" '
        return rect_template

    all_polys = []

    i = 0
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) in mesh_nodes:
        all_polys.append(get_rect_value(x1, x2, x3, x4, y1, y2, y3, y4, values[i]))
        i += 1

    template = template.replace("{}", "\n".join(all_polys))
    with open("mesh.gnuplot", "w") as filee:
        filee.write(template)

    subprocess.run(["gnuplot", "mesh.gnuplot"])


if __name__ == '__main__':
    with open("../dumps/run_latest/normal_mesh.json") as filee:
        mesh_json = json.load(filee)

    mesh = []
    values = []

    for node in mesh_json["nodes"]:
        node_repr = []
        for v_id in node["vertexes"]:
            node_repr.append(mesh_json["vertexes"][v_id])
        mesh.append(node_repr)
        values.append(.5)

    plot_mesh(mesh, values, "normal_mesh")
