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
import argparse

import pandas as pd
import os
import subprocess


def run_one_run(num_proc: int, mesh_size: int, available_nodes=("node1", "node2", "node3")):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO"

    nodes_to_run = []
    i = 0
    for q in range(num_proc):
        nodes_to_run.append(available_nodes[i])
        i += 1
        if i >= len(available_nodes):
            i = 0

    command = ["mpirun", "--oversubscribe", "--host", ",".join(nodes_to_run), bin_file, "-L", str(mesh_size), "-v", "--skip_history"]
    result = subprocess.run(command, capture_output=True, text=True)
    outs = result.stdout

    time_str = outs.split("\n")[-2].split(" ")[-1].split("[")[0]
    time_microseconds = int(time_str)

    print(f"Res(num_proc={num_proc}, mesh_size={mesh_size}) = {time_microseconds}")

    return time_microseconds


def generate_report(max_num_proc: int, mesh_sizes, available_nodes=("node1", "node2", "node3"), output_file="report.csv"):
    num_proc = []
    times_microseconds = []
    mesh_sizes_df = []
    for i in range(1, max_num_proc+1):
        for mesh_size in mesh_sizes:
            time_cur = run_one_run(i, mesh_size, available_nodes)
            num_proc.append(i)
            mesh_sizes_df.append(mesh_size)
            times_microseconds.append(time_cur)

    df_dict = {'num_proc': num_proc, 'times_microseconds': times_microseconds, "mesh_sizes": mesh_sizes_df}
    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-p', '--num_proc', required=True, type=int)
    parser.add_argument('-mf', '--mesh_size_from', required=True, type=int)
    parser.add_argument('-mt', '--mesh_size_to', required=True, type=int)
    parser.add_argument('-ms', '--mesh_size_step', required=True, type=int)
    parser.add_argument('-o', '--out_file', required=False, default="report.csv")
    parser.add_argument('-n', '--nodes', nargs='+', required=True, type=str)

    args = parser.parse_args()

    mesh_sizes = range(args.mesh_size_from, args.mesh_size_to, args.mesh_size_step)
    generate_report(args.num_proc, list(mesh_sizes), args.nodes, args.out_file)

