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


def run_one_run(num_proc: int, mesh_size: int, available_nodes, procs_per_node):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO"

    nodes_to_run = []
    i = 0
    for q in range(num_proc):
        one_node_list = []
        for _ in range(procs_per_node):
            one_node_list.append(available_nodes[i])
        nodes_to_run.append(",".join(one_node_list))
        i += 1
        if i >= len(available_nodes):
            i = 0

    time_microsecondss = []
    for q in range(5):
        command = ["mpirun", "--oversubscribe", "--host", ",".join(nodes_to_run), bin_file, "-L", str(mesh_size), "-v", "--skip_history", "-d", "ln", "-t", "100"]
        result = subprocess.run(command, capture_output=True, text=True)
        outs = result.stdout

        time_str = outs.split("\n")[-2].split(" ")[-1].split("[")[0]
        time_microseconds = int(time_str)
        time_microsecondss.append(time_microseconds)
        print(f"Iter {q} time = {time_microseconds}")

    time_microseconds = min(time_microsecondss)
    print(f"Res(num_proc={num_proc}, mesh_size={mesh_size}): min = {time_microseconds}")

    return time_microsecondss


def generate_report(num_proc_from: int, num_proc_to: int, mesh_sizes, available_nodes, procs_per_node, output_file):
    num_proc = []
    times_microseconds = []
    mesh_sizes_df = []
    for i in range(num_proc_from, num_proc_to+1):
        for mesh_size in mesh_sizes:
            times_cur = run_one_run(i, mesh_size, available_nodes, procs_per_node)
            for time_cur in times_cur:
                num_proc.append(i)
                mesh_sizes_df.append(mesh_size)
                times_microseconds.append(time_cur)

    df_dict = {'num_proc': num_proc, 'times_microseconds': times_microseconds, "mesh_sizes": mesh_sizes_df}
    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-pf', '--num_proc_from', required=True, type=int)
    parser.add_argument('-pt', '--num_proc_to', required=True, type=int)
    parser.add_argument('-mf', '--mesh_size_from', required=False, type=int)
    parser.add_argument('-mt', '--mesh_size_to', required=False, type=int)
    parser.add_argument('-ms', '--mesh_size_step', required=False, type=int)
    parser.add_argument('-m', '--meshes', required=False, nargs='+', type=int)
    parser.add_argument('-o', '--out_file', required=False, default="report_per_nodes_full.csv")
    parser.add_argument('-n', '--nodes', nargs='+', required=True, type=str)
    parser.add_argument('--procs_per_node', required=True, type=int, default=4)

    args = parser.parse_args()

    if args.meshes is None:
        mesh_sizes = range(args.mesh_size_from, args.mesh_size_to, args.mesh_size_step)
    else:
        mesh_sizes = args.meshes
    generate_report(args.num_proc_from, args.num_proc_to, list(mesh_sizes), args.nodes, args.procs_per_node, args.out_file)

