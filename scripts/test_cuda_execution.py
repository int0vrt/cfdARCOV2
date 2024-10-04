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


def run_one_run(mesh_size: int):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO"

    time_microseconds_cudas = []
    time_microseconds_parallels = []

    for q in range(5):
        command_cuda = [bin_file, "-L", str(mesh_size), "-d", "ln", "-t", "300", "-c"]
        result_cuda = subprocess.run(command_cuda, capture_output=True, text=True)
        outs_cuda = result_cuda.stdout

        time_str_cuda = outs_cuda.split("\n")[-2].split(" ")[-1].split("[")[0]
        time_microseconds_cuda = int(time_str_cuda)
        time_microseconds_cudas.append(time_microseconds_cuda)

    for q in range(5):
        # command_parallel = ["mpirun", "-n", "8", bin_file, "--skip_history", "-L", str(mesh_size), "-d", "ln", "-t", "300"]
        # result_parallel = subprocess.run(command_parallel, capture_output=True, text=True)
        # outs_parallel = result_parallel.stdout

        # time_str_parallel = outs_parallel.split("\n")[-2].split(" ")[-1].split("[")[0]
        # time_microseconds_parallel = int(time_str_parallel)
        # time_microseconds_parallels.append(time_microseconds_parallel)
        time_microseconds_parallels.append(0)

    # print(f"Res(mesh_size={mesh_size}) = cuda - {min(time_microseconds_cudas)}  parallel - {min(time_microseconds_parallels)}")
    print(f"Res(mesh_size={mesh_size}) = cuda - {min(time_microseconds_cudas)}  parallel - ")

    return time_microseconds_cudas, time_microseconds_parallels


def generate_report(mesh_sizes, output_file="report_cuda.csv"):
    times_microseconds_cuda = []
    times_microseconds_parallel = []
    mesh_sizes_df = []
    for mesh_size in mesh_sizes:
        time_cur_cudas, time_cur_parallels = run_one_run(mesh_size)
        for time_cur_cuda, time_cur_parallel in zip(time_cur_cudas, time_cur_parallels):
            mesh_sizes_df.append(mesh_size)
            times_microseconds_cuda.append(time_cur_cuda)
            times_microseconds_parallel.append(time_cur_parallel)
            print(time_cur_cuda, time_cur_parallel)

    df_dict = {"mesh_sizes": mesh_sizes_df, "times_microseconds_cuda": times_microseconds_cuda,
               "times_microseconds_parallel": times_microseconds_parallel}
    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-mf', '--mesh_size_from', required=False, type=int)
    parser.add_argument('-mt', '--mesh_size_to', required=False, type=int)
    parser.add_argument('-ms', '--mesh_size_step', required=False, type=int)
    parser.add_argument('-m', '--meshes', required=False, nargs='+', type=int)
    parser.add_argument('-o', '--out_file', required=False, default="report_cuda_history.csv")

    args = parser.parse_args()

    if args.meshes is None:
        mesh_sizes = range(args.mesh_size_from, args.mesh_size_to, args.mesh_size_step)
    else:
        mesh_sizes = args.meshes
    generate_report(list(mesh_sizes), args.out_file)

