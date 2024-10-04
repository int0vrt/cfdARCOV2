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

    time_microseconds_cuda_single_gpus = []
    time_microseconds_cuda_multi_gpus = []

    for q in range(5):
        command_single_gpu = ["mpirun", "-n", "1", bin_file, "--skip_history", "-L", str(mesh_size), "-d", "ln", "-t", "300", "-c"]
        result_single_gpu = subprocess.run(command_single_gpu, capture_output=True, text=True)
        outs_single_gpu = result_single_gpu.stdout

        time_str_single_gpu = outs_single_gpu.split("\n")[-2].split(" ")[-1].split("[")[0]
        time_microseconds_single_gpu = int(time_str_single_gpu)
        time_microseconds_cuda_single_gpus.append(time_microseconds_single_gpu)
        print(f"Done single GPU, mesh = {mesh_size} time = {time_microseconds_single_gpu}")

    for q in range(5):
        command_multi_gpu = ["mpirun", "-n", "2", bin_file, "--skip_history", "-L", str(mesh_size), "-d", "ln", "-t", "300", "-c", "--cuda_ranks", "2"]
        result_multi_gpu = subprocess.run(command_multi_gpu, capture_output=True, text=True)
        outs_multi_gpu = result_multi_gpu.stdout

        time_str_multi_gpu = outs_multi_gpu.split("\n")[-2].split(" ")[-1].split("[")[0]
        time_microseconds_multi_gpu = int(time_str_multi_gpu)
        time_microseconds_cuda_multi_gpus.append(time_microseconds_multi_gpu)
        print(f"Done multi GPU, mesh = {mesh_size} time = {time_microseconds_multi_gpu}")

    print(f"Res(mesh_size={mesh_size}) = cuda single - {min(time_microseconds_cuda_single_gpus)}  cuda multiple - {min(time_microseconds_cuda_multi_gpus)} ")

    return time_microseconds_cuda_single_gpus, time_microseconds_cuda_multi_gpus


def generate_report(mesh_sizes, output_file="report_cuda.csv"):
    time_microseconds_cuda_single_gpus = []
    time_microseconds_cuda_multi_gpus = []
    mesh_sizes_df = []
    for mesh_size in mesh_sizes:
        time_cur_cudas, time_cur_parallels = run_one_run(mesh_size)
        for time_cur_cuda, time_cur_parallel in zip(time_cur_cudas, time_cur_parallels):
            mesh_sizes_df.append(mesh_size)
            time_microseconds_cuda_single_gpus.append(time_cur_cuda)
            time_microseconds_cuda_multi_gpus.append(time_cur_parallel)

    df_dict = {"mesh_sizes": mesh_sizes_df, "time_microseconds_cuda_single_gpu": time_microseconds_cuda_single_gpus,
               "time_microseconds_cuda_multi_gpu": time_microseconds_cuda_multi_gpus}
    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHO bench')
    parser.add_argument('-mf', '--mesh_size_from', required=False, type=int)
    parser.add_argument('-mt', '--mesh_size_to', required=False, type=int)
    parser.add_argument('-ms', '--mesh_size_step', required=False, type=int)
    parser.add_argument('-m', '--meshes', required=False, nargs='+', type=int)
    parser.add_argument('-o', '--out_file', required=False, default="report_cuda_multi_gpu_history.csv")

    args = parser.parse_args()

    if args.meshes is None:
        mesh_sizes = range(args.mesh_size_from, args.mesh_size_to, args.mesh_size_step)
    else:
        mesh_sizes = args.meshes
    generate_report(list(mesh_sizes), args.out_file)

