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


N_REPEATS = 3
N_TIMESTEPS = 200
EXPERIMENTS = ["euler", "heat"]
CUDA_ENABLE = [True]
MESH_SIZES = {
    "euler": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400)],
    "wave": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
    "heat": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
}
OMP_CONF = {
    "cuda": [8],
    # "cpu": [1, 2, 4, 8, 12]
    "cpu": [64]
}


def run_single_benchmark(bench_name = "euler", cuda_enable = False, mesh_size = (10, 10, 10), n_timesteps = 100, omp_threads = 8, n_repeats = 5):
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO_" + bench_name

    argument_line = f"-Lx {mesh_size[0]} -Ly {mesh_size[1]} -Lz {mesh_size[2]} -dx 1 -dy 1 -dz 1 --visualize -t {n_timesteps} -dt 0.005"
    if cuda_enable:
        argument_line += " -c"

    command_cuda = [bin_file] + argument_line.split(" ")
    n_points = mesh_size[0] * mesh_size[1] * mesh_size[2] * n_timesteps
    results = []

    crr_env = os.environ.copy()
    if not cuda_enable:
        crr_env["OMP_NUM_THREADS"] = f"{omp_threads}"
    crr_env["HIP_PLATFORM"] = f"nvidia"

    test_configuration = {
        "bench_name": bench_name,
        "cuda_enable": cuda_enable,
        "mesh_size": mesh_size,
        "n_timesteps": n_timesteps,
        "omp_threads": omp_threads,
        "n_repeats": n_repeats
    }

    print(f"Running experiment {bench_name} cuda_enable = {cuda_enable} mesh_size = {mesh_size} n_timesteps = {n_timesteps} omp_threads = {omp_threads}")
    print("Exec command: ", command_cuda)

    mpts_p_sec_best = 0

    for q in range(n_repeats):
        sub_res = subprocess.run(command_cuda, capture_output=True, text=True, env=crr_env)
        outs = sub_res.stdout

        try:
            time_str_cuda = outs.split("\n")[-2].split(" ")[-1].split("[")[0]
            time_seconds = int(time_str_cuda) / 1000000
            mpts_p_sec = (n_points / time_seconds) / 1000000
        except:
            time_seconds = 0
            mpts_p_sec = 0
            print(outs)
            print(sub_res.stderr)

        results.append([time_seconds, mpts_p_sec])

        print(f"Iter {q}: time_seconds = {time_seconds} mpts_p_sec = {mpts_p_sec}")
        if mpts_p_sec_best < mpts_p_sec:
            mpts_p_sec_best = mpts_p_sec

    print(f"Res(mesh_size={mesh_size}, n_timesteps={n_timesteps}, cuda_enable={cuda_enable}) = {mpts_p_sec_best} mpts/sec")

    return test_configuration, results, mpts_p_sec_best


def perform_full_benchmark():
    all_results = []
    for bench_name in EXPERIMENTS:
        for cuda_enable in CUDA_ENABLE:
            execmode = "cuda" if cuda_enable else "cpu"
            for mesh_size in MESH_SIZES[bench_name]:
                for omp_threads in OMP_CONF[execmode]:
                    single_result = run_single_benchmark(bench_name = bench_name,
                                                         cuda_enable = cuda_enable,
                                                         mesh_size = mesh_size,
                                                         n_timesteps = N_TIMESTEPS,
                                                         omp_threads = omp_threads,
                                                         n_repeats = N_REPEATS)
                    all_results.append(single_result)
    return all_results


def generate_report(output_file="report_cfdarchov2.csv"):
    all_results = perform_full_benchmark()

    df_dict = {
        "experiment": [],
        "mesh_size": [],
        "num_points": [],
        "cuda": [],
        "omp_threads": [],
        "n_timesteps": [],
        "mpts": [],
    }

    for [expr_conf, _, mpts] in all_results:
        df_dict["experiment"].append(expr_conf["bench_name"])
        df_dict["mesh_size"].append(expr_conf["mesh_size"])
        df_dict["num_points"].append(expr_conf["mesh_size"][0] * expr_conf["mesh_size"][1] * expr_conf["mesh_size"][2])
        df_dict["cuda"].append(expr_conf["cuda_enable"])
        df_dict["omp_threads"].append(expr_conf["omp_threads"])
        df_dict["n_timesteps"].append(expr_conf["n_timesteps"])
        df_dict["mpts"].append(mpts)

    df = pd.DataFrame(df_dict)
    df.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cfdARCHOV2 bench')
    parser.add_argument('-o', '--out_file', required=False, default="report_history_a100_cuda_full_no_data_move_even_better_no_kernelbuilder.csv")

    args = parser.parse_args()
    generate_report(args.out_file)

