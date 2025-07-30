"""
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2025 cfdARCO team

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
N_TIMESTEPS = 200  # Reduced for benchmarking

BIN_NAME_MODIFIER = ""

# All available experiments with appropriate test configurations
EXPERIMENTS = [
    "euler", "wave", "heat", "maxwell", "burgers", 
    "elastic_wave", "navier_stokes", "reaction_diffusion", "phase_field"
]

CUDA_ENABLE = [True, False]

# Hardware-specific mesh configurations
MESH_SIZES_RTX3060 = {
    "euler": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150)],
    "wave": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200)],
    "heat": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200)],
    "burgers": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200)],
    "elastic_wave": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150)],
    "maxwell": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200)],
    "navier_stokes": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150)],
    "reaction_diffusion": [(20, 20, 3), (30, 30, 3), (60, 60, 3), (100, 100, 3), (150, 150, 3), (200, 200, 3)],  # Quasi-2D
    "phase_field": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200)],
}

MESH_SIZES_A100 = {
    "euler": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400)],
    "wave": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
    "heat": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
    "burgers": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
    "elastic_wave": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400)],
    "maxwell": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
    "navier_stokes": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400)],
    "reaction_diffusion": [(20, 20, 3), (30, 30, 3), (60, 60, 3), (100, 100, 3), (150, 150, 3), (200, 200, 3), (250, 250, 3), (300, 300, 3), (350, 350, 3), (400, 400, 3), (450, 450, 3)],  # Quasi-2D
    "phase_field": [(20, 20, 20), (30, 30, 30), (60, 60, 60), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250), (300, 300, 300), (350, 350, 350), (400, 400, 400), (450, 450, 450)],
}

# Default to RTX 3060 configuration
MESH_SIZES = MESH_SIZES_RTX3060

# Optimized parameters for each equation type
EQUATION_PARAMS = {
    "euler": {
        "dx": 1, "dy": 1, "dz": 1, "dt": 0.01, 
        "extra_args": "--visualize"
    },
    "wave": {
        "dx": 0.1, "dy": 0.1, "dz": 0.1, "dt": 0.01, 
        "extra_args": "--visualize"
    },
    "heat": {
        "dx": 1, "dy": 1, "dz": 1, "dt": 0.001, 
        "extra_args": "--visualize"
    },
    "maxwell": {
        "dx": 0.05, "dy": 0.05, "dz": 0.05, "dt": 0.00001, 
        "extra_args": "--visualize"
    },
    "burgers": {
        "dx": 0.05, "dy": 0.05, "dz": 0.05, "dt": 0.001, 
        "extra_args": "--visualize"
    },
    "elastic_wave": {
        "dx": 1, "dy": 1, "dz": 1, "dt": 0.0001, 
        "extra_args": "--visualize"
    },
    "navier_stokes": {
        "dx": 1, "dy": 1, "dz": 1, "dt": 0.001,
        "extra_args": "--visualize"
    },
    "reaction_diffusion": {
        "dx": 1, "dy": 1, "dz": 100000, "dt": 0.001, 
        "extra_args": "--visualize"
    },
    "phase_field": {
        "dx": 1, "dy": 1, "dz": 1, "dt": 0.000001, 
        "extra_args": "--visualize"
    }
}

OMP_CONF = {
    "cuda": [8],
    "cpu": [8]
}

def get_equation_specific_args(bench_name, mesh_size):
    """Get equation-specific command line arguments"""
    params = EQUATION_PARAMS.get(bench_name, {
        "dx": 1, "dy": 1, "dz": 1, "dt": 0.01, "extra_args": "--visualize"
    })
    
    base_args = f"-Lx {mesh_size[0]} -Ly {mesh_size[1]} -Lz {mesh_size[2]}"
    spacing_args = f"-dx {params['dx']} -dy {params['dy']} -dz {params['dz']}"
    time_args = f"-dt {params['dt']} -t {N_TIMESTEPS}"
    extra_args = params['extra_args']
    
    return f"{base_args} {spacing_args} {time_args} {extra_args}"

def run_single_benchmark(bench_name="euler", cuda_enable=False, mesh_size=(10, 10, 10), 
                        n_timesteps=100, omp_threads=8, n_repeats=5):
    """Run a single benchmark configuration"""
    bin_file = os.path.dirname(os.path.abspath(__file__)) + "/../bin/cfdARCO_" + bench_name + BIN_NAME_MODIFIER

    # Get equation-specific arguments
    argument_line = get_equation_specific_args(bench_name, mesh_size)
    
    # Override timesteps for benchmarking
    argument_line = argument_line.replace(f"-t {N_TIMESTEPS}", f"-t {n_timesteps}")
    
    # Add/remove CUDA flag based on cuda_enable
    if cuda_enable and "-c" not in argument_line:
        argument_line += " -c"
    elif not cuda_enable and "-c" in argument_line:
        argument_line = argument_line.replace(" -c", "")

    command_cuda = [bin_file] + argument_line.split(" ")
    n_points = mesh_size[0] * mesh_size[1] * mesh_size[2] * n_timesteps
    results = []

    crr_env = os.environ.copy()
    if not cuda_enable and omp_threads > 0:
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
    print("Exec command: ", ' '.join(command_cuda))

    mpts_p_sec_best = 0

    for q in range(n_repeats):
        try:
            sub_res = subprocess.run(command_cuda, capture_output=True, text=True, 
                                   env=crr_env)
            outs = sub_res.stdout

            # Parse time from unified output format: "Time elapsed = X [us]"
            time_str_cuda = None
            for line in outs.split("\n"):
                if "Time elapsed" in line and "[us]" in line:
                    time_str_cuda = line.split("=")[1].strip().split("[")[0].strip()
                    break
            
            if time_str_cuda:
                time_seconds = int(time_str_cuda) / 1000000
                mpts_p_sec = (n_points / time_seconds) / 1000000
            else:
                time_seconds = 0
                mpts_p_sec = 0
                print(f"Warning: Could not parse time from output for {bench_name}")

        except subprocess.TimeoutExpired:
            print(f"Timeout for {bench_name} with mesh_size={mesh_size}")
            time_seconds = 0
            mpts_p_sec = 0
        except Exception as e:
            print(f"Error running {bench_name}: {e}")
            time_seconds = 0
            mpts_p_sec = 0

        results.append([time_seconds, mpts_p_sec])
        print(f"Iter {q}: time_seconds = {time_seconds:.6f} mpts_p_sec = {mpts_p_sec:.2f}")
        
        if mpts_p_sec_best < mpts_p_sec:
            mpts_p_sec_best = mpts_p_sec

    print(f"Result({bench_name}, mesh={mesh_size}, cuda={cuda_enable}) = {mpts_p_sec_best:.2f} mpts/sec")
    return test_configuration, results, mpts_p_sec_best

def perform_full_benchmark():
    """Perform comprehensive benchmark across all equations"""
    all_results = []
    
    for bench_name in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {bench_name.upper()}")
        print(f"{'='*60}")
        
        if bench_name not in MESH_SIZES:
            print(f"Warning: No mesh sizes defined for {bench_name}, skipping...")
            continue
            
        for cuda_enable in CUDA_ENABLE:
            execmode = "cuda" if cuda_enable else "cpu"
            print(f"\nMode: {execmode.upper()}")
            
            for mesh_size in MESH_SIZES[bench_name]:
                for omp_threads in OMP_CONF[execmode]:
                    try:
                        single_result = run_single_benchmark(
                            bench_name=bench_name,
                            cuda_enable=cuda_enable,
                            mesh_size=mesh_size,
                            n_timesteps=N_TIMESTEPS,
                            omp_threads=omp_threads,
                            n_repeats=N_REPEATS
                        )
                        all_results.append(single_result)
                    except Exception as e:
                        print(f"Failed benchmark for {bench_name}: {e}")
                        continue
                        
    return all_results

def generate_report(output_file="report_cfdarchov3_comprehensive.csv"):
    """Generate comprehensive benchmark report"""
    print("Starting comprehensive cfdARCO benchmark...")
    print(f"Equations to test: {', '.join(EXPERIMENTS)}")
    print(f"Repeats per configuration: {N_REPEATS}")
    print(f"Timesteps per run: {N_TIMESTEPS}")
    
    # Print hardware configuration
    if MESH_SIZES == MESH_SIZES_A100:
        print("Hardware configuration: A100 (large meshes)")
    else:
        print("Hardware configuration: RTX 3060 (standard meshes)")
    
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
        df_dict["mesh_size"].append(str(expr_conf["mesh_size"]))
        df_dict["num_points"].append(expr_conf["mesh_size"][0] * expr_conf["mesh_size"][1] * expr_conf["mesh_size"][2])
        df_dict["cuda"].append(expr_conf["cuda_enable"])
        df_dict["omp_threads"].append(expr_conf["omp_threads"])
        df_dict["n_timesteps"].append(expr_conf["n_timesteps"])
        df_dict["mpts"].append(mpts)

    df = pd.DataFrame(df_dict)
    df.to_csv(output_file, index=False)
    
    print(f"\nBenchmark completed! Results saved to: {output_file}")
    print(f"Total configurations tested: {len(all_results)}")
    
    # Print summary statistics
    if len(df) > 0:
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 40)
        for experiment in df['experiment'].unique():
            exp_data = df[df['experiment'] == experiment]
            cuda_data = exp_data[exp_data['cuda'] == True]
            if len(cuda_data) > 0:
                max_mpts = cuda_data['mpts'].max()
                print(f"{experiment:20}: {max_mpts:8.2f} MPTS/s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='cfdARCO v2 Benchmark Suite',
        description='Comprehensive benchmarking for all cfdARCO equation solvers'
    )
    parser.add_argument('-o', '--out_file', required=False, 
                       default="report_cfdarchov3_comprehensive.csv",
                       help='Output CSV file for benchmark results')
    parser.add_argument('--equations', nargs='+', choices=EXPERIMENTS,
                       help='Specific equations to benchmark (default: all)')
    parser.add_argument('--cuda-only', action='store_true',
                       help='Only test CUDA configurations')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Only test CPU configurations')
    parser.add_argument('--no-kernel-builder', action='store_true',
                        help='Disable kernelbuilder')
    parser.add_argument('--hardware', choices=['rtx3060', 'a100'], default='rtx3060',
                       help='Hardware configuration: rtx3060 (default) or a100')

    args = parser.parse_args()
    
    # Apply argument filters
    if args.equations:
        EXPERIMENTS = args.equations
        
    if args.cuda_only:
        CUDA_ENABLE = [True]

    if args.cpu_only:
        CUDA_ENABLE = [False]

    if args.no_kernel_builder:
        BIN_NAME_MODIFIER = "_no_kernel_builder"

    # Set hardware-specific mesh sizes
    if args.hardware == 'a100':
        MESH_SIZES = MESH_SIZES_A100
        print("Using A100 configuration (large meshes)")
    else:
        MESH_SIZES = MESH_SIZES_RTX3060
        print("Using RTX 3060 configuration (standard meshes)")
    
    generate_report(args.out_file)

