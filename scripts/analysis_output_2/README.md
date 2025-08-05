# List of suplementary materials

This directory contains additional materials containing multiple benchmarks and corresponding visualizations of all experiments performed.

For detailed configurations description, please refer to the paper

## Base performance

Name convention: `./<backend>_analysis.pdf` 
X axis - number of points
Y axis - Mpts/s

These graphs correspond to the direct performance (expressed in megapoints per second) for each of the computational backends:

- [Up-to-date performance on A100 with async pipeline](./a100_pipeline_analysis.pdf)
- [Up-to-date performance on A100 without async pipeline](./a100_no_pipeline_analysis.pdf)
- [Up-to-date performance on RTX3060 with async pipeline](./rtx3060_cuda_analysis.pdf)
- [Up-to-date performance with OpenMP](./openmp_cpu_analysis.pdf)
- [Legacy (without KernelBuilder) performane on A100](./a100_no_kernelbuild_analysis.pdf)

## Pairwise equations comparisons

Name convention: `./pairwise_comparison_<equation>.pdf`
X axis - number of points
Y axis - Mpts/s

These graphs show the direct performance (in megapoints per second) for each equation separately for all backends, for a more explicit comparison.

- [Burgers equations performance](./pairwise_comparison_burgers.pdf)
- [Elastic wave equations performance](./pairwise_comparison_elastic.pdf)
- [Euler equations performance](./pairwise_comparison_euler.pdf)
- [Heat equations performance](./pairwise_comparison_heat.pdf)
- [Maxwell equations performance](./pairwise_comparison_maxwell.pdf)
- [Navier-Stocks equations performance](./pairwise_comparison_navier.pdf)
- [Phase-field equations performance](./pairwise_comparison_phase.pdf)
- [Reaction-diffusion equations performance](./pairwise_comparison_reaction.pdf)
- [Acoustic wave equations performance](./pairwise_comparison_wave.pdf)

## Speedup per equation

Name convention: `./speedup_scaling_<equation>.pdf`
X axis - number of points
Y axis - Speedup, times

These graphs correspond to the speedup compared to the performance on the legacy backend (without KernelBuilder) for various equations (according to the previous section).

- [Burgers equations speedup](./speedup_scaling_burgers.pdf)
- [Elastic wave equations speedup](./speedup_scaling_elastic.pdf)
- [Euler equations speedup](./speedup_scaling_euler.pdf)
- [Heat equations speedup](./speedup_scaling_heat.pdf)
- [Maxwell equations speedup](./speedup_scaling_maxwell.pdf)
- [Navier-Stocks equations speedup](./speedup_scaling_navier.pdf)
- [Phase-field equations speedup](./speedup_scaling_phase.pdf)
- [Reaction-diffusion equations speedup](./speedup_scaling_reaction.pdf)
- [Acoustic wave equations speedup](./speedup_scaling_wave.pdf)

## Speedup per mesh

Name convention: `./permesh_speedup_<equation>.pdf`
X axis - equation
Y axis - Speedup, times

Similarly, these graphs present a comparison of speedups (defined similarly to the previous section) for each mesh size to compare the speedup of each equation.

- [Speedup for 8K points mesh](./permesh_speedup_8000.pdf)
- [Speedup for 27K points mesh](./permesh_speedup_27000.pdf)
- [Speedup for 216K points mesh](./permesh_speedup_216000.pdf)
- [Speedup for 1M points mesh](./permesh_speedup_1000000.pdf)
- [Speedup for 3.3M points mesh](./permesh_speedup_3375000.pdf)
- [Speedup for 8M points mesh](./permesh_speedup_8000000.pdf)
- [Speedup for 15M points mesh](./permesh_speedup_15625000.pdf)
- [Speedup for 27M points mesh](./permesh_speedup_27000000.pdf)
- [Speedup for 42M points mesh](./permesh_speedup_42875000.pdf)
- [Speedup for 64M points mesh](./permesh_speedup_64000000.pdf)
- [Speedup for 91M points mesh](./permesh_speedup_91125000.pdf)


## Speedup per backend

Name convention: `./<backend>_speedup_scaling_vs_a100_no_kernelbuild.pdf`
X axis - number of points
Y axis - Speedup, times

The graphs for each backend show the speedup achieved for each backend.

- [Speedup on A100 with async pipeline](./a100_pipeline_speedup_scaling_vs_a100_no_kernelbuild.pdf)
- [Speedup on A100 without async pipeline](./a100_no_pipeline_speedup_scaling_vs_a100_no_kernelbuild.pdf)
- [Speedup on RTX3060 with async pipeline](./rtx3060_cuda_speedup_scaling_vs_a100_no_kernelbuild.pdf)
- [Speedup with OpenMP](./openmp_cpu_speedup_scaling_vs_a100_no_kernelbuild.pdf)

## Async pipeline

The graph shows the impact of applying an async pipeline optimization compared to the no-async-pipeline version.

[Async pipeline impact](./pipeline_impact_summary.pdf)

