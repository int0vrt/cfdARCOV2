# cfdARCO

A high-performance Computational Fluid Dynamics (CFD) framework for solving systems of Partial Differential Equations (PDEs) on multi-GPU systems with CUDA/HIP support.


## Key Features

### High-Performance GPU Computing
- **Multi-GPU Support**: Automatic distribution across multiple GPUs
- **CUDA/HIP Backend**: Support for both NVIDIA CUDA and AMD HIP
- **Kernel Building**: Just-In-Time (JIT) compilation of optimized GPU kernels
- **Equation Fusion**: Multiple equations fused into single GPU kernels for reduced memory transfers

### Advanced Numerical Methods
- **Finite Volume Method**: Conservative discretization for complex geometries
- **High-Order Operators**: Support for various spatial discretization schemes
- **Adaptive Time Stepping**: CFL-based and custom time step control
- **Boundary Conditions**: Flexible boundary condition framework

### Modern C++ Framework
- **Expression Templates**: Compile-time optimization of mathematical expressions
- **Template Metaprogramming**: Efficient code generation and optimization
- **Memory Management**: Smart memory handling for CPU/GPU data
- **Parallel Computing**: MPI support for distributed computing

## Kernel Building and Equation Fusion

cfdARCO uses advanced kernel building and equation fusion techniques to maximize GPU performance:

### NEntryVar and Equation Fusion
```cpp
// Multiple equations fused into a single GPU kernel
NEntryVar{
    {d1t(u), '=', -u * d1dx(u) - v * d1dy(u) + nu * lapl(u), true},
    {d1t(v), '=', -u * d1dx(v) - v * d1dy(v) + nu * lapl(v), true}
}.restruct()
```

### Benefits of Kernel Fusion
- **Reduced Memory Transfers**: Multiple equations computed in single kernel launch
- **Better Cache Utilization**: Shared memory access patterns optimized
- **Higher GPU Occupancy**: Better utilization of GPU compute resources
- **Lower Kernel Launch Overhead**: Fewer kernel launches per timestep

## Installation

### Prerequisites

- CMake 3.22 or higher
- C++17 compatible compiler
- CUDA toolkit (for GPU acceleration)
- MPI implementation
- Eigen3 library
- nlohmann/json library

### Build Instructions

```bash
# Clone the repository

# Configure with CMake
cmake -B build-release -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON

# Build
cmake --build build-release

# Install (optional)
cmake --install build-release
```

## Working Example Configurations

### Tested and Verified Configurations

#### **1. Euler Equations** (`cfdARCO_euler`)
```bash
./bin/cfdARCO_euler -Lx 30 -Ly 30 -Lz 30 -dx 1 -dy 1 -dz 1 --visualize -s -t 1000 -dt 0.1 -c
```
- **Physics**: 3D compressible Euler equations
- **Features**: Density variations, shock formation, compressible flow
- **Expected**: Shock waves, rarefaction waves, vortex structures

#### **2. Wave Equation** (`cfdARCO_wave`)
```bash
./bin/cfdARCO_wave -Lx 100 -Ly 100 -Lz 100 -dx 0.1 -dy 0.1 -dz 0.1 --visualize -t 500 -dt 0.1 -c
```
- **Physics**: 3D wave propagation
- **Features**: Two-point sources, spherical wave fronts
- **Expected**: Wave interference patterns, dipole radiation

#### **3. Elastic Wave** (`cfdARCO_elastic_wave`)
```bash
./bin/cfdARCO_elastic_wave -Lx 100 -Ly 100 -Lz 100 -dx 1 -dy 1 -dz 1 --visualize -t 200 -dt 0.0001 -c
```
- **Physics**: Elastic wave propagation in heterogeneous media
- **Features**: P-waves, S-waves, seismic propagation
- **Expected**: Wave reflections, multi-phase wave propagation

#### **4. Heat Equation** (`cfdARCO_heat`)
```bash
./bin/cfdARCO_heat -Lx 100 -Ly 100 -Lz 100 -dx 1 -dy 1 -dz 1 --visualize -t 10000 -dt 0.01 -c
```
- **Physics**: 3D heat diffusion
- **Features**: Hot boundary condition, thermal diffusion
- **Expected**: Heat conduction patterns, temperature gradients

#### **5. Maxwell Equations** (`cfdARCO_maxwell`)
```bash
./bin/cfdARCO_maxwell -Lx 100 -Ly 100 -Lz 100 -dx 0.05 -dy 0.05 -dz 0.05 --visualize -t 200000 -dt 0.0001 -c
```
- **Physics**: Electromagnetic wave propagation (FDTD)
- **Features**: Dipole antenna source, electromagnetic fields
- **Expected**: Spherical wavefronts, electromagnetic radiation

#### **6. Burgers Equation** (`cfdARCO_burgers`)
```bash
./bin/cfdARCO_burgers -Lx 80 -Ly 80 -Lz 80 -dx 0.05 -dy 0.05 -dz 0.05 --visualize -t 10000 -dt 0.001 -c
```
- **Physics**: 3D viscous Burgers equations
- **Features**: Gaussian pulse convection-diffusion
- **Expected**: Wave steepening, shock formation, viscous dissipation

#### **7. Gauss Pulse** (`cfdARCO_gauss_pulse`)
```bash
./bin/cfdARCO_gauss_pulse -Lx 101 -Ly 101 -Lz 1 -dx 0.01 -dy 0.01 -dz 1 -t 500 --visualize -dt 0.01
```
- **Physics**: 2D wave equation with Gaussian initial condition
- **Features**: Quasi-2D wave propagation
- **Expected**: Circular wave propagation, boundary reflections

#### **8. Sod Shock Tube** (`cfdARCO_sod_shock`)
```bash
./bin/cfdARCO_sod_shock -Lx 200 -Ly 1 -Lz 1 -dx 1 -dy 1 -dz 1 -t 5000 --visualize -dt 0.2 -c
```
- **Physics**: 1D shock tube (Riemann problem)
- **Features**: High/low pressure regions, shock formation
- **Expected**: Shock wave, contact discontinuity, rarefaction fan

#### **9. Navier-Stokes** (`cfdARCO_navier_stokes`)
```bash
./bin/cfdARCO_navier_stokes -Lx 200 -Ly 3 -Lz 200 -dx 0.1 -dy 100 -dz 0.1 --visualize -t 100000 -dt 0.001 -c
```
- **Physics**: Lid-driven cavity flow
- **Features**: Moving lid boundary, viscous flow
- **Expected**: Primary vortex, secondary corner vortices

#### **10. 2D Shallow Water** (`cfdARCO_shallow_water_2d`)
```bash
./bin/cfdARCO_shallow_water_2d -Lx 50 -Ly 50 -Lz 3 -dx 2 -dy 2 -dz 1000 --visualize -t 300000 -dt 0.00003 -c
```
- **Physics**: 2D shallow water equations with viscosity
- **Features**: Dam break simulation, viscous effects
- **Expected**: Wave propagation, hydraulic jumps, viscous damping

#### **11. Reaction-Diffusion** (`cfdARCO_reaction_diffusion`)
```bash
./bin/cfdARCO_reaction_diffusion -Lx 80 -Ly 80 -Lz 3 -dx 1 -dy 1 -dz 100000 --visualize -t 100000 -dt 0.0001 -c
```
- **Physics**: Turing pattern formation (Brusselator model)
- **Features**: Self-organizing chemical patterns
- **Expected**: Spots, stripes, labyrinthine structures

#### **12. Phase Field** (`cfdARCO_phase_field`)
```bash
./bin/cfdARCO_phase_field -Lx 50 -Ly 50 -Lz 50 -dx 1 -dy 1 -dz 1 -dt 0.00000003 -t 10000 --visualize -c
```
- **Physics**: Phase field crystal growth
- **Features**: Solid-liquid phase transitions
- **Expected**: Interface propagation, crystal morphology

### Command Line Parameters

- **-Lx, -Ly, -Lz**: Number of grid points (discretization)
- **-dx, -dy, -dz**: Grid spacing [physical units]
- **-dt**: Time step size [physical units]
- **-t**: Number of time steps
- **--visualize**: Enable visualization output
- **-c**: Enable CUDA acceleration
- **-s**: Enable saving intermediate results

## Visualization

Results are automatically saved in VTK format in the `dumps/` directory. Use visualization scripts:

```bash
# General 3D PDE visualization
python scripts/visualize_3d_pde_examples.py --example burgers --timestep 100

# Specialized visualization scripts
python scripts/burgers_visualizer.py       # For Burgers equation
python scripts/basic_visualizer.py         # General purpose visualizer
```

## Architecture

### Core Components

- **Mesh3D**: 3D structured mesh with finite volume discretization
- **Variable**: Field variables with boundary conditions and GPU support  
- **Equation**: Symbolic equation system with JIT compilation
- **Operators**: Spatial and temporal differential operators (`d1dx`, `d1dy`, `d1dz`, `lapl`, etc.)
- **DT**: Adaptive time stepping with CFL conditions

### Boundary Conditions Interface

```cpp
// CPU boundary condition function signature
Eigen::Matrix<float, -1, 1> boundary_function(Mesh3D *mesh, 
                                             Eigen::Matrix<float, -1, 1> &arr, 
                                             const DT *dt_);

// CUDA boundary condition function signature  
CudaDataMatrixD boundary_function_cu(Mesh3D *mesh, 
                                    CudaDataMatrixD &arr, 
                                    const DT *dt_);
```

### Performance Features

- **JIT Kernel Generation**: Runtime compilation for optimal performance
- **Memory Coalescing**: Optimized GPU memory access patterns
- **Kernel Fusion**: Multiple equations computed in single GPU kernel
- **Pipeline Optimization**: Overlapped computation and memory transfer

## Development

### Adding New Examples

1. Create new example file in `examples/`
2. Implement boundary conditions (CPU + CUDA versions)
3. Define initial conditions
4. Set up equation system using `NEntryVar{}.restruct()`
5. Add unified print statements (time elapsed as last output)
6. Test and add working configuration

### Benchmark Testing

```bash
# Run comprehensive benchmarks
python scripts/benchmark_cfdarchov2.py -o results.csv
```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
