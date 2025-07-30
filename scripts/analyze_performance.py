#!/usr/bin/env python3

#
# cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
# Copyright (C) 2025 cfdARCO team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import argparse
import re

# Set up plotting style for high-quality academic figures with white background
plt.style.use('default')


base_font_size = 24
separate_upper_label_size = 21

# Configure matplotlib for high-quality academic figures

conf = {
    # Font settings for readability in 2-column format
    # 'font.size': 14,           # Base font size
    # 'axes.titlesize': 16,      # Title font size
    # 'axes.labelsize': 14,      # Axis label font size
    # 'xtick.labelsize': 12,     # X-axis tick label size
    # 'ytick.labelsize': 12,     # Y-axis tick label size
    # 'legend.fontsize': 12,     # Legend font size
    # 'figure.titlesize': 18,    # Figure title font size

    # 'font.size': 14,           # Base font size
    # 'axes.titlesize': 18,      # Title font size
    # 'axes.labelsize': 16,      # Axis label font size
    # 'xtick.labelsize': 15,     # X-axis tick label size
    # 'ytick.labelsize': 15,     # Y-axis tick label size
    # 'legend.fontsize': 15,     # Legend font size
    # 'figure.titlesize': 18,    # Figure title font size

    'font.size': base_font_size,           # Base font size
    'axes.titlesize': base_font_size,      # Title font size
    'axes.labelsize': base_font_size,      # Axis label font size
    'xtick.labelsize': base_font_size,     # X-axis tick label size
    'ytick.labelsize': base_font_size,     # Y-axis tick label size
    'legend.fontsize': 22,     # Legend font size
    'figure.titlesize': base_font_size,    # Figure title font size
    
    # Line and marker settings for visibility
    'lines.linewidth': 2.5,    # Thicker lines
    'lines.markersize': 8,     # Larger markers
    'lines.markeredgewidth': 1.5,  # Marker edge width
    
    # Axes and grid settings
    'axes.linewidth': 1.0,     # Thinner axis lines
    # 'axes.spines.top': False,      # Remove top spine
    # 'axes.spines.right': False,    # Remove right spine
    # 'axes.spines.left': False,      # Remove bottom spine
    # 'axes.spines.bottom': False,    # Remove left spine
    'grid.linewidth': 0.8,     # Grid line thickness
    'grid.alpha': 0.3,         # Semi-transparent grid
    'xtick.major.width': 1.0,  # Tick mark thickness
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    
    # Background and figure settings
    'figure.facecolor': 'white',   # White figure background
    'axes.facecolor': 'white',     # White axes background
    'savefig.facecolor': 'white',  # White background when saving
    'figure.dpi': 100,             # Display DPI
    'savefig.dpi': 300,            # Save DPI for publications
    'savefig.bbox': 'tight',       # Tight bounding box
    'savefig.pad_inches': 0.0,     # Padding around figure
    
    # Better rendering
    'pdf.fonttype': 42,        # Embed fonts in PDF (required by many journals)
    'ps.fonttype': 42,         # Embed fonts in PostScript
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    
    # Color and style
    'axes.prop_cycle': plt.cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    
    # Legend settings for academic figures
    'legend.framealpha': 0.8,      # Semi-transparent legend background
    'legend.fancybox': True,       # Rounded corners
    'legend.shadow': False,        # No shadow for cleaner look
    'legend.edgecolor': 'none',    # No border
    'legend.facecolor': 'white',   # White background
    'legend.borderpad': 0.4,       # Padding inside legend
    'legend.columnspacing': 1.0,   # Space between columns
    'legend.handlelength': 1.5,    # Length of legend handles
    'legend.handletextpad': 0.5,   # Pad between handle and text
}

plt.rcParams.update(conf)

# Use a nice color palette without seaborn styling
# sns.set_palette("husl")

def load_benchmark_data(results_dir="results"):
    """Load all available benchmark data from results directory"""
    
    # Update data files to match current structure
    data_files = {
        'a100_no_pipeline': 'report_cfdarchov3_comprehensive_a100_no_pipeline_after_idx32_fixed_bc.csv',
        'a100_pipeline': 'report_cfdarchov3_comprehensive_a100_pipeline_1step_after_idx32.csv',
        'a100_no_kernelbuild': 'report_cfdarchov3_comprehensive_a100_no_kernbuilder_final_fixed_bc.csv',
        'rtx3060_cuda': 'report_cfdarchov3_comprehensive_rtx3060_final.csv',
        'openmp_cpu': 'report_cfdarchov3_comprehensive_ryzen6800h_final.csv',
    }
    
    datasets = {}
    
    # Check if results directory exists
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found. Looking in current directory...")
        results_path = Path(".")
    
    for name, filename in data_files.items():
        filepath = results_path / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                
                # Filter out zero performance values
                if 'mpts' in df.columns:
                    df = df[df['mpts'] > 0]
                elif 'max_mpts' in df.columns:
                    df = df[df['max_mpts'] > 0]

                # Check for performance metric columns
                perf_col = None
                for col in ['mpts', 'max_mpts', 'avg_mpts']:
                    if col in df.columns:
                        perf_col = col
                        break

                if perf_col:
                    df['mpts'] = df[perf_col]  # Standardize column name
                    # Additional filtering after standardization
                    df = df[df['mpts'] > 0]

                datasets[name] = df
                print(f"Loaded {name}: {len(df)} records (after filtering zero values)")
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    return datasets

def analyze_scaling_performance(df, title_suffix="", perf_col='mpts'):
    """Analyze performance scaling with mesh size"""
    
    if perf_col not in df.columns:
        print(f"Warning: {perf_col} column not found in dataset")
        return None
    
    # Equation type styles mapping for distinguishable lines
    equation_styles = {
        'burgers': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 1.5},
        'euler': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 1.5},
        'heat': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 1.5},
        'maxwell': {'color': '#d62728', 'marker': 'v', 'linestyle': ':', 'linewidth': 1.5},
        'navier': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'linewidth': 1.5},
        'phase': {'color': '#8c564b', 'marker': 'p', 'linestyle': '--', 'linewidth': 1.5},
        'reaction': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 1.5},
        'wave': {'color': '#7f7f7f', 'marker': '*', 'linestyle': ':', 'linewidth': 1.5},
        'elastic': {'color': '#bcbd22', 'marker': 'X', 'linestyle': '-', 'linewidth': 1.5},
        'shallow': {'color': '#17becf', 'marker': '+', 'linestyle': '--', 'linewidth': 1.5},
        'sod': {'color': '#ff9896', 'marker': '>', 'linestyle': '-.', 'linewidth': 1.5},
        'default': {'color': '#aec7e8', 'marker': 'o', 'linestyle': '-', 'linewidth': 2}
    }
    
    # fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    fig, axes = plt.subplots(1, 1, figsize=(12.5, 7.5))
    # fig.suptitle(f'Performance Scaling Analysis {title_suffix}', fontsize=16)
    
    
    # 2. Scaling with mesh size
    ax2 = axes
    if 'experiment' in df.columns and 'num_points' in df.columns:
        for experiment in df['experiment'].unique():
            exp_data = df[df['experiment'] == experiment].sort_values('num_points')
            if len(exp_data) > 1:
                # Find matching equation style
                equation_key = 'default'
                exp_lower = experiment.lower()
                for key in equation_styles.keys():
                    if key in exp_lower:
                        equation_key = key
                        break
                
                style = equation_styles[equation_key]
                ax2.plot(exp_data['num_points'], exp_data[perf_col], 
                        color=style['color'],
                        marker=style['marker'],
                        # linestyle=style['linestyle'],
                        linewidth=style['linewidth'],
                        label=experiment.replace('_', ' ').title(),
                        markersize=8,
                        alpha=0.8)
        
        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('MPTS')
        # ax2.set_title('Performance Scaling with Mesh Size')
        ax2.set_xscale('log')
        # ax2.set_yscale('log')

        handles, labels = ax2.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: len(t[0]), reverse=True))
        ax2.legend(handles, labels, loc='upper left', ncols=1)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No scaling data available', 
                ha='center', va='center', transform=ax2.transAxes)
        # ax2.set_title('Performance Scaling')


    # y_ticks = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    # ax2.set_yticks(y_ticks)

    plt.tight_layout(pad=2.0)
    return fig

def analyze_speedup_scaling_performance(datasets, target_backend, baseline_backend='a100_no_kernelbuild', perf_col='mpts'):
    """Analyze speedup scaling with mesh size relative to baseline backend"""
    
    if baseline_backend not in datasets:
        print(f"Warning: Baseline backend '{baseline_backend}' not found in datasets")
        return None
        
    if target_backend not in datasets:
        print(f"Warning: Target backend '{target_backend}' not found in datasets")
        return None
    
    baseline_df = datasets[baseline_backend]
    target_df = datasets[target_backend]
    
    if perf_col not in baseline_df.columns or perf_col not in target_df.columns:
        print(f"Warning: {perf_col} column not found in one of the datasets")
        return None
    
    # Equation type styles mapping for distinguishable lines (same as analyze_scaling_performance)
    equation_styles = {
        'burgers': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 1.5},
        'euler': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'linewidth': 1.5},
        'heat': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'linewidth': 1.5},
        'maxwell': {'color': '#d62728', 'marker': 'v', 'linestyle': ':', 'linewidth': 1.5},
        'navier': {'color': '#9467bd', 'marker': 'D', 'linestyle': '-', 'linewidth': 1.5},
        'phase': {'color': '#8c564b', 'marker': 'p', 'linestyle': '--', 'linewidth': 1.5},
        'reaction': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 1.5},
        'wave': {'color': '#7f7f7f', 'marker': '*', 'linestyle': ':', 'linewidth': 1.5},
        'elastic': {'color': '#bcbd22', 'marker': 'X', 'linestyle': '-', 'linewidth': 1.5},
        'shallow': {'color': '#17becf', 'marker': '+', 'linestyle': '--', 'linewidth': 1.5},
        'sod': {'color': '#ff9896', 'marker': '>', 'linestyle': '-.', 'linewidth': 1.5},
        'default': {'color': '#aec7e8', 'marker': 'o', 'linestyle': '-', 'linewidth': 2}
    }
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 7.5))
    
    # Check if required columns exist
    if 'experiment' not in baseline_df.columns or 'num_points' not in baseline_df.columns:
        ax.text(0.5, 0.5, 'No scaling data available in baseline', 
                ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout(pad=2.0)
        return fig
        
    if 'experiment' not in target_df.columns or 'num_points' not in target_df.columns:
        ax.text(0.5, 0.5, 'No scaling data available in target', 
                ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout(pad=2.0)
        return fig
    
    # Calculate speedups for each experiment
    for experiment in baseline_df['experiment'].unique():
        baseline_exp_data = baseline_df[baseline_df['experiment'] == experiment].sort_values('num_points')
        target_exp_data = target_df[target_df['experiment'] == experiment].sort_values('num_points')
        
        if len(baseline_exp_data) > 1 and len(target_exp_data) > 1:
            # Find common mesh sizes between baseline and target
            common_points = set(baseline_exp_data['num_points']) & set(target_exp_data['num_points'])
            
            if len(common_points) > 1:
                speedups = []
                mesh_sizes = []
                
                for num_points in sorted(common_points):
                    baseline_perf = baseline_exp_data[baseline_exp_data['num_points'] == num_points][perf_col].iloc[0]
                    target_perf = target_exp_data[target_exp_data['num_points'] == num_points][perf_col].iloc[0]
                    
                    if baseline_perf > 0:  # Avoid division by zero
                        speedup = target_perf / baseline_perf
                        speedups.append(speedup)
                        mesh_sizes.append(num_points)
                
                if len(speedups) > 1:
                    # Find matching equation style
                    equation_key = 'default'
                    exp_lower = experiment.lower()
                    for key in equation_styles.keys():
                        if key in exp_lower:
                            equation_key = key
                            break
                    
                    style = equation_styles[equation_key]
                    ax.plot(mesh_sizes, speedups, 
                            color=style['color'],
                            marker=style['marker'],
                            # linestyle=style['linestyle'],
                            linewidth=style['linewidth'],
                            label=experiment.replace('_', ' ').title(),
                            markersize=8,
                            alpha=0.8)
    
    # # Add reference line at 1.0x (no speedup)
    # ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
    #           label=f'{baseline_backend.replace("_", " ").title()} (Baseline)')
    
    # Formatting
    # ax.set_xlabel('Number of Points', fontsize=14)
    # ax.set_ylabel(f'Speedup', fontsize=14)
    ax.set_xlabel('Number of Points')
    ax.set_ylabel(f'Speedup')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add more y-axis tick marks for better readability
    import numpy as np
    # Create more detailed y-axis ticks
    y_ticks = [0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45, 60.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}x' if tick >= 1 else f'{tick:.1f}x' for tick in y_ticks])
    
    # Enable minor ticks for even finer resolution
    ax.tick_params(which='minor')
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.15, which='minor')
    
    # ax.legend(fontsize=12, loc='upper left', framealpha=0.8)
    ax.legend(loc='upper left', ncols=2)
    
    plt.tight_layout(pad=2.0)
    return fig

def find_common_configurations(datasets):
    """Find configurations that exist across multiple datasets for fair comparison"""
    
    hw_datasets = {k: v for k, v in datasets.items()}
    
    # Find common equations and mesh sizes
    common_configs = {}
    
    # Get all experiments and mesh sizes from each dataset
    all_experiments = set()
    all_mesh_sizes = set()
    
    for name, df in hw_datasets.items():
        if 'experiment' in df.columns:
            all_experiments.update(df['experiment'].unique())
        if 'mesh_size' in df.columns:
            all_mesh_sizes.update(df['mesh_size'].unique())
    
    # Find configurations present in multiple datasets
    for experiment in all_experiments:
        for mesh_size in all_mesh_sizes:
            config_key = f"{experiment}_{mesh_size}"
            datasets_with_config = []
            
            for name, df in hw_datasets.items():
                if ('experiment' in df.columns and 'mesh_size' in df.columns and 
                    len(df[(df['experiment'] == experiment) & (df['mesh_size'] == mesh_size)]) > 0):
                    datasets_with_config.append(name)
            
            if len(datasets_with_config) >= 2:  # At least 2 datasets have this config
                common_configs[config_key] = {
                    'experiment': experiment,
                    'mesh_size': mesh_size,
                    'datasets': datasets_with_config
                }
    
    return common_configs

def perform_pairwise_comparisons(datasets):
    """Perform detailed pairwise comparisons between hardware setups"""
    
    # Filter hardware datasets only
    hw_datasets = {k: v for k, v in datasets.items()}
    hardware_names = list(hw_datasets.keys())
    
    print("\n" + "="*80)
    print("DETAILED PAIRWISE HARDWARE COMPARISONS")
    print("="*80)
    
    # Find common configurations
    common_configs = find_common_configurations(datasets)
    
    if not common_configs:
        print("No common configurations found for comparison")
        return None
    
    print(f"\nFound {len(common_configs)} common configurations for comparison\n")
    
    # Create comparison results structure
    comparison_results = {}
    
    # Analyze each common configuration
    for config_key, config_info in common_configs.items():
        experiment = config_info['experiment']
        mesh_size = config_info['mesh_size']
        available_datasets = config_info['datasets']
        
        print(f"Configuration: {experiment} @ {mesh_size}")
        print("-" * 50)
        
        config_results = {}
        
        # Get performance for each available dataset
        for dataset_name in available_datasets:
            df = hw_datasets[dataset_name]
            matching_rows = df[(df['experiment'] == experiment) & (df['mesh_size'] == mesh_size)]
            
            if len(matching_rows) > 0:
                max_perf = matching_rows['mpts'].max()
                avg_perf = matching_rows['mpts'].mean()
                config_results[dataset_name] = {
                    'max_mpts': max_perf,
                    'avg_mpts': avg_perf,
                    'count': len(matching_rows)
                }
                print(f"  {dataset_name:20}: {max_perf:8.1f} MPTS (avg: {avg_perf:6.1f})")
        
        # Calculate speedups relative to CPU if available
        if 'openmp_cpu' in config_results:
            cpu_perf = config_results['openmp_cpu']['max_mpts']
            print(f"\n  Speedups vs CPU:")
            for dataset_name, results in config_results.items():
                if dataset_name != 'openmp_cpu':
                    speedup = results['max_mpts'] / cpu_perf
                    print(f"    {dataset_name:18}: {speedup:6.1f}x")
        
        # A100 pipeline comparison
        if 'a100_pipeline' in config_results and 'a100_no_pipeline' in config_results:
            pipe_perf = config_results['a100_pipeline']['max_mpts']
            no_pipe_perf = config_results['a100_no_pipeline']['max_mpts']
            improvement = ((pipe_perf - no_pipe_perf) / no_pipe_perf) * 100
            print(f"\n  A100 Pipeline Impact: {improvement:+.1f}% ({pipe_perf:.1f} vs {no_pipe_perf:.1f} MPTS)")
        
        # A100 vs no-kernelbuild comparison
        if 'a100_no_pipeline' in config_results and 'a100_no_kernelbuild' in config_results:
            modern_perf = config_results['a100_no_pipeline']['max_mpts']
            legacy_perf = config_results['a100_no_kernelbuild']['max_mpts']
            improvement = ((modern_perf - legacy_perf) / legacy_perf) * 100
            print(f"  A100 KernelBuilder Impact: {improvement:+.1f}% ({modern_perf:.1f} vs {legacy_perf:.1f} MPTS)")
        
        comparison_results[config_key] = config_results
        print()
    
    return comparison_results

def create_pairwise_comparison_plots(datasets, comparison_results):
    """Create separate visualization plots for each equation type"""
    
    if not comparison_results:
        print("No comparison results available for plotting")
        return []
    
    # Organize data by equation type and extract mesh sizes
    equation_data = {}
    
    for config_key, config_data in comparison_results.items():
        if len(config_data) < 2:  # Skip configs with insufficient data
            continue
            
        # Parse equation and mesh size
        parts = config_key.split('_', 1)
        if len(parts) < 2:
            continue
            
        experiment = parts[0]
        mesh_size_str = parts[1]
        
        # Extract numeric mesh size for sorting (assume cubic meshes)
        try:
            # Handle formats like "(200, 200, 200)" or "200, 200, 200"
            numbers = re.findall(r'\d+', mesh_size_str)
            if len(numbers) >= 3:
                mesh_size = int(numbers[0]) * int(numbers[1]) * int(numbers[2])
            elif len(numbers) == 1:
                mesh_size = int(numbers[0]) ** 3
            else:
                continue
        except:
            continue
        
        if experiment not in equation_data:
            equation_data[experiment] = {}
        
        # Filter out zero performance values
        filtered_hardware_data = {}
        for hw_name, hw_data in config_data.items():
            if hw_data.get('max_mpts', 0) > 0:  # Only include non-zero performance
                filtered_hardware_data[hw_name] = hw_data
        
        if filtered_hardware_data:  # Only add if we have valid data
            equation_data[experiment][mesh_size] = {
                'mesh_size_str': mesh_size_str,
                'hardware_data': filtered_hardware_data
            }
    
    if not equation_data:
        print("No valid equation data found for plotting")
        return []
    
    # Color and marker mapping for hardware platforms (updated with no-kernelbuild)
    hardware_styles = {
        'a100_pipeline': {'color': '#2E8B57', 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'label': 'A100 (Pipeline)'},
        'a100_no_pipeline': {'color': '#4682B4', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'A100 (No Pipeline)'},
        'a100_no_kernelbuild': {'color': '#8B4513', 'marker': 'v', 'linestyle': '-.', 'linewidth': 2, 'label': 'A100 (Legacy/No KernelBuild)'},
        'rtx3060_cuda': {'color': '#FF6347', 'marker': '^', 'linestyle': '-.', 'linewidth': 2, 'label': 'RTX 3060'},
        'openmp_cpu': {'color': '#9370DB', 'marker': 'D', 'linestyle': ':', 'linewidth': 2, 'label': 'CPU (OpenMP)'}
    }
    
    # Create separate plots for each equation type
    figures = []
    
    for experiment, mesh_data in equation_data.items():
        print(f"Creating plot for equation: {experiment}")
        
        # Create individual plot for this equation
        # fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        # fig, ax = plt.subplots(1, 1, figsize=(12.5, 7.5))
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 11.5))
        # fig.suptitle(f'Hardware Performance Comparison: {experiment.replace("_", " ").title()} Equation', fontsize=16)
        
        # Organize data by hardware platform
        hardware_plots = {}
        
        for mesh_size, data in mesh_data.items():
            for hw_name, hw_data in data['hardware_data'].items():
                if hw_data.get('max_mpts', 0) > 0:  # Additional zero check
                    if hw_name not in hardware_plots:
                        hardware_plots[hw_name] = {'mesh_sizes': [], 'performances': []}
                    
                    hardware_plots[hw_name]['mesh_sizes'].append(mesh_size)
                    hardware_plots[hw_name]['performances'].append(hw_data['max_mpts'])
        
        # Plot lines for each hardware platform
        for hw_name, plot_data in hardware_plots.items():
            if hw_name in hardware_styles and len(plot_data['mesh_sizes']) > 0:
                # Sort by mesh size for proper line plotting
                sorted_data = sorted(zip(plot_data['mesh_sizes'], plot_data['performances']))
                mesh_sizes, performances = zip(*sorted_data)
                
                style = hardware_styles[hw_name]
                ax.plot(mesh_sizes, performances, 
                       color=style['color'], 
                       marker=style['marker'],
                       linestyle=style['linestyle'],
                       linewidth=style['linewidth'],
                       markersize=8,
                       label=style['label'],
                       alpha=0.8)
        
        # Formatting
        # ax.set_xlabel('Total Grid Points', fontsize=12)
        # ax.set_ylabel('Performance (MPTS)', fontsize=12)
        ax.set_xlabel('Total Grid Points')
        ax.set_ylabel('Performance (MPTS)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        # ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=2)
        
        # Format x-axis to show mesh points in millions
        ax.tick_params(axis='both', which='major')
        
        # Add mesh size labels for better readability
        all_mesh_sizes = sorted(set([ms for plot_data in hardware_plots.values() for ms in plot_data['mesh_sizes']]))
        if len(all_mesh_sizes) <= 10:
            xlabels = []
            for mesh_size in all_mesh_sizes:
                if mesh_size >= 1000000:
                    xlabels.append(f'{mesh_size/1000000:.1f}M')
                elif mesh_size >= 1000:
                    xlabels.append(f'{mesh_size/1000:.0f}K')
                else:
                    xlabels.append(f'{mesh_size}')
            
            ax.set_xticks(all_mesh_sizes)
            ax.set_xticklabels(xlabels, rotation=45)
        
        # Add performance statistics as text
        if hardware_plots:
            max_perf = max([max(plot_data['performances']) for plot_data in hardware_plots.values()])
            min_perf = min([min(plot_data['performances']) for plot_data in hardware_plots.values()])
            # ax.text(0.02, 0.98, f'Performance Range: {min_perf:.0f} - {max_perf:.0f} MPTS', 
            #        transform=ax.transAxes, fontsize=10, verticalalignment='top',
            #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        figures.append((f'pairwise_comparison_{experiment}', fig))
        print(f"  Created plot with {len(hardware_plots)} hardware platforms")
    
    print(f"Generated {len(figures)} equation-specific comparison plots")
    return figures

def create_pipeline_impact_summary_plot(comparison_results):
    """Create a summary plot showing pipeline impact across different equation types"""
    
    if not comparison_results:
        return None
    
    # Collect pipeline impact data by equation type
    equation_impacts = {}
    
    for config_key, config_data in comparison_results.items():
        if 'a100_pipeline' not in config_data or 'a100_no_pipeline' not in config_data:
            continue
            
        # Parse equation type
        parts = config_key.split('_', 1)
        if len(parts) < 2:
            continue
        experiment = parts[0]
        
        # Calculate pipeline impact
        pipe_perf = config_data['a100_pipeline']['max_mpts']
        no_pipe_perf = config_data['a100_no_pipeline']['max_mpts']
        
        if no_pipe_perf > 0:  # Avoid division by zero
            impact = ((pipe_perf - no_pipe_perf) / no_pipe_perf) * 100
            
            if experiment not in equation_impacts:
                equation_impacts[experiment] = []
            equation_impacts[experiment].append(impact)
    
    if not equation_impacts:
        return None
    
    # Create the plot
    # fig, ax2 = plt.subplots(1, 1, figsize=(8, 6.5))
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 7.5))
    # fig.suptitle('A100 Pipeline Impact Analysis', fontsize=16)
    
    # Plot 1: Average pipeline impact by equation type
    eq_names = []
    avg_impacts = []
    impact_stds = []
    colors = []
    
    import matplotlib.cm as cm
    color_map = cm.get_cmap('RdYlGn')  # Red for negative, green for positive
    
    for eq_name, impacts in equation_impacts.items():
        eq_names.append(eq_name.replace('_', ' ').title())
        avg_impact = np.mean(impacts)
        avg_impacts.append(avg_impact)
        impact_stds.append(np.std(impacts))
        
        # Color based on impact (green for positive, red for negative)
        colors.append(color_map(0.7 if avg_impact > 0 else 0.3))
    
    # bars = ax1.bar(eq_names, avg_impacts, yerr=impact_stds, capsize=5, 
    #                color=colors, alpha=0.7, edgecolor='black')
    
    # # Add value labels on bars
    # for bar, avg_impact in zip(bars, avg_impacts):
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2., 
    #             height + (0.5 if height >= 0 else -0.5),
    #             f'{avg_impact:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
    #             fontweight='bold', fontsize=10)
    
    # ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    # ax1.set_title('Average Pipeline Impact by Equation Type')
    # ax1.set_ylabel('Pipeline Impact (%)')
    # ax1.tick_params(axis='x', rotation=45)
    # ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of pipeline impacts
    all_impacts = []
    equation_labels = []
    
    for eq_name, impacts in equation_impacts.items():
        all_impacts.extend(impacts)
        equation_labels.extend([eq_name.replace('_', ' ').title()] * len(impacts))
    
    # Create violin plot or box plot
    unique_equations = list(equation_impacts.keys())
    impact_data = [equation_impacts[eq] for eq in unique_equations]
    eq_labels = [eq.replace('_', ' ').title() for eq in unique_equations]
    
    bp = ax2.boxplot(impact_data, labels=eq_labels, patch_artist=True)
    
    # Color the boxes
    for patch, eq_name in zip(bp['boxes'], unique_equations):
        avg_impact = np.mean(equation_impacts[eq_name])
        patch.set_facecolor(color_map(0.7 if avg_impact > 0 else 0.3))
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    # ax2.set_title('Pipeline Impact Distribution')
    ax2.set_ylabel('Pipeline Impact (%)')
    ax2.tick_params(axis='x', rotation=45)
    # ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_detailed_comparison_summary(comparison_results):
    """Generate detailed summary of pairwise comparisons"""
    
    if not comparison_results:
        return
    
    print("\n" + "="*80)
    print("CONFIGURATION-SPECIFIC PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Find best performing configurations for each hardware
    hardware_best = {}
    
    for config_key, config_data in comparison_results.items():
        for hw_name, hw_data in config_data.items():
            if hw_name not in hardware_best:
                hardware_best[hw_name] = {'config': config_key, 'mpts': hw_data['max_mpts']}
            elif hw_data['max_mpts'] > hardware_best[hw_name]['mpts']:
                hardware_best[hw_name] = {'config': config_key, 'mpts': hw_data['max_mpts']}
    
    print("\nBest Configuration for Each Hardware:")
    print("-" * 40)
    for hw_name, best_info in hardware_best.items():
        config_parts = best_info['config'].split('_', 1)
        experiment = config_parts[0] if len(config_parts) > 0 else "unknown"
        mesh_size = config_parts[1] if len(config_parts) > 1 else "unknown"
        print(f"{hw_name:20}: {best_info['mpts']:8.1f} MPTS ({experiment} @ {mesh_size})")
    
    # Pipeline impact analysis across configurations
    pipeline_impacts = []
    for config_key, config_data in comparison_results.items():
        if 'a100_pipeline' in config_data and 'a100_no_pipeline' in config_data:
            pipe_perf = config_data['a100_pipeline']['max_mpts']
            no_pipe_perf = config_data['a100_no_pipeline']['max_mpts']
            impact = ((pipe_perf - no_pipe_perf) / no_pipe_perf) * 100
            pipeline_impacts.append(impact)
    
    if pipeline_impacts:
        avg_impact = np.mean(pipeline_impacts)
        print(f"\nA100 Pipeline Impact Analysis:")
        print(f"  Average improvement: {avg_impact:+.1f}%")
        print(f"  Range: {min(pipeline_impacts):+.1f}% to {max(pipeline_impacts):+.1f}%")
        print(f"  Configurations analyzed: {len(pipeline_impacts)}")
    
    # KernelBuilder impact analysis
    kernelbuild_impacts = []
    for config_key, config_data in comparison_results.items():
        if 'a100_no_pipeline' in config_data and 'a100_no_kernelbuild' in config_data:
            modern_perf = config_data['a100_no_pipeline']['max_mpts']
            legacy_perf = config_data['a100_no_kernelbuild']['max_mpts']
            impact = ((modern_perf - legacy_perf) / legacy_perf) * 100
            kernelbuild_impacts.append(impact)
    
    if kernelbuild_impacts:
        avg_impact = np.mean(kernelbuild_impacts)
        print(f"\nA100 KernelBuilder Impact Analysis:")
        print(f"  Average improvement: {avg_impact:+.1f}%")
        print(f"  Range: {min(kernelbuild_impacts):+.1f}% to {max(kernelbuild_impacts):+.1f}%")
        print(f"  Configurations analyzed: {len(kernelbuild_impacts)}")
    
    # Cross-platform speedup analysis
    print(f"\nCross-Platform Speedup Summary:")
    print("-" * 40)
    
    for config_key, config_data in comparison_results.items():
        if 'openmp_cpu' not in config_data:
            continue
            
        cpu_perf = config_data['openmp_cpu']['max_mpts']
        config_parts = config_key.split('_', 1)
        experiment = config_parts[0] if len(config_parts) > 0 else config_key
        
        print(f"\n{experiment.title()} Equation:")
        for hw_name, hw_data in config_data.items():
            if hw_name != 'openmp_cpu':
                speedup = hw_data['max_mpts'] / cpu_perf
                print(f"  {hw_name:18}: {speedup:6.1f}x speedup ({hw_data['max_mpts']:6.1f} vs {cpu_perf:6.1f} MPTS)")


def generate_performance_summary(datasets):
    """Generate comprehensive performance summary"""
    
    print("\n" + "="*60)
    print("CFDARCO PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall statistics
    for name, df in datasets.items():
        if len(df) > 0:
            print(f"\n{name.upper().replace('_', ' ')} DATASET:")
            print("-" * 30)
            print(f"Records: {len(df)}")

            if 'experiment' in df.columns:
                print(f"Equations tested: {df['experiment'].nunique()}")
            if 'mpts' in df.columns:
                print(f"Peak performance: {df['mpts'].max():.1f} MPTS")
                print(f"Average performance: {df['mpts'].mean():.1f} MPTS")

                # Top performers for regular datasets
                if 'experiment' in df.columns and 'mesh_size' in df.columns:
                    top_performers = df.nlargest(3, 'mpts')[['experiment', 'mesh_size', 'mpts']]
                    print("\nTop 3 Performers:")
                    for _, row in top_performers.iterrows():
                        print(f"  {row['experiment']:15} {str(row['mesh_size']):15} {row['mpts']:8.1f} MPTS")


def create_pairwise_speedup_scaling_plots(datasets, comparison_results):
    """Create separate speedup scaling plots for each equation type (similar to pairwise MPTS plots)"""
    
    if not comparison_results:
        print("No comparison results available for speedup scaling plots")
        return []
    
    # Define reference backend for speedup calculations
    reference_backend = 'a100_no_kernelbuild'
    # skip_backends = ['openmp_cpu', 'rtx3060_cuda']
    skip_backends = ['openmp_cpu']

    # Organize data by equation type and extract mesh sizes
    equation_data = {}
    
    for config_key, config_data in comparison_results.items():
        if len(config_data) < 2:  # Skip configs with insufficient data
            continue
            
        # Parse equation and mesh size
        parts = config_key.split('_', 1)
        if len(parts) < 2:
            continue
            
        experiment = parts[0]
        mesh_size_str = parts[1]
        
        # Extract numeric mesh size for sorting (assume cubic meshes)
        try:
            # Handle formats like "(200, 200, 200)" or "200, 200, 200"
            numbers = re.findall(r'\d+', mesh_size_str)
            if len(numbers) >= 3:
                mesh_size = int(numbers[0]) * int(numbers[1]) * int(numbers[2])
            elif len(numbers) == 1:
                mesh_size = int(numbers[0]) ** 3
            else:
                continue
        except:
            continue
        
        if experiment not in equation_data:
            equation_data[experiment] = {}
        
        # Filter out zero performance values and calculate speedups
        if reference_backend in config_data:
            ref_perf = config_data[reference_backend].get('max_mpts', 0)
            
            if ref_perf > 0:  # Only process if we have valid reference performance
                speedup_data = {}
                for hw_name, hw_data in config_data.items():
                    if hw_name != reference_backend and hw_name not in skip_backends and hw_data.get('max_mpts', 0) > 0:
                        speedup = hw_data['max_mpts'] / ref_perf
                        speedup_data[hw_name] = {
                            'speedup': speedup,
                            'max_mpts': hw_data['max_mpts']
                        }
                
                if speedup_data:  # Only add if we have valid speedup data
                    equation_data[experiment][mesh_size] = {
                        'mesh_size_str': mesh_size_str,
                        'reference_mpts': ref_perf,
                        'speedup_data': speedup_data
                    }
    
    if not equation_data:
        print("No valid equation data found for speedup scaling plots")
        return []
    
    # Color and marker mapping for hardware platforms (excluding reference)
    hardware_styles = {
        'a100_pipeline': {'color': '#2E8B57', 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'label': 'A100 (Pipeline)'},
        'a100_no_pipeline': {'color': '#4682B4', 'marker': 's', 'linestyle': '--', 'linewidth': 2, 'label': 'A100 (No Pipeline)'},
        'rtx3060_cuda': {'color': '#FF6347', 'marker': '^', 'linestyle': '-.', 'linewidth': 2, 'label': 'RTX 3060'},
        'openmp_cpu': {'color': '#9370DB', 'marker': 'D', 'linestyle': ':', 'linewidth': 2, 'label': 'CPU (OpenMP)'}
    }
    
    # Create separate plots for each equation type
    figures = []
    
    for experiment, mesh_data in equation_data.items():
        print(f"Creating speedup scaling plot for equation: {experiment}")
        
        # Create individual plot for this equation
        # fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        # fig, ax = plt.subplots(1, 1, figsize=(12.5, 7.5))
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 10.5))
        # fig.suptitle(f'Speedup Scaling vs A100 (Legacy): {experiment.replace("_", " ").title()} Equation', fontsize=16)
        
        # Organize data by hardware platform
        hardware_plots = {}
        
        for mesh_size, data in mesh_data.items():
            for hw_name, hw_data in data['speedup_data'].items():
                if hw_data['speedup'] > 0:  # Additional zero check
                    if hw_name not in hardware_plots:
                        hardware_plots[hw_name] = {'mesh_sizes': [], 'speedups': []}
                    
                    hardware_plots[hw_name]['mesh_sizes'].append(mesh_size)
                    hardware_plots[hw_name]['speedups'].append(hw_data['speedup'])
        
        # Prepare data for grouped bar plot
        all_mesh_sizes = sorted(set([ms for plot_data in hardware_plots.values() for ms in plot_data['mesh_sizes']]))
        
        # ax.set_ylabel('Speedup', fontsize=14)
        ax.set_ylabel('Speedup')
        # ax.set_yscale('log')

        if all_mesh_sizes:
            # Set up bar positions
            x = np.arange(len(all_mesh_sizes))
            # bar_width = 0.4
            bar_width = 0.315
            hardware_names = [hw_name for hw_name in hardware_styles.keys() if hw_name in hardware_plots]
            
            # Create bars for each hardware platform
            for i, hw_name in enumerate(hardware_names):
                if hw_name in hardware_plots and hw_name in hardware_styles:
                    plot_data = hardware_plots[hw_name]
                    style = hardware_styles[hw_name]
                    
                    # Sort by mesh size and align with all_mesh_sizes
                    speedup_values = []
                    for mesh_size in all_mesh_sizes:
                        if mesh_size in plot_data['mesh_sizes']:
                            idx = plot_data['mesh_sizes'].index(mesh_size)
                            speedup_values.append(plot_data['speedups'][idx])
                        else:
                            speedup_values.append(0)  # No data for this mesh size
                    
                    bars = ax.bar(x + i * bar_width, speedup_values, bar_width, 
                                 label=style['label'], color=style['color'], alpha=0.85, 
                                 edgecolor='black', linewidth=0.5)
                    
                    # Add value labels on bars
                    for bar, speedup in zip(bars, speedup_values):
                        if speedup > 0:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/1.8, height * 1.05,
                                   f'{speedup:.2f}x', ha='center', va='bottom', fontsize=separate_upper_label_size, rotation=90)
        
        # # Add reference line at 1.0x
        # ax.axhline(y=1.0, color='red', linestyle='-', alpha=0.8, linewidth=2, 
        #           label=f'{reference_backend.replace("_", " ").title()} (Reference)')
        
        # Formatting
        # ax.set_xlabel('Mesh Configuration', fontsize=12)
        ax.set_xlabel('Mesh Configuration')

        ax.grid(True, alpha=0.3, axis='y')
        # ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.9)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        
        # Format x-axis to show mesh configurations
        ax.tick_params(axis='both', which='major')
        
        # Set x-axis labels with mesh size information if we have bar data
        if all_mesh_sizes:
            xlabels = []
            for mesh_size in all_mesh_sizes:
                if mesh_size >= 1000000:
                    xlabels.append(f'{mesh_size/1000000:.1f}M')
                elif mesh_size >= 1000:
                    xlabels.append(f'{mesh_size/1000:.0f}K')
                else:
                    xlabels.append(f'{mesh_size}')
            
            # Center the bars and set labels
            if 'x' in locals() and 'hardware_names' in locals():
                ax.set_xticks(x + bar_width * (len(hardware_names) - 1) / 2)
                ax.set_xticklabels(xlabels, rotation=45, ha='right')
        
        # Add top margin for text labels
        if ax.get_ylim()[1] > 0:
            current_ylim = ax.get_ylim()
            ax.set_ylim(current_ylim[0], current_ylim[1] * 1.2)  # Add 15% top margin
        
        plt.tight_layout()

        figures.append((f'speedup_scaling_{experiment}', fig))
        plt.close()
    
    print(f"Generated {len(figures)} equation-specific speedup scaling plots")
    return figures


def create_permesh_plots(datasets, comparison_results):
    """Create separate plots for each mesh size showing MPTS performance across backends and equations"""
    
    if not comparison_results:
        print("No comparison results available for per-mesh plots")
        return []
    
    # Organize data by mesh size
    mesh_data = {}

    skip_backends = ['a100_no_pipeline']
    
    for config_key, config_data in comparison_results.items():
        if len(config_data) < 1:  # Skip configs with no data
            continue
            
        # Parse equation and mesh size
        parts = config_key.split('_', 1)
        if len(parts) < 2:
            continue
            
        experiment = parts[0]
        mesh_size_str = parts[1]
        
        # Extract numeric mesh size for sorting and grouping (using only first 2 dimensions)
        try:
            # Handle formats like "(200, 200, 200)" or "200, 200, 200"
            numbers = re.findall(r'\d+', mesh_size_str)
            if len(numbers) >= 2:
                mesh_size = int(numbers[0]) * int(numbers[0]) * int(numbers[0])
        except:
            continue
        
        # Initialize mesh size entry
        if mesh_size not in mesh_data:
            mesh_data[mesh_size] = {
                'mesh_size_str': str(mesh_size),
                'experiments': {}
            }
        
        # Filter out zero performance values and organize by experiment
        filtered_hardware_data = {}
        for hw_name, hw_data in config_data.items():
            if hw_data.get('max_mpts', 0) > 0:  # Only include non-zero performance
                filtered_hardware_data[hw_name] = hw_data['max_mpts']
        
        if filtered_hardware_data:  # Only add if we have valid data
            mesh_data[mesh_size]['experiments'][experiment] = filtered_hardware_data
    
    if not mesh_data:
        print("No valid mesh data found for per-mesh plots")
        return []
    
    # Color mapping for hardware platforms
    hardware_styles = {
        'a100_pipeline': {'color': '#2E8B57', 'label': 'A100 (Pipeline)'},
        'a100_no_pipeline': {'color': '#4682B4', 'label': 'A100 (No Pipeline)'},
        'a100_no_kernelbuild': {'color': '#8B4513', 'label': 'A100 (Legacy/No KernelBuild)'},
        'rtx3060_cuda': {'color': '#FF6347', 'label': 'RTX 3060'},
        'openmp_cpu': {'color': '#9370DB', 'label': 'CPU (OpenMP)'}
    }
    
    # Create separate plots for each mesh size
    figures = []
    
    for mesh_size, data in sorted(mesh_data.items()):
        mesh_size_str = data['mesh_size_str']
        experiments_data = data['experiments']
        
        if not experiments_data:
            continue
            
        print(f"Creating per-mesh plot for mesh size: {mesh_size_str}")
        
        # Create individual plot for this mesh size
        # fig, ax = plt.subplots(1, 1, figsize=(11.5, 6.5))
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 7.5))
        
        # Format mesh size for title
        if mesh_size >= 1000000:
            mesh_display = f'{mesh_size/1000000:.1f}M points'
        elif mesh_size >= 1000:
            mesh_display = f'{mesh_size/1000:.0f}K points'
        else:
            mesh_display = f'{mesh_size} points'
            
        # fig.suptitle(f'Performance Comparison by Equation Type\nMesh Size {mesh_display}', fontsize=16)
        
        # Prepare data for grouped bar plot
        experiments = sorted(experiments_data.keys())
        
        # Get all unique backends across all experiments for this mesh size
        all_backends = set()
        for exp_data in experiments_data.values():
            all_backends.update(exp_data.keys())
        
        # Filter to only backends we have styles for and sort consistently
        available_backends = [hw for hw in hardware_styles.keys() if hw in all_backends]

        for exclude_name in skip_backends:
            if exclude_name in available_backends:
                available_backends.remove(exclude_name)
        
        if not experiments or not available_backends:
            continue
        
        # Set up bar positions
        x = np.arange(len(experiments))
        bar_width = 0.215  # Proper width to avoid overlap with multiple backends
        
        # Create bars for each backend
        for i, backend in enumerate(available_backends):
            if backend in hardware_styles:
                style = hardware_styles[backend]
                
                # Collect MPTS values for this backend across all experiments
                mpts_values = []
                for experiment in experiments:
                    exp_data = experiments_data[experiment]
                    mpts_values.append(exp_data.get(backend, 0))  # 0 if backend not available for this experiment
                
                # Only plot if we have some non-zero values
                if any(val > 0 for val in mpts_values):
                    bars = ax.bar(x + i * bar_width, mpts_values, bar_width, 
                                 label=style['label'], color=style['color'], alpha=0.85, 
                                 edgecolor='black', linewidth=0.5)
                    
                    # Add value labels on bars
                    for bar, mpts in zip(bars, mpts_values):
                        if mpts > 0:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/1.8, height * 1.05,
                                   f'{mpts:.0f}', ha='center', va='bottom', fontsize=separate_upper_label_size, rotation=90)
        
        # Formatting
        # ax.set_xlabel('Equation Type', fontsize=14)
        # ax.set_ylabel('Performance (MPTS)', fontsize=14)
        ax.set_ylabel('Performance (MPTS)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        # ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, framealpha=0.9)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        
        # Format x-axis labels
        experiment_labels = [exp.replace('_', ' ').title() for exp in experiments]
        ax.set_xticks(x + bar_width * (len(available_backends) - 1) / 2)
        ax.set_xticklabels(experiment_labels, rotation=45, ha='right')

        # Add top margin for text labels
        if ax.get_ylim()[1] > 0:
            current_ylim = ax.get_ylim()
            ax.set_ylim(current_ylim[0], current_ylim[1] * 3.1)  # Add 15% top margin

        plt.tight_layout(pad=2.0)
        
        # Create filename-safe mesh size string
        mesh_safe = re.sub(r'[^\w\-_]', '_', mesh_size_str)
        figures.append((f'permesh_{mesh_safe}', fig))
        print(f"  Created plot with {len(experiments)} equations and {len(available_backends)} backends")
    
    print(f"Generated {len(figures)} mesh-specific comparison plots")
    return figures

def create_permesh_speedup_plots(datasets, comparison_results):
    """Create separate speedup plots for each mesh size showing speedup performance across backends and equations"""
    
    if not comparison_results:
        print("No comparison results available for per-mesh speedup plots")
        return []
    
    # Define reference backend for speedup calculations
    # reference_backend = 'openmp_cpu'  # Use CPU as baseline for speedup
    reference_backend = 'a100_no_kernelbuild'
    # skip_backends = ['openmp_cpu', 'rtx3060_cuda']
    skip_backends = ['a100_no_pipeline']
    # Organize data by mesh size
    mesh_data = {}
    
    for config_key, config_data in comparison_results.items():
        if len(config_data) < 1:  # Skip configs with no data
            continue
            
        # Parse equation and mesh size
        parts = config_key.split('_', 1)
        if len(parts) < 2:
            continue
            
        experiment = parts[0]
        mesh_size_str = parts[1]
        
        # Extract numeric mesh size for sorting and grouping (using cube of first dimension)
        try:
            # Handle formats like "(200, 200, 200)" or "200, 200, 200"
            numbers = re.findall(r'\d+', mesh_size_str)
            if len(numbers) >= 2:
                mesh_size = int(numbers[0]) * int(numbers[0]) * int(numbers[0])
            else:
                continue
        except:
            continue
        
        # Initialize mesh size entry
        if mesh_size not in mesh_data:
            mesh_data[mesh_size] = {
                'mesh_size_str': str(mesh_size),
                'experiments': {}
            }

        for exclude_name in skip_backends:
            if exclude_name in config_data:
                del config_data[exclude_name]
    

        # Calculate speedups relative to reference backend
        if reference_backend in config_data:
            ref_perf = config_data[reference_backend].get('max_mpts', 0)
            
            if ref_perf > 0:  # Only process if we have valid reference performance
                speedup_data = {}
                for hw_name, hw_data in config_data.items():
                    if hw_name != reference_backend and hw_data.get('max_mpts', 0) > 0:
                        speedup = hw_data['max_mpts'] / ref_perf
                        speedup_data[hw_name] = speedup
                
                # # Add reference backend with 1.0x speedup
                # speedup_data[reference_backend] = 1.0
                
                if speedup_data:  # Only add if we have valid speedup data
                    mesh_data[mesh_size]['experiments'][experiment] = speedup_data
        

    if not mesh_data:
        print("No valid mesh data found for per-mesh speedup plots")
        return []
    
    # Color mapping for hardware platforms
    hardware_styles = {
        'a100_pipeline': {'color': '#2E8B57', 'label': 'A100 (Pipeline)'},
        'a100_no_pipeline': {'color': '#4682B4', 'label': 'A100 (No Pipeline)'},
        'a100_no_kernelbuild': {'color': '#8B4513', 'label': 'A100 (Legacy/No KernelBuild)'},
        'rtx3060_cuda': {'color': '#FF6347', 'label': 'RTX 3060'},
        'openmp_cpu': {'color': '#9370DB', 'label': 'CPU (OpenMP)'}
    }
    
    # Create separate plots for each mesh size
    figures = []
    
    for mesh_size, data in sorted(mesh_data.items()):
        mesh_size_str = data['mesh_size_str']
        experiments_data = data['experiments']
        
        if not experiments_data:
            continue
            
        print(f"Creating per-mesh speedup plot for mesh size: {mesh_size_str}")
        
        # Create individual plot for this mesh size
        # fig, ax = plt.subplots(1, 1, figsize=(11.5, 6.5))
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 10.5))
        
        # Format mesh size for title
        if mesh_size >= 1000000:
            mesh_display = f'{mesh_size/1000000:.1f}M points'
        elif mesh_size >= 1000:
            mesh_display = f'{mesh_size/1000:.0f}K points'
        else:
            mesh_display = f'{mesh_size} points'
            
        # fig.suptitle(f'Speedup Comparison by Equation Type\nMesh Size {mesh_display}', fontsize=16)
        
        # Prepare data for grouped bar plot
        experiments = sorted(experiments_data.keys())
        
        # Get all unique backends across all experiments for this mesh size
        all_backends = set()
        for exp_data in experiments_data.values():
            all_backends.update(exp_data.keys())
        
        # Filter to only backends we have styles for and sort consistently
        available_backends = [hw for hw in hardware_styles.keys() if hw in all_backends]
        
        if not experiments or not available_backends:
            continue
        
        # Set up bar positions
        x = np.arange(len(experiments))
        # bar_width = 0.215  # Proper width to avoid overlap with multiple backends
        bar_width = 0.3  # Proper width to avoid overlap with multiple backends

        #  ax.set_ylabel('Speedup', fontsize=14)
        ax.set_ylabel('Speedup')
        ax.set_yscale('log')
        
        # Create bars for each backend
        for i, backend in enumerate(available_backends):
            if backend in hardware_styles:
                style = hardware_styles[backend]
                
                # Collect speedup values for this backend across all experiments
                speedup_values = []
                for experiment in experiments:
                    exp_data = experiments_data[experiment]
                    speedup_values.append(exp_data.get(backend, 0))  # 0 if backend not available for this experiment
                
                # Only plot if we have some non-zero values
                if any(val > 0 for val in speedup_values):
                    bars = ax.bar(x + i * bar_width, speedup_values, bar_width, 
                                 label=style['label'], color=style['color'], alpha=0.85, 
                                 edgecolor='black', linewidth=0.5)
                    
                    # Add value labels on bars
                    for bar, speedup in zip(bars, speedup_values):
                        if speedup > 0:
                            height = bar.get_height()
                            # ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                            #        f'{speedup:.2f}x', ha='center', va='bottom', 
                            #        fontweight='bold', fontsize=12, rotation=90)
                            ax.text(bar.get_x() + bar.get_width()/1.45, height * 1.05,
                                   f'{speedup:.2f}x', ha='center', va='bottom', fontsize=separate_upper_label_size, rotation=90)
        
        # # Add reference line at 1.0x
        # ax.axhline(y=1.0, color='red', linestyle='-', alpha=0.8, linewidth=2, 
        #           label=f'{reference_backend.replace("_", " ").title()} (Baseline)')
        
        # Formatting
        # ax.set_xlabel('Equation Type', fontsize=14)

        ax.grid(True, alpha=0.3, axis='y')
        # ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, framealpha=0.9)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, framealpha=0.9)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        
        # Format x-axis labels
        experiment_labels = [exp.replace('_', ' ').title() for exp in experiments]
        ax.set_xticks(x + bar_width * (len(available_backends) - 1) / 2)
        ax.set_xticklabels(experiment_labels, rotation=45, ha='right')
        
        # Add top margin for text labels
        if ax.get_ylim()[1] > 0:
            current_ylim = ax.get_ylim()
            ax.set_ylim(current_ylim[0], current_ylim[1] * 2.3)  # Add 15% top margin
        
        plt.tight_layout(pad=2.0)
        
        # Create filename-safe mesh size string
        mesh_safe = re.sub(r'[^\w\-_]', '_', mesh_size_str)
        figures.append((f'permesh_speedup_{mesh_safe}', fig))
        print(f"  Created speedup plot with {len(experiments)} equations and {len(available_backends)} backends")
    
    print(f"Generated {len(figures)} mesh-specific speedup plots")
    return figures


def main():
    """Main analysis function"""
    
    parser = argparse.ArgumentParser(description='Analyze cfdARCO performance results')
    parser.add_argument('-d', '--results-dir', default='results',
                       help='Directory containing results files (default: results)')
    parser.add_argument('-o', '--output-dir', default='.',
                       help='Output directory for plots (default: current directory)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively')
    parser.add_argument('--detailed-comparison', action='store_true',
                       help='Include detailed pairwise comparisons')
    
    args = parser.parse_args()
    
    print("Starting cfdARCO Performance Analysis...")
    print(f"Looking for results in: {args.results_dir}")
    
    # Load data
    datasets = load_benchmark_data(args.results_dir)
    
    if not datasets:
        print("No benchmark data found!")
        return
    
    # Generate summary
    generate_performance_summary(datasets)
    
    # Perform detailed pairwise comparisons
    comparison_results = perform_pairwise_comparisons(datasets)
    if comparison_results:
        generate_detailed_comparison_summary(comparison_results)
    
    if args.no_plots:
        return
    
    # Create visualizations
    figures = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analyze each dataset
    for name, df in datasets.items():
        fig = analyze_scaling_performance(df, f"({name.replace('_', ' ').title()})")
        if fig:
            figures.append((f'{name}_analysis', fig))

    # Create speedup scaling plots relative to legacy A100 baseline
    baseline_backend = 'a100_no_kernelbuild'
    if baseline_backend in datasets:
        for name in datasets.keys():
            if name != baseline_backend:  # Don't compare baseline to itself
                fig = analyze_speedup_scaling_performance(datasets, name, baseline_backend)
                if fig:
                    figures.append((f'{name}_speedup_scaling_vs_{baseline_backend}', fig))

    # Create pairwise comparison plots (separate plots for each equation)
    if comparison_results:
        pairwise_figs = create_pairwise_comparison_plots(datasets, comparison_results)
        if pairwise_figs:
            figures.extend(pairwise_figs)
    
    # Create pipeline impact summary plot
    if comparison_results:
        fig = create_pipeline_impact_summary_plot(comparison_results)
        if fig:
            figures.append(('pipeline_impact_summary', fig))
    

    # Create pairwise speedup scaling plots
    if comparison_results:
        pairwise_figs = create_pairwise_speedup_scaling_plots(datasets, comparison_results)
        if pairwise_figs:
            figures.extend(pairwise_figs)

    # # Create per-mesh comparison plots
    # if comparison_results:
    #     permesh_figs = create_permesh_plots(datasets, comparison_results)
    #     if permesh_figs:
    #         figures.extend(permesh_figs)

    # Create per-mesh speedup plots
    if comparison_results:
        permesh_speedup_figs = create_permesh_speedup_plots(datasets, comparison_results)
        if permesh_speedup_figs:
            figures.extend(permesh_speedup_figs)

    # Save figures
    for name, fig in figures:
        filename = output_dir / f"{name}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
    
    print(f"\nAnalysis complete! Plots saved to {output_dir}")

if __name__ == "__main__":
    main() 