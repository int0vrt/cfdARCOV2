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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

df_krnbuilder = pd.read_csv("report_history_a100_cuda_full_no_data_move_even_better.csv")
df_krnbuilder = df_krnbuilder[df_krnbuilder["cuda"] == True]

df_legacy_kernels = pd.read_csv("report_history_a100_cuda_full_no_data_move_even_better_no_kernelbuilder.csv")
df_legacy_kernels = df_legacy_kernels[df_legacy_kernels["cuda"] == True]


fontsize = 22
font = {'size'   : fontsize}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)


df_part_euler_krnbuilder = df_krnbuilder[df_krnbuilder["experiment"] == "euler"]
df_part_heat_krnbuilder  = df_krnbuilder[df_krnbuilder["experiment"] == "heat"]
# df_part_wave_krnbuilder  = df_krnbuilder[df_krnbuilder["experiment"] == "wave"]

df_part_euler_legacy_kernels = df_legacy_kernels[df_legacy_kernels["experiment"] == "euler"]
df_part_heat_legacy_kernels  = df_legacy_kernels[df_legacy_kernels["experiment"] == "heat"]
# df_part_wave_legacy_kernels  = df_legacy_kernels[df_legacy_kernels["experiment"] == "wave"]

plt.plot(df_part_heat_legacy_kernels["num_points"], df_part_heat_krnbuilder["mpts"] / df_part_heat_legacy_kernels["mpts"], label="Heat equation")
# plt.plot(df_part_wave_legacy_kernels["num_points"], df_part_wave_legacy_kernels["mpts"] / df_part_wave_krnbuilder["mpts"], label="Wave equation")
plt.plot(df_part_euler_legacy_kernels["num_points"], df_part_euler_krnbuilder["mpts"] / df_part_euler_legacy_kernels["mpts"], label="Euler equations")

ax.set_xlabel('Num of Points')
ax.set_ylabel('Times, (Original perf / KernelBuilder perf)')
plt.legend()

fig.savefig('kernelbuilder_compare_a100.pdf', dpi=300)
plt.cla()
