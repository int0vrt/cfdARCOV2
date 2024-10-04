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

df_cu = pd.read_csv("report_history_rtx3060_no_cuda_boundary.csv")
df_cu = df_cu[df_cu["cuda"] == True]

df_hip = pd.read_csv("report_history_rtx3060_hip_no_cuda_boundary.csv")
df_hip = df_hip[df_hip["cuda"] == True]


fontsize = 21
font = {'size'   : fontsize}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(11, 8), tight_layout=True)


df_part_heat_cu = df_cu[df_cu["experiment"] == "heat"]
df_part_wave_cu = df_cu[df_cu["experiment"] == "wave"]

df_part_heat_hip = df_hip[df_hip["experiment"] == "heat"]
df_part_wave_hip = df_hip[df_hip["experiment"] == "wave"]

plt.plot(df_part_heat_cu["num_points"], df_part_heat_cu["mpts"] / df_part_heat_hip["mpts"], label="Heat equation")
plt.plot(df_part_wave_cu["num_points"], df_part_wave_cu["mpts"] / df_part_wave_hip["mpts"], label="Acoustic wave equations")

ax.set_xlabel('Num of Points')
ax.set_ylabel('Times, (CUDA perf / HIP emulated perf)')
plt.legend()

fig.savefig('cuda_hip_compare_rtx3060.pdf', dpi=300)
plt.cla()
