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

df = pd.read_csv("report_history_a100_cuda_full_no_data_move_even_better.csv")
df = df[df["cuda"] == True]

fontsize = 21
font = {'size'   : fontsize}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)

df_part_euler = df[df["experiment"] == "euler"]
df_part_heat = df[df["experiment"] == "heat"]
df_part_wave = df[df["experiment"] == "wave"]

plt.plot(df_part_heat["num_points"], df_part_heat["mpts"], label="Heat equation")
plt.plot(df_part_wave["num_points"], df_part_wave["mpts"], label="Acoustic wave equations")
# plt.yscale('log')

ax.set_xlabel('Num of Points')
ax.set_ylabel('Mpts/sec')
plt.legend()

fig.savefig('scaling_simple_eqs_a100.pdf', dpi=300)
plt.cla()
ax.cla()

plt.plot(df_part_euler["num_points"], df_part_euler["mpts"], label="Euler equations")
# plt.yscale('log')

ax.set_xlabel('Num of Points')
ax.set_ylabel('Mpts/sec')
plt.legend()

fig.savefig('scaling_euler_eqs_a100.pdf', dpi=300)
plt.cla()