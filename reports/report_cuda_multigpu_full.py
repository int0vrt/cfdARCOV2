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

df = pd.read_csv("report_cuda_multi_gpu_history.csv")

df_std = df.groupby(["mesh_sizes"]).std().reset_index()
df = df.groupby(["mesh_sizes"]).min().reset_index()


df["time_microseconds_cuda_single_gpu"] = df["time_microseconds_cuda_single_gpu"] / 1000000
df["time_microseconds_cuda_multi_gpu"] = df["time_microseconds_cuda_multi_gpu"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2
df_std["time_microseconds_cuda_single_gpu"] = df_std["time_microseconds_cuda_single_gpu"] / 1000000
df_std["time_microseconds_cuda_multi_gpu"] = df_std["time_microseconds_cuda_multi_gpu"] / 1000000
df_std["mesh_sizes"] = df_std["mesh_sizes"] ** 2

print(df)
print(df_std)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["time_microseconds_cuda_single_gpu"], label="One GPU")
plt.plot(df["mesh_sizes"], df["time_microseconds_cuda_multi_gpu"], label="Two GPUs")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()
plt.savefig('cuda_multi_gpu_time.pdf')


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["time_microseconds_cuda_single_gpu"] / df["time_microseconds_cuda_multi_gpu"], label="Time 2 GPUs faster")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Times')
plt.legend()
plt.savefig('cuda_multi_gpu_speedup.pdf')

plt.show()
