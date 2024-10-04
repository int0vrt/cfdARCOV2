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

df = pd.read_csv("report_cuda_2_proc.csv")

df["times_microseconds_parallel"] = df["times_microseconds_parallel"] / 1000000
df["times_microseconds_cuda"] = df["times_microseconds_cuda"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2

print(df)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"], label="CPU, parallel")
plt.plot(df["mesh_sizes"], df["times_microseconds_cuda"], label="CUDA")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Time, seconds')
plt.legend()


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
plt.plot(df["mesh_sizes"], df["times_microseconds_parallel"] / df["times_microseconds_cuda"], label="Time CUDA faster")
ax.set_xlabel('Mesh size')
ax.set_ylabel('Timess')
plt.legend()

plt.show()
