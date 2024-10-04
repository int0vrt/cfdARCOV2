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

df = pd.read_csv("report_per_nodes_8_meshines_aws.csv")


df["times_microseconds"] = df["times_microseconds"] / 1000000
df["mesh_sizes"] = df["mesh_sizes"] ** 2

df = df.pivot(index='num_proc', columns='mesh_sizes', values='times_microseconds')
df = 1 / df.div(df.iloc[0])
df = df[:8]
print(df)
df.plot(legend=True)
plt.show()
