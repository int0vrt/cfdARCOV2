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

df = pd.read_csv("report_per_nodes_full_reduced.csv", index_col=None)
df = df.drop(columns=["Unnamed: 0"])
df["Mesh sizes"] = df["mesh_sizes"] ** 2
df["Number of nodes"] = df["num_proc"]
df["times_microseconds"] = df["times_microseconds"] / 1000000

df.drop(columns=["num_proc", "mesh_sizes"])

df_group = df.groupby(['Number of nodes', 'Mesh sizes'])


df = df_group.min().reset_index()

df = df.pivot(index='Number of nodes', columns='Mesh sizes', values='times_microseconds')
df = 1 / df.div(df.iloc[0])
print(df)
fig, ax = plt.subplots(1,1)
df.plot(legend=True, ax=ax)
# ax.axline((0, 0), slope=1, linestyle='dashed', color="black")
ax.plot([0,1],[0,1], transform=ax.transAxes, linestyle='dashed', color="black")
ax.set_ylabel("Speedup, times")
plt.savefig('8_aws_machines_experiment.pdf')

# df = df_group.std().reset_index()
#
# df = df.pivot(index='Number of nodes', columns='Mesh sizes', values='times_microseconds')
# print(df)
# df.plot(legend=True)
# plt.savefig('8_aws_machines_experiment_std.pdf')
#
# plt.show()
