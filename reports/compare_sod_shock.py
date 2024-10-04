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
import sodshock
import matplotlib.pyplot as plt
import matplotlib
import tqdm

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


def get_sim_data(val_id, val_name):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(f"dumps/run_latest/{val_name}/res_{val_id}.vti")
    reader.Update()
    image = reader.GetOutput()

    rows, cols, _ = image.GetDimensions()
    sc = image.GetPointData().GetScalars()
    a = vtk_to_numpy(sc)
    a = a.reshape(rows)

    return a


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


def mape(y_pred, y_true):
    y_true = y_true + 1e-10
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



if __name__ == '__main__':

    gamma = 1.4
    dustFrac = 0.0
    npts = 200
    t = 0.22
    right_state = (1,1,0)
    left_state = (0.1, 0.125, 0.)

    positions, regions, values = sodshock.solve(left_state=left_state,
                                                right_state=right_state, geometry=(0., 1., 0.5), t=t,
                                                gamma=gamma, npts=npts, dustFrac=dustFrac)

    comp_array = values['rho']

    best_err = 99999
    best_id = -1

    for i in tqdm.trange(4998):
        val = get_sim_data(i, "rho")
        if rmse(val, comp_array) < best_err:
            best_id = i
            best_err = rmse(val, comp_array)

    fontsize = 20
    font = {'size'   : fontsize}
    matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)

    print("root mean square error rho = ", rmse(get_sim_data(best_id, "rho"), values['rho']))
    print("mean absolute percentage error rho = ", mape(get_sim_data(best_id, "rho"), values['rho']), "%")
    ax.plot(values['x'], values['rho'], linewidth=1.5, color='b', label="rho analytical")
    ax.plot(values['x'], get_sim_data(best_id, "rho"), linewidth=1.5, color='r', label="rho simulated")
    ax.set_xlabel("X space (m)")
    ax.set_ylabel("rho")
    ax.legend()
    fig.savefig('rho.pdf', dpi=300)
    ax.cla()

    print("root mean square error p = ", rmse(get_sim_data(best_id, "p"), values['p']))
    print("mean absolute percentage error p = ", mape(get_sim_data(best_id, "p"), values['p']), "%")
    ax.plot(values['x'], values['p'], linewidth=1.5, color='b', label="p analytical")
    ax.plot(values['x'], get_sim_data(best_id, "p"), linewidth=1.5, color='r', label="p simulated")
    ax.set_xlabel("X space (m)")
    ax.set_ylabel("p")
    ax.legend()
    fig.savefig('p.pdf', dpi=300)
    ax.cla()

    print("root mean square error u = ", rmse(get_sim_data(best_id, "u"), values['u']))
    print("mean absolute percentage error u (not valid here, many zeroes in y_pred) = ", mape(get_sim_data(best_id, "u"), values['u']), "%")
    ax.plot(values['x'], values['u'], linewidth=1.5, color='b', label="u analytical")
    ax.plot(values['x'], get_sim_data(best_id, "u"), linewidth=1.5, color='r', label="u simulated")
    ax.set_xlabel("X space (m)")
    ax.set_ylabel("u")
    ax.legend()
    fig.savefig('u.pdf', dpi=300)
    ax.cla()

