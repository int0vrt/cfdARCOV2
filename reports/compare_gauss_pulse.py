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
    a = a.reshape((rows, cols))

    return a[:, int(a.shape[1] / 2)]


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))


def mape(y_pred, y_true):
    # print(y_pred.shape)
    # print(y_true.shape)
    # y_true = y_true + 1e-10
    # y_pred = y_pred + 1e-10
    res = np.abs((y_true - y_pred) / y_true)
    res[0] = 0
    res[-1] = 0
    return np.mean(res) * 100


# # Analytical solution for a 2D standing wave
def analytical_sol(x, y, t, A=1.0, c=343, k_x=np.pi, k_y=np.pi):
    """
    Compute the analytical solution for a 2D standing wave.

    Parameters:
    x (numpy array): Array of x-coordinates.
    y (numpy array): Array of y-coordinates.
    t (float): Time at which the solution is evaluated.
    A (float): Amplitude of the standing wave (default 1.0).
    c (float): Speed of sound (default 343.0 m/s for air).
    k_x (float): Wavenumber in the x-direction (default pi).
    k_y (float): Wavenumber in the y-direction (default pi).

    Returns:
    numpy array: The pressure field at time t for the standing wave.
    """
    # Angular frequency for the standing wave
    omega = c * np.sqrt(k_x**2 + k_y**2)

    # Standing wave pattern
    return A * np.sin(k_x * x) * np.sin(k_y * y) * np.cos(omega * t)


if __name__ == '__main__':
    A = 1.0
    sigma = 0.1
    Lx = 1
    c = 0.3
    t = 1
    nx = 100

    x_vals = np.linspace(0, 1, 101)  # 200 points in the x-direction
    y_vals = np.linspace(0, 1, 101)  # 200 points in the y-direction
    X, Y = np.meshgrid(x_vals, y_vals)
    analytical_val = analytical_sol(X, Y, t)[:, 51]

    best_err = 99999
    best_id = -1

    for i in tqdm.trange(499):
        val = get_sim_data(i, "u")
        if mape(val, analytical_val) < best_err:
            best_id = i
            best_err = mape(val, analytical_val)


    fontsize = 20
    font = {'size'   : fontsize}
    matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)

    print("root mean square error p = ", rmse(get_sim_data(best_id, "u"), analytical_val))
    print("mean absolute percentage error p = ", mape(get_sim_data(best_id, "u"), analytical_val), "%")
    ax.plot(x_vals, analytical_val, linewidth=1.5, color='b', label="p analytical")
    ax.plot(x_vals, get_sim_data(best_id, "u"), linewidth=1.5, color='r', label="p simulated")
    ax.set_xlabel("X space (m)", fontsize=fontsize)
    ax.set_ylabel("p", fontsize=fontsize)
    ax.legend()
    fig.savefig('p_standing_wave.pdf', dpi=300)
    plt.show()
    ax.cla()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # Analytical solution for a 2D standing wave
# def standing_wave_solution(x, y, t, A=1.0, c=343, k_x=np.pi, k_y=np.pi):
#     """
#     Compute the analytical solution for a 2D standing wave.
#
#     Parameters:
#     x (numpy array): Array of x-coordinates.
#     y (numpy array): Array of y-coordinates.
#     t (float): Time at which the solution is evaluated.
#     A (float): Amplitude of the standing wave (default 1.0).
#     c (float): Speed of sound (default 343.0 m/s for air).
#     k_x (float): Wavenumber in the x-direction (default pi).
#     k_y (float): Wavenumber in the y-direction (default pi).
#
#     Returns:
#     numpy array: The pressure field at time t for the standing wave.
#     """
#     # Angular frequency for the standing wave
#     omega = c * np.sqrt(k_x**2 + k_y**2)
#
#     # Standing wave pattern
#     return A * np.sin(k_x * x) * np.sin(k_y * y) * np.cos(omega * t)
#
# # Define the spatial grid
# x_vals = np.linspace(0, 1, 100)  # 200 points in the x-direction
# y_vals = np.linspace(0, 1, 100)  # 200 points in the y-direction
# X, Y = np.meshgrid(x_vals, y_vals)
#
# # Time step and total number of frames
# t_max = 0.006  # 10 ms
# num_frames = 100  # Number of frames in the animation
# dt = t_max / num_frames  # Time increment for each frame
#
# # Set up the figure and axis
# fig, ax = plt.subplots()
# # cax = ax.contourf(X, Y, standing_wave_solution(X, Y, 0), levels=50, cmap='viridis')
# cax = ax.plot(x_vals, standing_wave_solution(X, Y, 0)[:, 50])
# # fig.colorbar(cax, label='Pressure')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_ylim([-1.1, 1.1])
# ax.set_title('2D Standing Wave')
#
# # Function to update the plot for each frame
# def update(frame):
#     t = frame * dt
#     ax.clear()  # Clear the current plot
#     pressure_field = standing_wave_solution(X, Y, t)
#     # cax = ax.contourf(X, Y, pressure_field, levels=50, cmap='viridis')
#     cax = ax.plot(x_vals, pressure_field[:, 50])
#     ax.set_ylim([-1.1, 1.1])
#     print(pressure_field[:, 50])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title(f'2D Standing Wave at t = {t:.4f} s')
#     return cax
#
# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, blit=False)
#
# # Save the animation as a video file (optional)
# # ani.save('standing_wave_animation.mp4', writer='ffmpeg', fps=30)
#
# # Display the animation
# plt.show()
