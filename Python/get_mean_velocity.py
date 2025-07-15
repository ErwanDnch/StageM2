import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cycler import cycler

import os

#dx = 0.00671875  #Measured field
dx = 1.72/pow(2,9)
#linestyles = [
#    '-',          # solid
#    '--',         # dashed
#    ':',          # dotted
#    '-.',         # dash-dot
#    (0, (3, 1, 1, 1)),       # loosely dash-dot
#    (0, (5, 3, 1, 2, 1, 3)),
#    (0, (3, 10, 1, 10)),     # dash-dot with long spaces
#    (0, (1, 10)),            # dotted with long spaces
#
#]
#
#tol_bright = [
#    '#4477AA',  # bright orange
#    '#EE6677',  # bright blue
#    '#228833',  # bright cyan
#    '#CCBB44',  # teal
#    '#66CCEE',  # pink
#    '#AA3377',  # red
#    '#BBBBBB',  # light gray
#    '#000000',  # black
#]
#
#style_cycler = cycler(color=tol_bright) + cycler(linestyle=linestyles)
#
#plt.rcParams['axes.prop_cycle'] = style_cycler

def get_files_in_order(src):
    files_temp = [int(f.split(".npy")[0].split("_")[-1].split("s")[-1]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    files_temp.sort()
    return [f"results_{n}.npy" for n in files_temp]

def loop(file, mean_array, n_files, mean_vel, mass_error):
    vel_array = np.load(file)[:,:,:2]
    if mean_array is int:
        mean_array = np.zeros_like(vel_array)
    mean_array += vel_array/n_files
    mean_vel.append(mean_section(0, vel_array))
    mass_error.append(np.mean(np.abs(np.gradient(vel_array[:,:,0], dx, axis=1) + np.gradient(vel_array[:,:,1], dx, axis=0))))
    return mean_array, mean_vel, mass_error

def mean_section(section, array):
    mean = 0
    for i in array[:,section]:
        mean += i/np.shape(array)[0]
    return mean
cross_section = [0.0,0.1, 0.4, 0.6, 0.9]
cross_section = [int(x/dx) for x in cross_section]
if __name__=='__main__':
    src_directory = "/home/erwan/Master/Stage/SlipFromSim/data/results_U0.24_D0.1_N1024_LEVEL9"
    files = get_files_in_order(os.path.join(src_directory,"numpy"))

    n_files = len(files)
    mean_array = 0
    mean_vel = []
    mass_error = []
    for file in files:
        print(file)
        mean_array, mean_vel, mass_error = loop(os.path.join(src_directory, "numpy", file), mean_array, n_files, mean_vel, mass_error)

    mag_mean = np.zeros_like(mean_array[:,:,0])
    mag_mean = np.sqrt(mean_array[:,:,0]**2+mean_array[:,:,1]**2)

    last_array = np.load(os.path.join(src_directory, "numpy", files[-1]))[:,:,:2]
    first_array = np.load(os.path.join(src_directory, "numpy", files[0]))[:,:,:2]
    #print(files)
    nx, ny = np.shape(mag_mean)
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dx)

    fig = plt.figure(layout="constrained", figsize=(10,3))
    gs = GridSpec(1, 2, figure=fig)

    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])

    print(np.shape(X), np.shape(Y))
    ax0.contourf(X , Y, mag_mean)
    prop = 3
    ax0.quiver(X[::prop, ::prop], Y[::prop, ::prop], mean_array[::prop,::prop,0], mean_array[::prop,::prop,1], width = 0.001)
    ax0.set_title("Mean Velocity Field")
    ax0.set_aspect('equal')

    y = np.arange(ny) * dx
    for x in cross_section:
        vx_cross = mean_array[:, x, 0]
        vy_cross = mean_array[:, x, 1]
        vmag_cross = np.sqrt(vx_cross**2 + vy_cross**2)

        print(np.shape(vmag_cross))
        ax1.plot(vmag_cross,y, label=f"{x*dx}m")
        ax0.axvline(x * dx, linestyle='--', linewidth=0.8, alpha=0.8, zorder=5)
        #ax0.axvline(x, linewidth=0.5, alpha=0.8, zorder=0)


    vx_cross_last = last_array[:, 0, 0]
    vy_cross_last = last_array[:, 0, 1]
    vmag_cross_last = np.sqrt(vx_cross_last**2 + vy_cross_last**2)
    ax1.plot(vmag_cross_last,y, label=f"0.0m last time step")

    vx_cross_first = first_array[:, 0, 0]
    vy_cross_first = first_array[:, 0, 1]
    vmag_cross_first = np.sqrt(vx_cross_first**2 + vy_cross_first**2)
    ax1.plot(vmag_cross_first,y, label=f"0.0m first time step")

    ax1.legend()
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(src_directory,"mean_velocity.png"))

    plt.plot(mean_vel)
    plt.show()

    plt.plot(mass_error)
    plt.show()
