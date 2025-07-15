import numpy as np
import matplotlib.pyplot as plt
import os

def get_files_in_order(src):
    files_temp = [int(f.split(".npy")[0].split("_")[-1].split("s")[-1]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    files_temp.sort()
    return [f"results_{n}.npy" for n in files_temp]

def loop(file, mean_array, n_files):
    vel_array = np.load(file)[:,:,:2]
    if mean_array is int:
        mean_array = np.zeros_like(vel_array)
    mean_array += vel_array/n_files
    return mean_array



if __name__=='__main__':
    src_directory = "/home/erwan/Master/Stage/SlipFromSim/data/results_U0p24_D0p1_L9"
    files = get_files_in_order(os.path.join(src_directory,"numpy"))

    n_files = len(files)
    mean_array = 0
    for file in files:
        print(file)
        mean_array = loop(os.path.join(src_directory, "numpy", file), mean_array, n_files)

    mag_mean = np.zeros_like(mean_array[:,:,0])
    mag_mean = np.sqrt(mean_array[:,:,0]**2+mean_array[:,:,1]**2)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.contourf(mag_mean)
    nx, ny = np.shape(mag_mean)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    prop = 3
    ax.quiver(X[::prop, ::prop], Y[::prop, ::prop], mean_array[::prop,::prop,0], mean_array[::prop,::prop,1], width = 0.001)
    ax.set_aspect('equal')
    ax.set_title("Mean Velocity Field")
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(src_directory,"mean_velocity.png"))

    plt.plot(mag_mean[:,145], X[:,0])
    plt.show()
