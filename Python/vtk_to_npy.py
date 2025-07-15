import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os
import faulthandler

src_dir = '/home/erwan/Master/Stage/SlipFromSim/data/results_U0.24_D0.1_N1024_LEVEL9'     #
out_dir = '/home/erwan/Master/Stage/SlipFromSim/data/results_U0.24_D0.1_N1024_LEVEL9/numpy'                               # Where the numpy files will be saved

L0 = 1.72
LEVEL = 9
LEVEL_MIN = 4
N = 2**LEVEL

x_min = -0.5
x_max = x_min + L0

y_min = -L0/2
y_max = L0/2

x_offset = -0.5
y_offset = -L0/2

dx = L0/(2**LEVEL-1)

info = {
    "L0": L0,
    "LEVEL": LEVEL,
    "LEVEL_MIN": LEVEL_MIN,
    "N": N,
    "x_min": x_min,
    "x_max": x_max,
    "y_min": y_min,
    "y_max": y_max,
    "y_offset": y_offset,
    "x_offset": x_offset,
    "dx": dx
}

def vtu_to_numpy(vtu_data, info):
    xx, yy = np.meshgrid(
        np.linspace(info["x_min"] + info["dx"]/2, info["x_max"] - info["dx"]/2, info["N"]),
        np.linspace(info["y_min"] + info["dx"]/2, info["y_max"] - info["dx"]/2, info["N"]),
        indexing="ij"
    )

    zz = np.zeros_like(xx)

    # Stack into shape (N, N, 3) for PyVista
    grid = pv.StructuredGrid(xx, yy, zz)

    grid_data = grid.sample(vtu_data)

    data_mat = np.full((N, N, 5), np.nan)

    data_mat[:, :, 0] = grid_data.point_data["u.x"][:, 0].reshape(info["N"], info["N"])      # u_x
    data_mat[:, :, 1] = grid_data.point_data["u.x"][:, 1].reshape(info["N"], info["N"])      # u_y
    data_mat[:, :, 2] = grid_data.point_data["p"].reshape(info["N"], info["N"])              # pressure
    data_mat[:, :, 3] = grid_data.point_data["omega"].reshape(info["N"], info["N"])          # vorticity
    data_mat[:, :, 4] = np.sqrt(np.pow(data_mat[:, :, 0], 2) + np.pow(data_mat[:, :, 1],2))  # umag

    return data_mat


def get_files_in_order(src):
    files_temp = [int(f.split(".pvtu")[0].split("_")[-1]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    files_temp.sort()
    return [f"{src}/results_{n}.pvtu" for n in files_temp]


def save_files(data, out_dir, file):
    L0 = 1.72
    offx = 0.5
    lenx = len(data)
    lenf = int(2 * lenx / L0 * offx)

    out_file = ''.join(''.join(file.split('/')[-1]).split('.')[0:-1])+'.npy'
    out_path = f"{out_dir}/{out_file}"
    print(out_path)
    np.save(out_path, data)
    #np.save(out_path, data[int(lenx/2-lenf/2):int(lenx/2+lenf/2)][:lenf][:])


def main(info, src_dir, out_dir):
    print("GETTING FILES IN THE RIGHT ORDER")
    files = get_files_in_order(src_dir)
    print("STARTING ITERATING OVER THE FILES")
    l_files = len(files)
    for i,f in enumerate(files):
        print(i,"/",l_files)
        data = pv.read(f)
        data_np = vtu_to_numpy(data, info)
        save_files(data_np, out_dir, f)




if __name__=="__main__":
    main(info, src_dir, out_dir)
