import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import pandas as pd
import time
import os
import multiprocessing

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

N = 2**9
H = 1.72
#H = 1.512
L0 = H
dx= L0/N       # Simulation => L0 / N, Video => 0.0028


dt = 0.05

rho_W = 1e3

pos_cyl = 0, 0   # Simulation => 0, videos => 533*dx, 268*dx
R = 0.05 # radius of the pier
mu = 0.5

offx = -0.5             # 0.63 from measurement, -0.5 for SImulation
#offy = -0,5
offy = -H/2                # 0.0 from measurement, -H/2 for Simulation

d = 0.006
rho_S = 582
L = 0.6
o = 0.15
Cd = 0.2


M = rho_S * np.pi / 4 * d**2 * L
Iomega = (1/12 + (o/L)**2)*M*L

phi0 = 0
om0 = 0


Loop = False     # False when multiple files
N_loops = 2000

upstream_position = -R - 0.01
downstream_position = R + 0.01

box_extent = (offx, offx+L0, offy, offy+L0)
#box_extent = (263, 603, 0, 540)

def sign(num):
    return -1 if num < 0 else 1


def get_closest_cell(posR, phi, coord, dx, L, offx=-0.5, offy=-H/2):            # R position of point R, phi angle, coord coordinate of the actual cell, dx length of a cell, L length of the stick
    """
    xi = coord[0] * dx + offx + dx/2 * sign(np.sin(phi))
    yi = coord[1] * dx - dx/2 * sign(np.cos(phi)) + L0/2
    sy = abs((posR[1]-yi)/np.cos(phi)) if np.cos(phi)!=0 else 1e10
    sx = abs((xi-posR[0])/np.sin(phi)) if np.sin(phi)!=0 else 1e10
"""
    xi = coord[0] * dx + dx/2 + offx
    yi = coord[1] * dx + dx/2 + offy
    sy = abs((posR[1] - (yi+dx*sign(np.cos(phi))))/np.cos(phi)) if np.cos(phi)!=0 else 1e10
    sx = abs((posR[0] - (xi+dx*sign(np.sin(phi))))/np.sin(phi)) if np.sin(phi)!=0 else 1e10

    if sx < 0 or sy < 0:
        print(f'xi: {xi}, x: {posR[0]}, phi: {phi}')
        print(f'yi: {yi}, y: {posR[1]}, phi: {phi}')
        print(f'sx: {sx}    sy: {sy}')
        raise ValueError('Negative Distance')
    elif sx > L and sy > L:
        return 0
    else:
        return (coord[0]+1*sign(np.sin(phi)), coord[1], sx) if sx < sy else (coord[0], coord[1]-1*sign(np.cos(phi)), sy)


def get_cell_first_point(R, dx, N, offx=-0.5, offy=-H/2, L0=1.72):
    return round(R[0]/dx-offx/L0*N), round(R[1]/dx-offy/L0*N) # Returns the cell indices of the point R ( adjusting for the offset of the simulation )


def get_pos_contact_point(R, phi, x_cyl, y_cyl):
    return x_cyl - R*np.cos(phi), y_cyl - R*np.sin(phi)


def get_stick_elements(parameters, phi):
    dx = parameters[7]
    N = parameters[8]
    L = parameters[3]
    pos_R, pos_PC, pos_L = get_points_position(phi, parameters)
    L0 = N*dx
    x0, y0 = get_cell_first_point(pos_R, dx, N, L0=L0, offx = parameters[15], offy=parameters[16])
    stick_elements = [[0, None, x0, y0]]

    ret = 1
    ite=0

    while ret != 0 and ite < N:
        ret = get_closest_cell(pos_R, phi, stick_elements[ite][-2:], dx, L, offx = parameters[15], offy = parameters[16])
        if ret != 0:
            stick_elements[ite][1] = ret[2]
            stick_elements.append([ret[2], None, ret[0], ret[1]])
        else:
            stick_elements[ite][1] = L
        ite+=1
    return stick_elements


def compute_forces(parameters, vel_array, phi, omega):
    rho, Cd, d, L, o, R, pos_cyl, dx, N, Iomega, dt, mu, M, up_pos, down_pos, offx, offy, box_extent = parameters
    stick_elements = get_stick_elements(parameters, phi)

    f_er = 0
    f_phi = 0
    f_er_iner = 0
    f_er_wo_iner = 0
    T = 0
    T_l = 0
    T_r = 0

    array_lenx = len(vel_array[:,0,0])
    array_leny = len(vel_array[0,:,0])
    print(array_lenx, array_leny)
    for iteration, element in enumerate(stick_elements):
        s_p, s_m = element[1], element[0]
        ds = s_p - s_m
        xs, ys = element[2], element[3]
        if xs < 0 or ys < 0 or xs>array_lenx or ys>array_leny:
            raise IndexError(f"Stick out of the Domain: xs = {xs}, ys = {ys}\nAngle: {phi/(2*np.pi)*360}°")
        Ux = vel_array[element[2], element[3], 0]
        Uy = vel_array[element[2], element[3], 1]
        xn = get_xn(s_p, s_m, phi, L, o, R)
        A = 0.5 * rho * Cd * d

        f_er_inf, f_er_iner_inf, f_er_wo_iner_inf = get_f_er_inf(A, phi, omega, xn, Ux, Uy)
        f_er -= f_er_inf * ds
        f_er_iner -= f_er_iner_inf * ds
        f_er_wo_iner -= f_er_wo_iner_inf * ds
        if xn > 0:
            T_l -= xn * f_er_inf * ds
        else:
            T_r -= xn * f_er_inf * ds
        T -= xn * f_er_inf * ds

        f_phi += get_f_phi(A, phi, Ux, Uy, iteration, element, d, rho, len(stick_elements), ds)


    return f_er, f_phi, T_l, T_r, f_er_iner, f_er_wo_iner


def get_f_phi(A, phi, Ux, Uy, iteration, element, d, rho, l_elements, ds):
    U_par = (Ux + Uy) * np.cos(phi) * np.sin(phi)
    f_phi_inf = A * U_par**2 * ds

    if iteration==0 and phi>0:
        f_phi_inf += 1.12 * d**2/4 * np.pi * rho * (Ux * np.sin(phi) + Uy * np.cos(phi))**2
    elif iteration==l_elements-1 and phi<0:
        f_phi_inf += 1.12 * d**2/4 * np.pi * rho * (Ux * np.sin(phi) + Uy * np.cos(phi))**2
    return f_phi_inf


def get_xn(s_p, s_m, phi, L, o, R):
    return (s_p + s_m)/2 - L/2 + o - R*phi


def get_f_er_inf(A, phi, omega, xn, Ux, Uy):
    U_perp = Ux*np.cos(phi) + Uy*np.sin(phi) - xn*omega
    U_stat = Ux*np.cos(phi) + Uy*np.sin(phi)
    f_er_inf = -A * U_perp**2 * sign(U_perp)
    f_er_iner_inf = abs(A * ((xn*omega)**2-2*xn*omega*(Ux*np.cos(phi)+Uy*np.sin(phi))))
    #f_er_wo_iner = abs(A * (Ux*np.cos(phi) + Uy*np.sin(phi))**2)
    f_er_wo_iner = -A * U_stat**2 * sign(U_stat)
    return f_er_inf, f_er_iner_inf, f_er_wo_iner


def get_domega(T, I_omega):
    return T/I_omega


def update_angle(phi, omega, domega, dt):
    omega += domega * dt
    phi += omega * dt
    return phi, omega


def time_loop(history, parameters, file, phi, omega):
    dt = parameters[10]
    mu = parameters[11]
    M = parameters[12]
    slid = history["slid"][-1]
    vel_array = np.load(file)[:,:,:2]

    I_omega = M*L**2*(1/12+((R*phi-parameters[4]*L0)/L)**2)


    f_er, f_phi, T_l, T_r, f_er_iner, f_er_wo_iner = compute_forces(parameters, vel_array, phi, omega)
    T = T_l + T_r
    if f_phi > mu * f_er and not slid:
        slid = 1
    domega = get_domega(T, I_omega)
    phi, omega = update_angle(phi, omega, domega, dt)

    history['f_er'].append(f_er)
    history['f_phi'].append(f_phi)
    history['T_l'].append(T_l)
    history['min_T_r'].append(-T_r)
    history['T'].append(T)
    history["phi"].append(phi)
    history["omega"].append(omega)
    history["domega"].append(domega)
    history["I_omega"].append(I_omega)
    history["slid"].append(slid)
    history["f_er_iner"].append(f_er_iner)
    history["f_er_wo_iner"].append(f_er_wo_iner)

    return history, phi, omega


def draw_plots(history, parameters, animation=False):

    from cycler import cycler

    colorblind_colors = ['#0072B2', '#D55E00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#E69F00']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colorblind_colors)

    if animation:
        box_extent = parameters[17]
        print("BOX EXTENT LENGTH: ", len(box_extent))
        fig = plt.figure(layout="constrained", figsize=(15,10))

        gs = GridSpec(3, 2, figure=fig)

        # fig, axes = plt.subplots(n_plots, 1, figsize=(10, 10), gridspec_kw={"height_ratios": height_ratios})

        ax0 = fig.add_subplot(gs[0,0])
        history["muFn"] = [mu*i for _, i in enumerate(history["f_er"])]

        print("muFn shape:", np.shape(history["muFn"]))
        print(len(history["t"]))
        print("type of t:", type(history["t"]))
        print("t shape:", np.shape(history["t"]))

        ax0.plot(history["t"], history["muFn"], label='$\mu f_{er}$')
        ax0.plot(history["t"], history["f_phi"], label='$f_{\phi}$')
        ax0.legend()
        ax0.set_title("Time Evolution of Forces")
        ax0.set_ylabel("Force (N)")
        ax0.set_xlabel("Time  (s)")

        ax1 = fig.add_subplot(gs[1,0])
        ax1.plot(history['t'], history['phi'])
        ax1.set_title("Time evolution of the angle $\phi$")
        ax1.set_ylabel("$\phi$ (rad)")
        ax1.set_xlabel("Time (s)")

        ax2 = fig.add_subplot(gs[2,0]) if animation else fig.add_subplot(gs[0,1])
        ax2.plot(history['t'], history['omega'])
        ax2.set_title("Time evolution of the angular velocity $\omega$")
        ax2.set_ylabel("$\omega$ (rad.$s^{-1})$")
        ax2.set_xlabel("Time (s)")

        history["muDiff"] = [mu*i-j for _, (i,j) in enumerate(zip(history["f_er"],history["f_phi"]))]
        ax3 = fig.add_subplot(gs[2,1]) if animation else fig.add_subplot(gs[1,1])
        ax3.plot(history["t"], history["muDiff"])
        ax3.plot(history["t"], [0 for i in history["t"]], "k--")
        ax3.set_title("$\mu F_{er}$-$F_{\phi}$")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Difference (N)")

        ax_anim = fig.add_subplot(gs[:2,1])

        ax_anim.set_xlim(box_extent[0], box_extent[1])
        ax_anim.set_ylim(box_extent[2], box_extent[3])
        ax_anim.set_title("Motion of Points Over Time")
        ax_anim.set_aspect('equal')
        ax_anim.set_ylabel("y (m)")
        ax_anim.set_xlabel("x (m)")

        line_rl, = ax_anim.plot([], [], 'k-')

        point_r, = ax_anim.plot([], [], 'ro', label="Right Point")
        point_pc, = ax_anim.plot([], [], 'mo', label="Contact POint")
        point_l, = ax_anim.plot([], [], 'bo', label="Left Point")
        image = ax_anim.imshow(plt.imread(history["image"][0]), extent=box_extent , animated=True)


        t_text = ax_anim.text(box_extent[0]+0.1,box_extent[2]+0.15, "", fontsize=12, color="black", ha="left", va="top")
        phi_text = ax_anim.text(box_extent[0]+0.1, box_extent[2]+0.1, "", fontsize=12, color="black", ha="left", va="top")
        slid_text = ax_anim.text(box_extent[1]-0.5, box_extent[2]+0.3, "", fontsize=20, color="red", ha="center", va="top")
        slid_text.set_fontweight('bold')

        moving_point01, = ax0.plot([], [], 'ro', markersize=8)
        moving_point02, = ax0.plot([], [], 'ro', markersize=8)

        moving_point1, = ax1.plot([], [], 'ro', markersize=8)
        moving_point2, = ax2.plot([], [], 'ro', markersize=8)

        moving_point3, = ax3.plot([],[], 'ro', markersize=8)

        cylinder = plt.Circle(parameters[6], parameters[5], color='grey')
        ax_anim.add_patch(cylinder)


        def update(frame):
            phi = history["phi"][frame]
            t = history["t"][frame]
            slid = history["slid"][frame]
            (x_r, y_r), (x_pc, y_pc), (x_l, y_l) = get_points_position(phi, parameters)

            line_rl.set_data([x_r, x_l], [y_r, y_l])

            moving_point01.set_data([history["t"][frame]], [history["muFn"][frame]])
            moving_point02.set_data([history["t"][frame]], [history["f_phi"][frame]])
            moving_point1.set_data([history["t"][frame]], [history["phi"][frame]])
            moving_point2.set_data([history["t"][frame]], [history["omega"][frame]])
            moving_point3.set_data([history["t"][frame]], [history["muDiff"][frame]])

            point_r.set_data([x_r], [y_r])
            print("Point R: ",[x_r], [y_r])
            point_pc.set_data([x_pc], [y_pc])
            print("Contact Point: ", [x_pc], [y_pc])
            point_l.set_data([x_l], [y_l])
            print("Point L: ", [x_l], [y_l])
            print(history["image"][frame], frame)
            image.set_data(plt.imread(history["image"][frame]))
            image.set_extent(box_extent)


            t_text.set_text(f"t = {t:.2f}s")
            phi_deg = (phi/(2*np.pi)*360)
            phi_text.set_text(f"$\phi$ = {phi_deg:.2f}°")
            slid_text.set_text(f'{"SLID OFF" if slid else ""}')

            return line_rl, point_r, point_pc, point_l, phi_text, slid_text, t_text, moving_point01, moving_point02, moving_point1, moving_point2, image, moving_point3


        ax_anim.legend()
        plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=len(history["t"]), interval=50, blit=True)
        ani.save(f"{stick_dir}/videos/animation_mu{mu}_o{parameters[4]}_{len(history["T"])*parameters[10]:.2f}s.mp4", writer="ffmpeg", fps=20)
    else:
        fig = plt.figure(layout="constrained", figsize=(15,5))

        gs = GridSpec(1, 2, figure=fig)

        ax0 = fig.add_subplot(gs[0,0])
        history["muFn"] = [mu*i for _, i in enumerate(history["f_er"])]
        ax0.plot(history["t"], history["muFn"], label=r'$\mu f_{er}$', ls='-')
        ax0.plot(history["t"], history["f_phi"], label=r'$f_{\phi}$', ls='-.')
        ax0.legend()
        ax0.set_title("Time Evolution of Forces")
        ax0.set_ylabel("Force (N)")
        ax0.set_xlabel("Time  (s)")

        ax1 = fig.add_subplot(gs[0,1])
        ax1.plot(history['t'], history['phi'])
        ax1.set_title(r"Time evolution of the angle $\phi$")
        ax1.set_ylabel(r"$\phi$ (rad)")
        ax1.set_xlabel("Time (s)")

        for ax in fig.axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman']})

        plt.savefig("debug_figure.pdf", bbox_inches='tight')


def get_points_position(phi, parameters):
    x_cyl, y_cyl = parameters[6]
    print(parameters[6])
    o = parameters[4]
    R = parameters[5]
    L = parameters[3]

    x_pc = x_cyl - R*np.cos(phi)
    y_pc = y_cyl - R*np.sin(phi)

    L_r = L/2-o+R*phi
    x_r = x_pc - L_r * np.sin(phi)
    y_r = y_pc + L_r * np.cos(phi)

    x_l = x_r + L * np.sin(phi)
    y_l = y_r - L * np.cos(phi)

    return ((x_r,y_r), (x_pc,y_pc), (x_l,y_l))


def get_files_in_order(src):
    files_temp = [int(f.split(".npy")[0].split("_")[-1].split("s")[-1]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    files_temp.sort()
    return [f"results_{n}.npy" for n in files_temp]


def get_images_in_order(src):
    files_temp = [int(f.split(".png")[0].split("_")[-1]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    files_temp.sort()
    files = [os.path.join(src,f"results_{n}.png") for n in files_temp]
    return files


def save_data(init, parameters, history, filename, N_loops, src_directory):

    history_copy = history.copy()

    length = len(history['t'])


    for key in history:
        history_copy[key] = history[key][:length]

    #for i in history_copy.keys():
    #    print(i, len(history_copy[i]))
    df = pd.DataFrame(history_copy)
    df.to_csv(filename, index=False)
    print(filename)


def get_image_history(image_path, N_loops):
    im_list = []
    for i in range(N_loops):
        im_list.append(image_path)
    return im_list


def main(src_directory, parameters, init, save=True, anim=False, Loop=False, N_loops=10):

    version = "_".join(src_directory.split("/")[-1].split("_")[1:])
    csv_dir = os.path.join("/home/erwan/Master/Stage/SlipFromSim/csv_data", version)

    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    #filename = f"{csv_dir}/Stick{parameters[3]}/data_o{parameters[4]}_phi{init[0]}_om{init[0]}.csv"
    filename = os.path.join(csv_dir, f"Stick{parameters[3]}/data_o{parameters[4]}_phi{init[0]}_om{init[0]}.csv")

    stick_dir = os.path.join(csv_dir, f"Stick{parameters[3]}")
    if not os.path.isdir(stick_dir):
            os.makedirs(stick_dir, exist_ok=True)

    if os.path.isfile(filename) and not save:
        df = pd.read_csv(filename)
        history = df.to_dict(orient="list")

        if Loop:
            image_path = get_images_in_order(os.path.join(src_directory,"images"))[0]
            history["image"] = get_image_history(image_path, N_loops)

        draw_plots(history, parameters, animation=anim)
    else:
        history={'f_er':[],
                 'f_phi':[],
                 'T':[],
                 'phi':[init[0]],
                 'omega':[init[1]],
                 'domega':[],
                 't':[0],
                 'slid':[0],
                 'image':get_images_in_order(os.path.join(src_directory,"images")),
                 'T_l':[],
                 'min_T_r':[],
                 'f_er_iner':[],
                 'f_er_wo_iner':[],
                 'I_omega':[],
        }
        files = get_files_in_order(os.path.join(src_directory,"numpy"))

        print("*****************************")
        t = 0
        dt = parameters[10]

        if Loop:
            for i in range(N_loops):
                for file in files:
                    print('*****************************************************')
                    print(f'      T   =   {t}')
                    print('*****************************************************')
                    history["t"].append(t)
                    phi = history["phi"][-1]
                    omega = history["omega"][-1]
                    history, phi, omega = time_loop(history, parameters, os.path.join(src_directory, "numpy", file), phi, omega)

                    t += dt

            image_path = get_images_in_order(os.path.join(src_directory,"images"))[0]
            history["image"] = get_image_history(image_path, N_loops)

        else:
            for file in files:
                print('*****************************************************')
                print(f'      T   =   {t}')
                print('*****************************************************')
                history["t"].append(t)
                phi = history["phi"][-1]
                omega = history["omega"][-1]
                history, phi, omega = time_loop(history, parameters, os.path.join(src_directory, "numpy", file), phi, omega)

                t += dt
        history["phi"] = history["phi"][:-1]
        history["t"] = history["t"][:-1]
        history["omega"] = history["omega"][:-1]

        if save:
            save_data(init, parameters, history, filename, N_loops, src_directory)
        else:
            draw_plots(history, parameters, animation=anim)

    return 0

o_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
stick_length = [0.3, 0.4, 0.5]
#o_values = [0.0]


if __name__ == '__main__':
    init = [phi0,om0]          # phi, omega
    src_directory = "/home/erwan/Master/Stage/SlipFromSim/data/results_U0.24_D0.1_N1024_LEVEL9"

    # rho:0, Cd:1, d:2, L:3, o:4, R:5, pos_cyl:6, dx:7, N:8, Iomega:9, dt:10, mu:11, M:12, up_pos:13, down_pos:14, offx: 15, offy: 16, box_extent: 17
    parameters = [
        rho_W, Cd, d, L, 0, R, pos_cyl, dx, N, Iomega, dt, mu, M, upstream_position, downstream_position, offx, offy, box_extent
    ]
    for stick in stick_length:
        stick_dir = src_directory + "/Stick_length_" + str(stick)
        if not os.path.isdir(stick_dir):
            os.makedirs(stick_dir, exist_ok=True)
        if not os.path.isdir(stick_dir+"/videos"):
            os.makedirs(stick_dir+"/videos", exist_ok=True)
        parameters[3] = stick
        for value in o_values:
            parameters[4] = value
            main(src_directory, parameters, init, save=True, anim=False, Loop=Loop, N_loops=N_loops)
