from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import pandas as pd
import os

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
matplotlib.rc('text', usetex=True)

plt.rcParams.update({
    "font.size": 10,          # general font size
    "axes.labelsize": 10,     # axis labels
    "xtick.labelsize": 9,    # tick labels
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.titlesize": 12,     # title
})

tol_muted = [
    '#332288',  # dark blue
    '#88CCEE',  # light blue
    '#44AA99',  # teal
    '#117733',  # green
    '#999933',  # mustard
    '#DDCC77',  # sand
    '#CC6677',  # rose
    '#882255',  # wine
]

tol_bright = [
    '#4477AA',  # bright orange
    '#EE6677',  # bright blue
    '#228833',  # bright cyan
    '#CCBB44',  # teal
    '#66CCEE',  # pink
    '#AA3377',  # red
    '#BBBBBB',  # light gray
    '#000000',  # black
]

tol_vibrant = [
    '#0077BB',
    '#33BBEE',
    '#009988',
    '#EE7733',
    '#CC3311',
    '#EE3377',
    '#BBBBBB',
    '#000000',  # black
]

linestyles = [
    '-',          # solid
    '--',         # dashed
    ':',          # dotted
    '-.',         # dash-dot
    (0, (3, 1, 1, 1)),       # loosely dash-dot
    (0, (5, 3, 1, 2, 1, 3)),
    (0, (3, 10, 1, 10)),     # dash-dot with long spaces
    (0, (1, 10)),            # dotted with long spaces

]

marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'x']

# Match lengths (cycler repeats if shorter)
style_cycler = cycler(color=tol_bright) + cycler(linestyle=linestyles)

# Apply to matplotlib
plt.rcParams['axes.prop_cycle'] = style_cycler


#sliding_angles = []
#filenames = [f'MeasuredField/csv_data/data_o{x}_phi{phi0}_om{om0}.csv' for x in ecc]
#dfs = [pd.read_csv(f) for f in filenames]


def find_root(muFn, Fp, angle):
    diff = [list(muFn)[i]-list(Fp)[i] for i in range(len(list(muFn)))]
    for i, res in enumerate(diff):
        if res < 0:
            up_index = i
            bot_index = i-1
            return (angle[up_index] + angle[bot_index]) / 2


def main(dfs, out_dir):
    mu_list = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    ecc = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]

    phi0 = 0
    om0 = 0
    L0 = 0.6

    sliding_angles = []

    for mu in mu_list:
        slip_angles = []
        max_fig2 = 0

        fig = plt.figure(layout="constrained", figsize=(10,3))

        gs = GridSpec(1, 3, figure=fig)

        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[0,2])

        for n,df in enumerate(dfs):
            df["muFn"] = [mu*i for _, i in enumerate(df["f_er"])]
            val = find_root(df["muFn"], df["f_phi"], df["phi"])
            if val != None:
                slip_angles.append(val)

            df["muFn_wo_in"] = [mu*i for _, i in enumerate(df["f_er_wo_iner"])]
            ax1.plot(df["phi"], df["muFn"]-df["f_phi"], label=f'e = {ecc[n]/L0:.2f}')
            ax0.plot(df['t'][:400], df['phi'][:400], label=f'e = {ecc[n]/L0:.2f}')

            ax2.plot(df["phi"], df["f_er_wo_iner"]/df["f_er"])

            rap_listes = [list(df["f_er_wo_iner"])[i]/list(df["f_er"])[i] for i in range(len(list(df["f_er"])))]
            if max(rap_listes) > max_fig2:
                max_fig2 = max(rap_listes)

        max_index = slip_angles.index(max(slip_angles))
        slip_angles.pop(max_index)
        min_index = slip_angles.index(min(slip_angles))
        slip_angles.pop(min_index)
        mean_angle = np.mean(slip_angles)
        sliding_angles.append(mean_angle)

        ax0.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.8, zorder=0)
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.8, zorder=0)
        ax1.axvline(mean_angle, color='red', linestyle='-', linewidth=0.5, alpha=0.8, zorder=0)
        ax2.axvline(mean_angle, color='red', linestyle='-', linewidth=0.5, alpha=0.8, zorder=0)

        ax1.legend()
        #ax0.set_title("Time Evolution of the Difference")
        ax1.set_ylabel(" $F_R \mu - F_{\phi}$(N)")
        ax1.set_xlabel("Angle (rad)")
        xmin, _ = ax1.get_xlim()
        ax1.set_xlim(xmin, 1 if mean_angle < 0.8 else (mean_angle+0.2 if not np.isnan(mean_angle) else 1.5))
        ax1.minorticks_on()
        ax1.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.8)
        ax1.grid(True, which='major', linestyle='-.', linewidth=0.3, alpha=0.8)

        ax2.set_ylabel("$F_Rstatic / F_Rdyn$")
        ax2.set_xlabel("Angle (rad)")
        xmin, _ = ax2.get_xlim()
        ax2.set_xlim(xmin, 1 if mean_angle < 0.8 else (mean_angle+0.2 if not np.isnan(mean_angle) else 1.5))
        ax2.set_ylim(0.8,2)
        ax2.minorticks_on()
        ax2.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.8)
        ax2.grid(True, which='major', linestyle='-.', linewidth=0.3, alpha=0.8)


        #ax1.set_title("Time evolution of the angle $\phi$")
        ax0.set_ylabel("$\phi$ (rad)")
        ax0.set_xlabel("Time (s)")
        #ax0.legend()
        ax0.minorticks_on()
        ax0.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.8)
        ax0.grid(True, which='major', linestyle='-.', linewidth=0.3, alpha=0.8)

        for ax in fig.axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax0.text(0.99, 1.05, '(a)', transform=ax0.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right')
        ax1.text(0.99, 1.05, '(b)', transform=ax1.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right')
        ax2.text(0.99, 1.05, '(c)', transform=ax2.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right')

        ax1.text(mean_angle, 0, f'{mean_angle:.2f}', color='red',
                ha='left', va='bottom', fontsize=9,
                transform=ax1.get_xaxis_transform())

        ax2.text(mean_angle, 0, f'{mean_angle:.2f}', color='red',
                ha='left', va='bottom', fontsize=9,
                transform=ax2.get_xaxis_transform())

        name_save = f'plots_mu{mu}_D0.1_Measured.pdf'
        plt.savefig(os.path.join(out_dir, name_save), bbox_inches='tight')
        name_save = f'plots_mu{mu}_D0.1_Measured.png'
        plt.savefig(os.path.join(out_dir, name_save), bbox_inches='tight', dpi=500)


    plt.figure(figsize=(10,3))
    plt.plot(mu_list, sliding_angles)

    ax = plt.gca()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend()
    plt.ylabel("Angle (rad)")
    plt.xlabel("Friction Parameter")
    plt.xlim(min(sliding_angles)-0.1, max(sliding_angles)+0.1)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.8)
    plt.grid(True, which='major', linestyle='-.', linewidth=0.3, alpha=0.8)

    sliding_angles_path = os.path.join(out_dir, "sliding_angles.png")
    plt.savefig(sliding_angles_path)


source_directory = "/home/erwan/Master/Stage/SlipFromSim/csv_data/U0.24_D0.1_N1024_LEVEL9"
output_directory = "/home/erwan/Master/Stage/SlipFromSim/Figures/U0p24_D0p1_LEVEL9"

if not os.path.isdir(output_directory):
            os.makedirs(output_directory, exist_ok=True)

for subdir, dirs, files in os.walk(source_directory):
    print(subdir)
    if subdir == source_directory:
        continue
    out_dir = os.path.join(output_directory, os.path.basename(subdir))
    if not os.path.isdir(os.path.join(out_dir)):
        os.makedirs(out_dir, exist_ok=True)
    print(files)
    print(out_dir)
    dfs = [pd.read_csv(os.path.join(subdir,f)) for f in files]

    main(dfs, out_dir)
