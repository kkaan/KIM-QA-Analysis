"""
Overlay KIM trajectory and hexamotion trace with interactive time alignment slider.

Usage:
    python plot_overlay.py <kim_trajectory_file> <hexamotion_trace_file>
"""

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_overlay(kim_filepath, hex_filepath):
    # Load KIM trajectory
    df = pd.read_csv(kim_filepath)
    df.columns = df.columns.str.strip()
    kim_time = df['Time (sec)'].values
    kim_axes = {
        'LR': df['Marker_0_LR'].values,
        'SI': df['Marker_0_SI'].values,
        'AP': df['Marker_0_AP'].values,
    }

    # Load hexamotion trace
    hex_data = np.loadtxt(hex_filepath, skiprows=1)
    hex_time = np.arange(len(hex_data)) * 0.025
    hex_axes = {'LR': hex_data[:, 0], 'SI': hex_data[:, 1], 'AP': hex_data[:, 2]}

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.subplots_adjust(bottom=0.15)

    # Title from CDOG path
    parts = os.path.normpath(kim_filepath).split(os.sep)
    cdog_idx = next((i for i, p in enumerate(parts) if 'CDOG' in p), None)
    if cdog_idx is not None:
        title = os.sep.join(parts[cdog_idx:])
    else:
        title = os.sep.join(parts[-3:])
    fig.suptitle(title, fontsize=14)

    kim_colors = sns.color_palette("deep", 3)
    hex_lines = []

    for ax, (label, kim_vals), color in zip(axs, kim_axes.items(), kim_colors):
        sns.scatterplot(x=kim_time, y=kim_vals, ax=ax, color=color, s=15,
                        edgecolor='none', label='KIM')
        line, = ax.plot(hex_time, hex_axes[label], color='red', linewidth=0.8,
                        alpha=0.7, label='Hexamotion')
        hex_lines.append(line)
        ax.set_ylabel(f'{label} (mm)')
        ax.set_ylim(-10, 10)

    axs[0].legend(loc='upper right', fontsize=9)
    axs[-1].set_xlabel('Time (sec)')

    # Time offset slider
    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.025])
    slider = Slider(ax_slider, 'Time offset (s)', -300, 300, valinit=0, valstep=0.01,
                    valfmt='%.2f')

    def update(val):
        t_offset = slider.val
        for line, label in zip(hex_lines, hex_axes):
            line.set_xdata(hex_time + t_offset)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save PNG
    out_path = os.path.splitext(kim_filepath)[0] + '_overlay.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python plot_overlay.py <kim_trajectory_file> <hexamotion_trace_file>")
        sys.exit(1)
    plot_overlay(sys.argv[1], sys.argv[2])
