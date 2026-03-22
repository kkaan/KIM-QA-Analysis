"""
Hexamotion trace plot tool.
Plots X, Y, Z position over time from tab-separated trace files.
Each row is a 25 ms increment.

Usage:
    python plot_hexamotion_trace.py <trace_file>
"""

import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_hexamotion_trace(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    dt = 0.025  # 25 ms per row
    time = np.arange(len(data)) * dt

    labels = ['X', 'Y', 'Z']
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    parts = os.path.normpath(filepath).split(os.sep)
    title = os.sep.join(parts[-2:])
    fig.suptitle(title, fontsize=14)

    colors = sns.color_palette("deep", 3)

    for ax, col, label, color in zip(axs, range(3), labels, colors):
        ax.plot(time, data[:, col], color=color, linewidth=0.5)
        ax.set_ylabel(f'{label} (mm)')

    axs[-1].set_xlabel('Time (sec)')

    plt.tight_layout()

    out_path = os.path.splitext(filepath)[0] + '_plot.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_hexamotion_trace.py <trace_file>")
        sys.exit(1)
    plot_hexamotion_trace(sys.argv[1])
