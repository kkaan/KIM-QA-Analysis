"""
Overlay KIM trajectory and hexamotion trace with interactive time alignment sliders.
Per-axis time offsets allow independent temporal alignment of LR, SI, AP.

Usage:
    python plot_overlay.py <kim_trajectory_file> <hexamotion_trace_file> [x_max_seconds]
    python plot_overlay.py --auto <kim_trajectory_file> <hexamotion_trace_file> [x_max_seconds]
"""

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def load_kim(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    kim_time = df['Time (sec)'].values
    kim_axes = {
        'LR': df['Marker_0_LR'].values,
        'SI': df['Marker_0_SI'].values,
        'AP': df['Marker_0_AP'].values,
    }
    return kim_time, kim_axes


def load_hex(filepath):
    hex_data = np.loadtxt(filepath, skiprows=1)
    hex_indices = np.arange(len(hex_data))
    hex_time = hex_indices * 0.025
    hex_axes = {'LR': hex_data[:, 0], 'SI': hex_data[:, 1], 'AP': hex_data[:, 2]}
    return hex_time, hex_indices, hex_axes


def make_title(kim_filepath):
    parts = os.path.normpath(kim_filepath).split(os.sep)
    cdog_idx = next((i for i, p in enumerate(parts) if 'CDOG' in p), None)
    if cdog_idx is not None:
        return os.sep.join(parts[cdog_idx:])
    return os.sep.join(parts[-3:])


def find_best_offset(kim_time, kim_vals, hex_time, hex_vals, search_range=(-300, 300),
                     step=0.1):
    """Find time offset that minimises RMSE between interpolated hex and KIM data."""
    best_offset = 0
    best_rmse = np.inf
    for offset in np.arange(search_range[0], search_range[1], step):
        shifted = hex_time + offset
        # Interpolate hex onto KIM time points within the overlap
        mask = (kim_time >= shifted[0]) & (kim_time <= shifted[-1])
        if mask.sum() < 10:
            continue
        interp_vals = np.interp(kim_time[mask], shifted, hex_vals)
        rmse = np.sqrt(np.mean((kim_vals[mask] - interp_vals) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_offset = offset
    return best_offset, best_rmse


def plot_overlay(kim_filepath, hex_filepath, x_max=100):
    kim_time, kim_axes = load_kim(kim_filepath)
    hex_time, hex_indices, hex_axes = load_hex(hex_filepath)

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(bottom=0.27)
    fig.suptitle(make_title(kim_filepath), fontsize=14)

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
        ax.set_xlim(0, x_max)

    axs[0].legend(loc='upper right', fontsize=9)
    axs[-1].set_xlabel('Time (sec)')

    # Interval slider
    ax_dt = plt.axes([0.15, 0.18, 0.7, 0.02])
    s_dt = Slider(ax_dt, 'Hex interval (ms)', 10, 50, valinit=25, valstep=0.1, valfmt='%.1f')

    # Per-axis time offset sliders
    ax_lr_t = plt.axes([0.15, 0.13, 0.7, 0.02])
    ax_si_t = plt.axes([0.15, 0.08, 0.7, 0.02])
    ax_ap_t = plt.axes([0.15, 0.03, 0.7, 0.02])

    s_lr = Slider(ax_lr_t, 'LR time (s)', -300, 300, valinit=0, valstep=0.01, valfmt='%.2f')
    s_si = Slider(ax_si_t, 'SI time (s)', -300, 300, valinit=0, valstep=0.01, valfmt='%.2f')
    s_ap = Slider(ax_ap_t, 'AP time (s)', -300, 300, valinit=0, valstep=0.01, valfmt='%.2f')

    sliders = [s_lr, s_si, s_ap]

    def update(val):
        dt = s_dt.val / 1000.0
        scaled_time = hex_indices * dt
        for line, s in zip(hex_lines, sliders):
            line.set_xdata(scaled_time + s.val)
        fig.canvas.draw_idle()

    s_dt.on_changed(update)
    for s in sliders:
        s.on_changed(update)

    out_path = os.path.splitext(kim_filepath)[0] + '_overlay.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


def find_best_offset_combined(kim_time, kim_axes, hex_time, hex_axes,
                              search_range=(-300, 300), step=0.1):
    """Find single time offset that minimises combined RMSE across all axes."""
    labels = list(kim_axes.keys())
    best_offset = 0
    best_rmse = np.inf
    for offset in np.arange(search_range[0], search_range[1], step):
        shifted = hex_time + offset
        mask = (kim_time >= shifted[0]) & (kim_time <= shifted[-1])
        if mask.sum() < 10:
            continue
        total_se = 0
        count = 0
        for label in labels:
            interp_vals = np.interp(kim_time[mask], shifted, hex_axes[label])
            total_se += np.sum((kim_axes[label][mask] - interp_vals) ** 2)
            count += mask.sum()
        rmse = np.sqrt(total_se / count)
        if rmse < best_rmse:
            best_rmse = rmse
            best_offset = offset
    return best_offset, best_rmse


def plot_overlay_auto(kim_filepath, hex_filepath, x_max=100):
    """Auto-align hexamotion to KIM via combined RMSE minimisation, save plot (no GUI)."""
    kim_time, kim_axes = load_kim(kim_filepath)
    hex_time, hex_indices, hex_axes = load_hex(hex_filepath)

    # Find single best offset across all axes
    offset, rmse = find_best_offset_combined(kim_time, kim_axes, hex_time, hex_axes)
    print(f'Best offset: {offset:.2f} s (combined RMSE = {rmse:.3f} mm)')
    offsets = {label: offset for label in kim_axes}

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(make_title(kim_filepath), fontsize=14)

    kim_colors = sns.color_palette("deep", 3)

    for ax, (label, kim_vals), color in zip(axs, kim_axes.items(), kim_colors):
        sns.scatterplot(x=kim_time, y=kim_vals, ax=ax, color=color, s=15,
                        edgecolor='none', label='KIM')
        shifted_time = hex_time + offsets[label]
        ax.plot(shifted_time, hex_axes[label], color='red', linewidth=0.8,
                alpha=0.7, label=f'Hexamotion (t+{offsets[label]:.1f}s)')
        ax.set_ylabel(f'{label} (mm)')
        ax.set_ylim(-10, 10)
        ax.set_xlim(0, x_max)
        ax.legend(loc='upper right', fontsize=9)

    axs[-1].set_xlabel('Time (sec)')
    plt.tight_layout()

    out_path = os.path.splitext(kim_filepath)[0] + '_overlay_aligned.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    auto_mode = '--auto' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--auto']

    if len(args) < 2:
        print("Usage: python plot_overlay.py [--auto] <kim_trajectory_file> <hexamotion_trace_file> [x_max_seconds]")
        sys.exit(1)

    x_max = float(args[2]) if len(args) > 2 else 100

    if auto_mode:
        plot_overlay_auto(args[0], args[1], x_max)
    else:
        plot_overlay(args[0], args[1], x_max)
