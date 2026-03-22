"""
Single-marker trajectory scatter plot tool.
Plots Marker_0 LR, SI, AP over time with gantry angle on hover.

Usage:
    python plot_single_marker.py <trajectory_file>
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_trajectory(filepath):
    """Load and return trajectory DataFrame with cleaned column names."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def plot_single_marker(filepath):
    df = load_trajectory(filepath)

    time = df['Time (sec)'].values
    gantry = df['Gantry'].values
    filename = df['Filename (of base frame in average framestack)'].values
    axes_data = {
        'LR': df['Marker_0_LR'].values,
        'SI': df['Marker_0_SI'].values,
        'AP': df['Marker_0_AP'].values,
    }

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Single Marker Trajectory', fontsize=14)

    colors = sns.color_palette("deep", 3)
    scatter_artists = []

    for ax, (label, values), color in zip(axs, axes_data.items(), colors):
        sns.scatterplot(x=time, y=values, ax=ax, color=color, s=15, edgecolor='none')
        ax.set_ylabel(f'{label} (mm)')
        ax.set_ylim(-5, 5)  # ±5 mm
        scatter_artists.append(ax)

    axs[-1].set_xlabel('Time (sec)')

    # Per-axis hover annotations
    annots = {}
    for ax in axs:
        a = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.9),
            fontsize=9,
            visible=False,
        )
        annots[ax] = a

    def on_hover(event):
        for ax, a in annots.items():
            if event.inaxes != ax:
                if a.get_visible():
                    a.set_visible(False)

        if event.inaxes not in annots:
            fig.canvas.draw_idle()
            return

        idx = np.argmin(np.abs(time - event.xdata))
        t_val = time[idx]
        g_val = gantry[idx]

        for ax, (label, values) in zip(axs, axes_data.items()):
            if event.inaxes == ax:
                v_val = values[idx]
                fname = filename[idx]
                text = f'Time: {t_val:.2f}s\n{label}: {v_val:.2f} mm\nGantry: {g_val:.1f}\u00b0\n{fname}'
                annots[ax].xy = (t_val, v_val)
                annots[ax].set_text(text)
                annots[ax].set_visible(True)
                break

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_single_marker.py <trajectory_file>")
        sys.exit(1)
    plot_single_marker(sys.argv[1])
