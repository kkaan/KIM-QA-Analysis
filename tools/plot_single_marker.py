"""
Single-marker trajectory scatter plot tool.
Plots Marker_0 LR, SI, AP over time with gantry angle on hover.

Usage:
    python plot_single_marker.py <trajectory_file>
"""

import sys
import os
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
    parts = os.path.normpath(filepath).split(os.sep)
    # Show path from the CDOG session directory onward
    cdog_idx = next((i for i, p in enumerate(parts) if 'CDOG' in p), None)
    if cdog_idx is not None:
        title = os.sep.join(parts[cdog_idx:])
    else:
        title = os.sep.join(parts[-3:])
    fig.suptitle(title, fontsize=14)

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

    # On-click: show debug image for nearest data point
    # Look for debugimg in the trajectory folder first, then in sibling directories
    debugimg_dir = os.path.join(os.path.dirname(filepath), 'debugimg')
    if not os.path.isdir(debugimg_dir):
        parent = os.path.dirname(os.path.dirname(filepath))
        for sibling in os.listdir(parent):
            candidate = os.path.join(parent, sibling, 'debugimg')
            if os.path.isdir(candidate):
                debugimg_dir = candidate
                break

    def on_click(event):
        if event.inaxes not in axs or event.button != 1:
            return
        idx = np.argmin(np.abs(time - event.xdata))
        fname = str(filename[idx]).strip()
        img_path = os.path.join(debugimg_dir, fname + '.png')
        if not os.path.exists(img_path):
            print(f'Debug image not found: {img_path}')
            return
        img = plt.imread(img_path)
        img_fig, img_ax = plt.subplots(figsize=(8, 6))
        img_ax.imshow(img)
        img_ax.set_title(f'{fname}  |  t={time[idx]:.2f}s  |  Gantry={gantry[idx]:.1f}\u00b0', fontsize=10)
        img_ax.axis('off')
        img_fig.tight_layout()
        img_fig.show()

    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()

    # Save PNG next to the input file
    out_path = os.path.splitext(filepath)[0] + '_plot.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_single_marker.py <trajectory_file>")
        sys.exit(1)
    plot_single_marker(sys.argv[1])
