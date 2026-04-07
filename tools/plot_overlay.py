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
from matplotlib.widgets import Slider, CheckButtons


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


def plot_overlay(kim_filepath, hex_filepath, x_max=100, init_dt_ms=25,
                 init_offset=0):
    kim_time, kim_axes = load_kim(kim_filepath)
    hex_time, hex_indices, hex_axes = load_hex(hex_filepath)

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(bottom=0.27)
    fig.suptitle(make_title(kim_filepath), fontsize=14)

    kim_colors = sns.color_palette("deep", 3)
    hex_lines = []

    y_limits = {'LR': (-10, 10), 'SI': (-4, 4), 'AP': (-10, 10)}
    init_hex_time = hex_indices * (init_dt_ms / 1000.0) + init_offset
    for ax, (label, kim_vals), color in zip(axs, kim_axes.items(), kim_colors):
        sns.scatterplot(x=kim_time, y=kim_vals, ax=ax, color=color, s=15,
                        edgecolor='none', label='KIM')
        line, = ax.plot(init_hex_time, hex_axes[label], color='red', linewidth=0.8,
                        alpha=0.7, label='Hexamotion')
        hex_lines.append(line)
        ax.set_ylabel(f'{label} (mm)')
        ax.set_ylim(*y_limits[label])
        ax.set_xlim(0, x_max)

    axs[0].legend(loc='upper right', fontsize=9)
    axs[-1].set_xlabel('Time (sec)')

    # Interval slider
    ax_dt = plt.axes([0.15, 0.18, 0.7, 0.02])
    s_dt = Slider(ax_dt, 'Hex interval (ms)', 10, 200, valinit=init_dt_ms, valstep=0.1, valfmt='%.1f')

    # Per-axis time offset sliders
    ax_lr_t = plt.axes([0.15, 0.13, 0.7, 0.02])
    ax_si_t = plt.axes([0.15, 0.08, 0.7, 0.02])
    ax_ap_t = plt.axes([0.15, 0.03, 0.7, 0.02])

    s_lr = Slider(ax_lr_t, 'LR time (s)', -500, 500, valinit=init_offset, valstep=0.01, valfmt='%.2f')
    s_si = Slider(ax_si_t, 'SI time (s)', -500, 500, valinit=init_offset, valstep=0.01, valfmt='%.2f')
    s_ap = Slider(ax_ap_t, 'AP time (s)', -500, 500, valinit=init_offset, valstep=0.01, valfmt='%.2f')

    sliders = [s_lr, s_si, s_ap]

    # Lock checkbox to synchronise all time sliders
    ax_lock = plt.axes([0.02, 0.13, 0.08, 0.06])
    chk_lock = CheckButtons(ax_lock, ['Lock'], [False])
    locked = [False]

    def on_lock(label):
        locked[0] = not locked[0]

    chk_lock.on_clicked(on_lock)

    # Track which slider is being dragged to propagate to others
    _updating = [False]

    def make_update(source_slider):
        def update(val):
            if _updating[0]:
                return
            dt = s_dt.val / 1000.0
            scaled_time = hex_indices * dt
            if locked[0] and source_slider in sliders:
                _updating[0] = True
                for s in sliders:
                    if s is not source_slider:
                        s.set_val(source_slider.val)
                _updating[0] = False
            for line, s in zip(hex_lines, sliders):
                line.set_xdata(scaled_time + s.val)
            fig.canvas.draw_idle()
        return update

    s_dt.on_changed(make_update(s_dt))
    for s in sliders:
        s.on_changed(make_update(s))

    # Scroll to zoom, right-click drag to pan
    def on_scroll(event):
        if event.inaxes not in axs:
            return
        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        scale = 0.8 if event.button == 'up' else 1.25
        ax.set_xlim(xdata - (xdata - xlim[0]) * scale,
                    xdata + (xlim[1] - xdata) * scale)
        ax.set_ylim(ydata - (ydata - ylim[0]) * scale,
                    ydata + (ylim[1] - ydata) * scale)
        fig.canvas.draw_idle()

    _pan_state = [None]

    def on_press(event):
        if event.inaxes not in axs or event.button != 3:
            return
        _pan_state[0] = (event.inaxes, event.xdata, event.ydata,
                         event.inaxes.get_xlim(), event.inaxes.get_ylim())

    def on_release(event):
        _pan_state[0] = None

    def on_motion(event):
        if _pan_state[0] is None or event.inaxes is None:
            return
        ax, x0, y0, xlim0, ylim0 = _pan_state[0]
        if event.inaxes != ax:
            return
        dx = x0 - event.xdata
        dy = y0 - event.ydata
        ax.set_xlim(xlim0[0] + dx, xlim0[1] + dx)
        ax.set_ylim(ylim0[0] + dy, ylim0[1] + dy)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    out_path = os.path.splitext(kim_filepath)[0] + '_overlay.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


def _compute_rmse(kim_time, kim_axes, hex_indices, hex_axes, dt, offset,
                  min_overlap_frac=0.5):
    """Compute combined RMSE for a given hex interval (dt) and time offset.

    Requires hex to overlap at least min_overlap_frac of the KIM time range.
    """
    labels = list(kim_axes.keys())
    hex_time = hex_indices * dt + offset
    mask = (kim_time >= hex_time[0]) & (kim_time <= hex_time[-1])
    n_overlap = mask.sum()
    if n_overlap < 10:
        return np.inf
    if n_overlap / len(kim_time) < min_overlap_frac:
        return np.inf
    total_se = 0
    count = 0
    for label in labels:
        interp_vals = np.interp(kim_time[mask], hex_time, hex_axes[label])
        total_se += np.sum((kim_axes[label][mask] - interp_vals) ** 2)
        count += mask.sum()
    return np.sqrt(total_se / count)


def _compute_neg_corr(kim_time, kim_axes, hex_indices, hex_axes, dt, offset,
                      min_overlap_frac=0.5):
    """Compute negative mean Pearson correlation (for minimisation).

    Correlation is much more sensitive to frequency matching than RMSE.
    """
    labels = list(kim_axes.keys())
    hex_time = hex_indices * dt + offset
    mask = (kim_time >= hex_time[0]) & (kim_time <= hex_time[-1])
    n_overlap = mask.sum()
    if n_overlap < 20:
        return 1.0
    if n_overlap / len(kim_time) < min_overlap_frac:
        return 1.0
    corrs = []
    for label in labels:
        interp_vals = np.interp(kim_time[mask], hex_time, hex_axes[label])
        kim_v = kim_axes[label][mask]
        # Pearson correlation
        kim_std = np.std(kim_v)
        hex_std = np.std(interp_vals)
        if kim_std < 1e-6 or hex_std < 1e-6:
            corrs.append(0)
            continue
        r = np.corrcoef(kim_v, interp_vals)[0, 1]
        corrs.append(r)
    return -np.mean(corrs)


def _find_best_offset_corr(kim_time, kim_axes, hex_indices, hex_axes, dt,
                           step=1.0):
    """Find the best time offset for a given interval dt using correlation."""
    n_hex = len(hex_indices)
    hex_duration = n_hex * dt
    off_min = kim_time[0] - hex_duration
    off_max = kim_time[-1]
    best_offset, best_score = 0, 1.0
    for offset in np.arange(off_min, off_max, step):
        score = _compute_neg_corr(kim_time, kim_axes, hex_indices, hex_axes,
                                  dt, offset, min_overlap_frac=0.5)
        if score < best_score:
            best_score = score
            best_offset = offset
    return best_offset, best_score


def find_best_interval_and_offset(kim_time, kim_axes, hex_indices, hex_axes):
    """Find hex interval (ms) and time offset that minimise combined RMSE.

    Strategy: estimate interval from KIM duration, use correlation-based
    matching (frequency-sensitive) for coarse search, then refine with RMSE.
    """
    kim_duration = kim_time[-1] - kim_time[0]
    n_hex = len(hex_indices)

    # Estimate: hex trace should span roughly the KIM duration
    dt_est = kim_duration / n_hex
    print(f'Duration-based estimate: {dt_est*1000:.1f} ms '
          f'(KIM={kim_duration:.0f}s / {n_hex} points)')

    # Coarse search using CORRELATION: ±50% of estimated interval, 1ms steps
    dt_lo = max(0.005, dt_est * 0.5)
    dt_hi = dt_est * 1.5
    best_dt, best_offset, best_score = dt_est, 0, 1.0
    print(f'Coarse correlation search: interval {dt_lo*1000:.0f}-{dt_hi*1000:.0f} ms ...')
    for dt_ms in np.arange(dt_lo * 1000, dt_hi * 1000, 1.0):
        dt = dt_ms / 1000.0
        offset, score = _find_best_offset_corr(
            kim_time, kim_axes, hex_indices, hex_axes, dt, step=2.0)
        if score < best_score:
            best_score = score
            best_dt = dt
            best_offset = offset

    print(f'Coarse: dt={best_dt*1000:.1f}ms, offset={best_offset:.1f}s, '
          f'correlation={-best_score:.3f}')

    # Refine interval with correlation: ±3ms in 0.2ms steps
    for dt_ms in np.arange(best_dt * 1000 - 3, best_dt * 1000 + 3, 0.2):
        dt = dt_ms / 1000.0
        if dt <= 0:
            continue
        offset, score = _find_best_offset_corr(
            kim_time, kim_axes, hex_indices, hex_axes, dt, step=1.0)
        if score < best_score:
            best_score = score
            best_dt = dt
            best_offset = offset

    # Final RMSE-based fine refinement of offset
    best_rmse = np.inf
    for offset in np.arange(best_offset - 5, best_offset + 5, 0.1):
        rmse = _compute_rmse(kim_time, kim_axes, hex_indices, hex_axes,
                             best_dt, offset)
        if rmse < best_rmse:
            best_rmse = rmse
            best_offset = offset

    return best_dt, best_offset, best_rmse


def plot_overlay_auto(kim_filepath, hex_filepath, x_max=100):
    """Auto-align hexamotion to KIM via combined RMSE minimisation, save plot (no GUI)."""
    kim_time, kim_axes = load_kim(kim_filepath)
    hex_time, hex_indices, hex_axes = load_hex(hex_filepath)

    # Find best interval and offset across all axes
    best_dt, offset, rmse = find_best_interval_and_offset(
        kim_time, kim_axes, hex_indices, hex_axes)
    print(f'Best interval: {best_dt*1000:.1f} ms, offset: {offset:.2f} s '
          f'(combined RMSE = {rmse:.3f} mm)')
    aligned_time = hex_indices * best_dt + offset

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(make_title(kim_filepath), fontsize=14)

    kim_colors = sns.color_palette("deep", 3)

    for ax, (label, kim_vals), color in zip(axs, kim_axes.items(), kim_colors):
        sns.scatterplot(x=kim_time, y=kim_vals, ax=ax, color=color, s=15,
                        edgecolor='none', label='KIM')
        ax.plot(aligned_time, hex_axes[label], color='red', linewidth=0.8,
                alpha=0.7, label=f'Hex ({best_dt*1000:.1f}ms, t+{offset:.1f}s)')
        ax.set_ylabel(f'{label} (mm)')
        ax.set_ylim(-10, 10)
        ax.set_xlim(0, x_max)
        ax.legend(loc='upper right', fontsize=9)

    axs[-1].set_xlabel('Time (sec)')
    plt.tight_layout()

    out_name = os.path.splitext(os.path.basename(kim_filepath))[0] + '_overlay_aligned.png'
    # Try saving next to KIM file; fall back to current directory
    out_path = os.path.join(os.path.dirname(kim_filepath), out_name)
    try:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    except OSError:
        out_path = out_name
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    auto_mode = '--auto' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--auto']

    if len(args) < 2:
        print("Usage: python plot_overlay.py [--auto] <kim_file> <hex_file> "
              "[x_max] [init_dt_ms] [init_offset_s]")
        sys.exit(1)

    x_max = float(args[2]) if len(args) > 2 else 100
    init_dt_ms = float(args[3]) if len(args) > 3 else 25
    init_offset = float(args[4]) if len(args) > 4 else 0

    if auto_mode:
        plot_overlay_auto(args[0], args[1], x_max)
    else:
        plot_overlay(args[0], args[1], x_max, init_dt_ms, init_offset)
