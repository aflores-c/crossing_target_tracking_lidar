#!/usr/bin/env python3
"""Diagnostic plots and descriptive metrics from offline_process outputs.

Usage:
  ros2 run crossing_target_tracking evaluate --runs <dir> [<dir> ...] --out <plots_dir>
      [--min-frames 20] [--min-displacement 5] [--top 6]

Per run (paper Figs. 15-18 style, for tracks WITHOUT ground truth):
  trajectories.png   estimated x/y trajectories of high-level objects (Fig. 17 style)
  velocity.png       vx, vy vs time for the longest moving objects (Fig. 18 style)
  speed_accel.png    speed and acceleration by numerical differentiation (Figs. 15-16 (d))
  counts.png         number of blobs / low-level confirmed / clusters / high-level tracks vs time
  latency.png        per-stage latency (box plot) and total latency vs time
  metrics.json       descriptive metrics (no ground truth -> no accuracy claims)
With several runs, ablation_comparison.md/.png compare the descriptive metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

STAGES = ['ground', 'registration', 'grid', 'dynamic', 'low_level', 'clustering', 'high_level']


def read_csv(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return list(r)


def ffloat(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def load_run(d):
    frames = read_csv(os.path.join(d, 'frames.csv'))
    objs = read_csv(os.path.join(d, 'objects.csv'))
    summ = json.load(open(os.path.join(d, 'summary.json')))
    tracks = defaultdict(list)
    for o in objs:
        tracks[int(o['id'])].append({k: (ffloat(v) if k not in ('status', 'member_ids') else v)
                                     for k, v in o.items()})
    return frames, tracks, summ


def track_arrays(rows):
    t = np.array([r['t'] for r in rows])
    a = {k: np.array([r[k] for r in rows]) for k in ('x', 'y', 'vx', 'vy', 'speed', 'omega')}
    obs = np.array([r['status'] == 'confirmed' for r in rows])
    return t, a, obs


def acceleration(t, vx, vy):
    """Numerical differentiation of the estimated velocity (paper: acceleration is not a state)."""
    if len(t) < 3:
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    sp = np.hypot(vx, vy)
    along = np.gradient(sp, t)            # longitudinal (speed derivative)
    return along, np.hypot(ax, ay)


def descriptive_metrics(frames, tracks, min_frames, min_disp, dt_nom=0.1):
    n_frames = len(frames)
    valid = [f for f in frames if f['valid'] == '1']
    per = {}
    for tid, rows in tracks.items():
        t, a, obs = track_arrays(rows)
        disp = float(np.hypot(a['x'][-1] - a['x'][0], a['y'][-1] - a['y'][0]))
        along, amag = acceleration(t, a['vx'], a['vy'])
        per[tid] = {'frames': len(rows), 'duration_s': float(t[-1] - t[0]), 'displacement_m': disp,
                    'observed_fraction': float(obs.mean()), 'mean_speed': float(np.mean(a['speed'])),
                    'max_speed': float(np.max(a['speed'])),
                    'accel_abs_p50': float(np.nanmedian(np.abs(along))) if len(t) > 2 else float('nan'),
                    'jerk_rms': float(np.sqrt(np.nanmean(np.gradient(along, t) ** 2))) if len(t) > 3 else float('nan'),
                    't_start': float(t[0]), 't_end': float(t[-1]), 'x_end': float(a['x'][-1]),
                    'y_end': float(a['y'][-1]), 'x_start': float(a['x'][0]), 'y_start': float(a['y'][0]),
                    'vx_end': float(a['vx'][-1]), 'vy_end': float(a['vy'][-1])}
    long = [p for p in per.values() if p['frames'] >= min_frames]
    moving = [p for p in long if p['displacement_m'] >= min_disp]
    static = [p for p in long if p['displacement_m'] < 1.0]
    # fragmentation proxy: a track starting within 1.5 s and (3 m + v*dt) of where another ended
    ends = sorted(per.values(), key=lambda p: p['t_end'])
    frag = 0
    for p in per.values():
        for q in ends:
            dt = p['t_start'] - q['t_end']
            if 0 < dt <= 1.5:
                px = q['x_end'] + q['vx_end'] * dt
                py = q['y_end'] + q['vy_end'] * dt
                if np.hypot(p['x_start'] - px, p['y_start'] - py) < 3.0:
                    frag += 1
                    break
    counts = {k: np.array([ffloat(f[k]) for f in valid]) for k in
              ('n_blobs', 'n_low_confirmed', 'n_clusters', 'n_high_tracks')}
    lat = {s: np.array([ffloat(f[f'ms_{s}']) for f in valid]) for s in STAGES + ['total']}
    return {
        'frames': n_frames, 'valid_frames': len(valid),
        'mean_blobs_per_frame': float(np.nanmean(counts['n_blobs'])) if valid else None,
        'mean_low_confirmed': float(np.nanmean(counts['n_low_confirmed'])) if valid else None,
        'mean_clusters': float(np.nanmean(counts['n_clusters'])) if valid else None,
        'mean_high_tracks': float(np.nanmean(counts['n_high_tracks'])) if valid else None,
        'high_tracks_total': len(per),
        f'high_tracks_ge_{min_frames}_frames': len(long),
        f'moving_tracks_disp_ge_{min_disp}m': len(moving),
        'static_tracks_disp_lt_1m (false-detection proxy)': len(static),
        'fragmentation_proxy_restarts': frag,
        'moving_tracks_median_duration_s': float(np.median([p['duration_s'] for p in moving])) if moving else None,
        'moving_tracks_observed_fraction_median': float(np.median([p['observed_fraction'] for p in moving]))
        if moving else None,
        'moving_tracks_accel_abs_p50_median': float(np.nanmedian([p['accel_abs_p50'] for p in moving]))
        if moving else None,
        'moving_tracks_jerk_rms_median (velocity-stability proxy)':
            float(np.nanmedian([p['jerk_rms'] for p in moving])) if moving else None,
        'latency_ms': {s: {'mean': float(np.nanmean(v)), 'p95': float(np.nanpercentile(v, 95)),
                           'max': float(np.nanmax(v))} for s, v in lat.items() if len(v)},
        'note': 'Descriptive only: no ground truth for this bag. Proxies are labelled as such.',
    }, per


def plot_run(d, out, min_frames, min_disp, top):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frames, tracks, summ = load_run(d)
    m, per = descriptive_metrics(frames, tracks, min_frames, min_disp)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump({'run': d, 'label': summ.get('label'), 'summary': {k: summ[k] for k in
                   ('frames', 'valid_frames', 'registration_methods', 'effective_pipeline_rate_hz',
                    'cpu_utilisation_of_one_core', 'max_rss_mb', 'odom_clock_offset_s', 'angular_density')},
                   'metrics': m}, f, indent=1)
    moving = sorted([tid for tid, p in per.items() if p['frames'] >= min_frames and p['displacement_m'] >= min_disp],
                    key=lambda k: -per[k]['frames'])
    sel = moving[:top]
    cmap = plt.get_cmap('tab10')
    # trajectories
    fig, ax = plt.subplots(figsize=(7, 7))
    sx = np.array([ffloat(f['sensor_x']) for f in frames])
    sy = np.array([ffloat(f['sensor_y']) for f in frames])
    ax.plot(sx, sy, 'm-', lw=2, label='sensor')
    for tid, rows in tracks.items():
        if per[tid]['frames'] >= min_frames:
            t, a, _ = track_arrays(rows)
            ax.plot(a['x'], a['y'], '-', color='0.75', lw=0.8)
    for k, tid in enumerate(sel):
        t, a, _ = track_arrays(tracks[tid])
        ax.plot(a['x'], a['y'], '-', color=cmap(k % 10), lw=2, label=f'id {tid}')
        ax.plot(a['x'][0], a['y'][0], 'o', color=cmap(k % 10))
    ax.set_aspect('equal')
    ax.set_xlabel('x [m] (fixed frame)')
    ax.set_ylabel('y [m]')
    ax.set_title(f'High-level trajectories (>= {min_frames} frames; colour: longest moving)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'trajectories.png'), dpi=120)
    plt.close(fig)
    # velocity (Fig. 18 style)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k, tid in enumerate(sel):
        t, a, _ = track_arrays(tracks[tid])
        axs[0].plot(t, a['vx'], color=cmap(k % 10), label=f'id {tid}')
        axs[1].plot(t, a['vy'], color=cmap(k % 10))
    axs[0].set_ylabel('vx [m/s]')
    axs[1].set_ylabel('vy [m/s]')
    axs[1].set_xlabel('t [s]')
    axs[0].legend(fontsize=7, ncol=3)
    axs[0].set_title('High-level EKF velocity estimates (no reference available)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'velocity.png'), dpi=120)
    plt.close(fig)
    # speed + acceleration (Figs. 15-16 (b-d) style)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k, tid in enumerate(sel):
        t, a, _ = track_arrays(tracks[tid])
        along, _ = acceleration(t, a['vx'], a['vy'])
        axs[0].plot(t, a['speed'], color=cmap(k % 10), label=f'id {tid}')
        axs[1].plot(t, along, color=cmap(k % 10))
    axs[0].set_ylabel('speed [m/s]')
    axs[1].set_ylabel('d|v|/dt [m/s^2]')
    axs[1].set_xlabel('t [s]')
    axs[0].legend(fontsize=7, ncol=3)
    axs[1].set_title('Acceleration by numerical differentiation of the estimated speed')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'speed_accel.png'), dpi=120)
    plt.close(fig)
    # counts
    t = np.array([ffloat(f['t']) for f in frames])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for k, lab in (('n_blobs', 'blobs'), ('n_low_confirmed', 'low-level confirmed'), ('n_clusters', 'clusters'),
                   ('n_high_tracks', 'high-level tracks')):
        ax.plot(t, [ffloat(f[k]) for f in frames], label=lab, lw=1)
    inval = np.array([f['valid'] != '1' for f in frames])
    if inval.any():
        ax.fill_between(t, 0, 1, where=inval, transform=ax.get_xaxis_transform(), color='r', alpha=0.15,
                        label='invalid registration')
    ax.set_xlabel('t [s]')
    ax.set_ylabel('count')
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'counts.png'), dpi=120)
    plt.close(fig)
    # latency
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw={'width_ratios': [1.3, 2]})
    data = [[ffloat(f[f'ms_{s}']) for f in frames if f['valid'] == '1'] for s in STAGES + ['total']]
    axs[0].boxplot(data, labels=STAGES + ['total'], showfliers=False)
    axs[0].tick_params(axis='x', rotation=45, labelsize=7)
    axs[0].set_ylabel('ms')
    axs[1].plot(t, [ffloat(f['ms_total']) for f in frames], lw=0.7)
    axs[1].axhline(100, color='r', ls='--', lw=0.8, label='10 Hz budget')
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('total ms')
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'latency.png'), dpi=120)
    plt.close(fig)
    return m, summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-frames', type=int, default=20)
    ap.add_argument('--min-displacement', type=float, default=5.0)
    ap.add_argument('--top', type=int, default=6)
    a = ap.parse_args(argv)
    rows = []
    for d in a.runs:
        name = os.path.basename(os.path.normpath(d))
        m, summ = plot_run(d, os.path.join(a.out, name), a.min_frames, a.min_displacement, a.top)
        rows.append((name, summ.get('label') or name, m, summ))
        print(name, json.dumps({k: v for k, v in m.items() if k != 'latency_ms'}, indent=None))
    if len(rows) > 1:
        keys = [k for k in rows[0][2] if k not in ('latency_ms', 'note', 'frames')]
        L = ['| run | ' + ' | '.join(keys) + ' | total ms mean |', '|' + '---|' * (len(keys) + 2)]
        for name, label, m, _ in rows:
            vals = [(f'{m[k]:.2f}' if isinstance(m[k], float) else str(m[k])) for k in keys]
            L.append(f'| {label} | ' + ' | '.join(vals) + f" | {m['latency_ms']['total']['mean']:.1f} |")
        with open(os.path.join(a.out, 'comparison.md'), 'w') as f:
            f.write('\n'.join(L) + '\n')
        print('\n'.join(L))


if __name__ == '__main__':
    main()
