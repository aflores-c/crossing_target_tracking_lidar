#!/usr/bin/env python3
"""Reproducible real-bag experiments: main runs, ablations, registration variants.

Usage:
  ros2 run crossing_target_tracking run_experiments --bag <BAG_DIR> --out <DIR>
      [--config <yaml>] [--only name ...] [--jobs 4]

Time intervals refer to bag time (seconds from the bag start) and were chosen
from docs/DATASET_REPORT.md for bag_go2_20261105_165413:
  stationary : 40-137 s  (odometry: 1.6 deg yaw, 2.5 cm drift)
  walking    : 145-205 s (4.4 m path + turns of -156 deg and -54 deg)
Pass --stationary / --walking to use other intervals for another bag.
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from multiprocessing import Pool  # noqa: E402


def experiments(stat, walk):
    s0, s1 = stat
    w0, w1 = walk
    S = dict(start=s0, duration=s1 - s0)
    W = dict(start=w0, duration=w1 - w0)
    return {
        'stationary_main': dict(**S, set=[], label='paper (default), odom registration'),
        'stationary_shadow_off': dict(**S, set=['shadow.enabled=false'], label='ablation: shadow off'),
        'stationary_position_only': dict(**S, set=['clustering.use_velocity=false'],
                                         label='ablation: position-only clustering'),
        'stationary_raw_count': dict(**S, set=['grid.normalization=raw_count'],
                                     label='ablation: raw-count binarisation'),
        'stationary_mode_stationary': dict(**S, set=['registration.mode=stationary'],
                                           label='registration: stationary assumption'),
        'walking_odom': dict(**W, set=['registration.cpd_diagnostic=true'],
                             label='walking, odom registration (+CPD diagnostic)'),
        'walking_cpd': dict(**W, set=['registration.mode=cpd'], label='walking, CPD registration (3D)'),
        'walking_cpd2d': dict(**W, set=['registration.mode=cpd', 'registration.cpd_project_2d=true'],
                              label='walking, CPD registration (2D BEV)'),
        'walking_no_compensation': dict(**W, set=['registration.mode=stationary'],
                                        label='walking, NO ego-motion compensation (negative control)'),
        'full_bag': dict(start=0.0, duration=0.0, set=[], label='full bag, default'),
    }


def _run(args):
    name, e, bag, out, config = args
    from crossing_target_tracking.tools import offline_process
    argv = ['--bag', bag, '--out', os.path.join(out, name), '--start', str(e['start']),
            '--duration', str(e['duration']), '--label', e['label']]
    if config:
        argv += ['--config', config]
    for s in e['set']:
        argv += ['--set', s]
    s = offline_process.run(offline_process.build_parser().parse_args(argv))
    return name, {k: s[k] for k in ('frames', 'valid_frames', 'registration_methods', 'effective_pipeline_rate_hz',
                                    'high_level_track_stats')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--stationary', nargs=2, type=float, default=[40.0, 137.0])
    ap.add_argument('--walking', nargs=2, type=float, default=[145.0, 205.0])
    ap.add_argument('--no-eval', action='store_true')
    a = ap.parse_args(argv)
    ex = experiments(a.stationary, a.walking)
    names = a.only or list(ex)
    jobs = [(n, ex[n], a.bag, a.out, a.config) for n in names]
    with Pool(min(a.jobs, len(jobs))) as p:
        for name, s in p.imap_unordered(_run, jobs):
            print(name, json.dumps(s))
    if not a.no_eval:
        from crossing_target_tracking.tools import evaluate
        groups = {'stationary': [n for n in names if n.startswith('stationary')],
                  'walking': [n for n in names if n.startswith('walking')],
                  'full': [n for n in names if n == 'full_bag']}
        for g, ns in groups.items():
            if ns:
                evaluate.main(['--runs', *[os.path.join(a.out, n) for n in ns],
                               '--out', os.path.join(a.out, 'plots', g)])


if __name__ == '__main__':
    main()
