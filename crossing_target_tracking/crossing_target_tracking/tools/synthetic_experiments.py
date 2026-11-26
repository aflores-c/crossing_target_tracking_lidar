#!/usr/bin/env python3
"""Run the synthetic experiments (scenarios 1-8 + extras) and the ablations.

Usage:
  ros2 run crossing_target_tracking synthetic_experiments --out results/synthetic [--jobs 8]
Writes synthetic_results.json and synthetic_results.md (a table). Ground truth
is exact here, so position/velocity RMSE, ID switches and precision/recall are
real metrics -- for the SYNTHETIC data only.
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from multiprocessing import Pool  # noqa: E402

from crossing_target_tracking.tools.scenarios import evaluate_scenario, run_scenario, scenarios  # noqa: E402

VARIANTS = {
    'paper (default)': [],
    'ablation: shadow off': ['shadow.enabled=false'],
    'ablation: position-only clustering': ['clustering.use_velocity=false'],
    'ablation: raw-count binarisation': ['grid.normalization=raw_count'],
}


def _job(a):
    name, var = a
    m = evaluate_scenario(run_scenario(scenarios()[name], VARIANTS[var]))
    m['variant'] = var
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='synthetic_results')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(n, v) for v in a.variants for n in scenarios()]
    with Pool(a.jobs) as p:
        res = p.map(_job, jobs)
    with open(os.path.join(a.out, 'synthetic_results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    desc = {k: s.description for k, s in scenarios().items()}
    L = ['| variant | scenario | GT obj | HL ids per GT obj | ID sw | false HL ids | precision | recall | '
         'pos RMSE [m] | vel RMSE [m/s] | ms/frame |', '|' + '---|' * 11]

    def f(v, d=2):
        return '–' if v != v else f'{v:.{d}f}'
    for m in res:
        ids = ', '.join(f'{k}:{v}' for k, v in m['track_ids_per_gt'].items()) or '–'
        L.append(f"| {m['variant']} | {m['scenario']} | {m['gt_objects']} | {ids} | {m['id_switches']} | "
                 f"{m['false_track_ids']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['position_rmse_m'])} | "
                 f"{f(m['velocity_rmse_mps'])} | {f(m['mean_ms_per_frame'], 1)} |")
    L += ['', 'Scenarios:'] + [f'* `{k}`: {v}' for k, v in desc.items()]
    with open(os.path.join(a.out, 'synthetic_results.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
