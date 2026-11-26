#!/usr/bin/env python3
"""Offline processing of a rosbag2 with the SAME core pipeline as the ROS node.

Usage:
  ros2 run crossing_target_tracking offline_process --bag <BAG_DIR> \
      [--config config/crossing_tracking.yaml] [--set section.key=value ...] \
      [--start 40] [--duration 60] [--out results/run1] [--snapshots 20] [--label name]

Outputs in --out (all CSV comma separated, header row, SI units, fixed frame):
  frames.csv      one row per scan: validity, registration, counters, per-stage latency [ms]
  blobs.csv       per-frame blobs (Sec. 3.2)
  low_tracks.csv  per-frame low-level tracks (Sec. 3.3)
  clusters.csv    per-frame clusters with member track ids (Sec. 3.4)
  objects.csv     per-frame high-level tracks (Sec. 3.5)
  summary.json    configuration, counts, latency statistics, CPU / memory, track statistics
  snapshots/      optional BEV PNGs (gray: occupied, dark: background, white: dynamic,
                  blue: shadow, green: high-level tracks + velocity, yellow: clusters)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

from crossing_target_tracking.core.config import load_config, to_dict  # noqa: E402
from crossing_target_tracking.core.pipeline import FrameResult, Scan, TrackingPipeline  # noqa: E402
from crossing_target_tracking.core.registration import yaw_of  # noqa: E402
from crossing_target_tracking.core.tracker import STATUS_NAMES  # noqa: E402
from crossing_target_tracking.ros.pointcloud import cloud_to_xyz  # noqa: E402
from crossing_target_tracking.tools.bag_io import iterate, metadata, stamp_sec  # noqa: E402

STAGES = ['ground', 'registration', 'grid', 'dynamic', 'low_level', 'clustering', 'high_level', 'total']
REG_KEYS = ['odom_offset', 'odom_bracket', 'odom_extrapolation', 'ego_speed', 'cpd_sigma2',
            'cpd_iterations', 'cpd_matched', 'cpd_dx', 'cpd_dy', 'cpd_dyaw_deg', 'cpd_vs_odom_dxy',
            'cpd_vs_odom_dyaw_deg', 'odom_moved_m', 'odom_dyaw_deg']


class Writers:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.files = {}
        self.w = {}
        hdr = {
            'frames': ['frame', 'stamp', 't', 'valid', 'reason', 'method', 'missed', 'sensor_x', 'sensor_y',
                       'sensor_yaw', 'n_points', 'n_obstacle_points', 'ground_d', 'ground_tilt_deg',
                       'ground_inliers', 'n_blobs', 'n_low_tracks', 'n_low_confirmed', 'n_clusters',
                       'n_high_tracks', 'shadow_cells', 'warmup_cells', 'background_cells',
                       'dynamic_raw_cells', 'frames_in_history'] + REG_KEYS + [f'ms_{s}' for s in STAGES],
            'blobs': ['frame', 'stamp', 't', 'blob_id', 'x', 'y', 'area', 'x_min', 'y_min', 'x_max', 'y_max',
                      'n_pixels'],
            'low_tracks': ['frame', 'stamp', 't', 'id', 'status', 'x', 'y', 'vx', 'vy', 'omega', 'std_x', 'std_y',
                           'std_vx', 'std_vy', 'hits', 'misses', 'age'],
            'clusters': ['frame', 'stamp', 't', 'cluster_id', 'x', 'y', 'vx', 'vy', 'omega', 'n_members',
                         'member_ids'],
            'objects': ['frame', 'stamp', 't', 'id', 'status', 'x', 'y', 'vx', 'vy', 'speed', 'heading',
                        'heading_valid', 'omega', 'var_x', 'var_y', 'var_vx', 'var_vy', 'var_omega', 'age',
                        'hits', 'misses', 'member_ids'],
        }
        for k, h in hdr.items():
            f = open(os.path.join(out, f'{k}.csv'), 'w', newline='')
            self.files[k] = f
            self.w[k] = csv.writer(f)
            self.w[k].writerow(h)

    def close(self):
        for f in self.files.values():
            f.close()


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.6g}' if np.isfinite(v) else ''
    return v


def write_frame(W: Writers, r: FrameResult, t0: float, min_heading_speed: float):
    t = r.stamp - t0
    T = r.T_fixed_sensor
    reg = r.registration.diag if r.registration else {}
    c = r.counters
    row = [r.index, f'{r.stamp:.6f}', f'{t:.4f}', int(r.valid), r.reason,
           r.registration.method if r.registration else '', r.missed_instants,
           _fmt(float(T[0, 3])) if T is not None else '', _fmt(float(T[1, 3])) if T is not None else '',
           _fmt(yaw_of(T)) if T is not None else '']
    row += [_fmt(c.get(k, float('nan'))) for k in
            ['n_points', 'n_obstacle_points', 'ground_d', 'ground_tilt_deg', 'ground_inliers', 'n_blobs',
             'n_low_tracks', 'n_low_confirmed', 'n_clusters', 'n_high_tracks', 'shadow_cells',
             'warmup_cells', 'background_cells', 'dynamic_raw_cells', 'frames_in_history']]
    row += [_fmt(float(reg[k])) if k in reg and reg[k] is not None else '' for k in REG_KEYS]
    row += [_fmt(r.timings.get(s, float('nan')) * 1e3) for s in STAGES]
    W.w['frames'].writerow(row)
    for b in r.blobs:
        W.w['blobs'].writerow([r.index, f'{r.stamp:.6f}', f'{t:.4f}', b.id, _fmt(b.x), _fmt(b.y), _fmt(b.area),
                               *(_fmt(float(v)) for v in b.bbox), b.n_pixels])
    for s in r.low_tracks:
        sd = np.sqrt(np.diag(s.P))
        W.w['low_tracks'].writerow([r.index, f'{r.stamp:.6f}', f'{t:.4f}', s.id, STATUS_NAMES[s.status],
                                    *(_fmt(float(v)) for v in s.x), *(_fmt(float(v)) for v in sd[:4]),
                                    s.hits, s.consecutive_misses, _fmt(r.stamp - s.first_stamp)])
    for cl in r.clusters:
        W.w['clusters'].writerow([r.index, f'{r.stamp:.6f}', f'{t:.4f}', cl.id, _fmt(float(cl.position[0])),
                                  _fmt(float(cl.position[1])), _fmt(float(cl.velocity[0])),
                                  _fmt(float(cl.velocity[1])), _fmt(cl.omega), cl.n_members,
                                  ' '.join(map(str, cl.member_ids))])
    for s in r.high_tracks:
        x = s.x
        sp = float(np.hypot(x[2], x[3]))
        W.w['objects'].writerow([r.index, f'{r.stamp:.6f}', f'{t:.4f}', s.id, STATUS_NAMES[s.status],
                                 *(_fmt(float(v)) for v in x[:4]), _fmt(sp), _fmt(float(np.arctan2(x[3], x[2]))),
                                 int(sp >= min_heading_speed), _fmt(float(x[4])),
                                 *(_fmt(float(v)) for v in np.diag(s.P)), _fmt(r.stamp - s.first_stamp),
                                 s.hits, s.consecutive_misses, ' '.join(map(str, s.member_ids))])


def snapshot(path: str, r: FrameResult, crop: float = 30.0):
    import cv2
    if r.dyn is None:
        return
    spec = r.spec
    img = np.zeros(spec.shape + (3,), np.uint8)
    img[r.btilde > 0] = (110, 110, 110)
    img[r.dyn.background > 0] = (60, 60, 60)
    img[r.dyn.shadow] = (90, 30, 0)
    img[r.dyn.dynamic > 0] = (255, 255, 255)
    g = spec.resolution
    ox, oy = spec.origin

    def px(x, y):
        return int((x - ox) / g), int((y - oy) / g)

    for cl in r.clusters:
        cv2.circle(img, px(*cl.position), 4, (0, 220, 255), 1)
    for s in r.high_tracks:
        col = (0, 200, 0) if s.status == 1 else (0, 0, 255)
        p0 = px(s.x[0], s.x[1])
        p1 = px(s.x[0] + s.x[2], s.x[1] + s.x[3])
        cv2.circle(img, p0, 3, col, -1)
        cv2.arrowedLine(img, p0, p1, col, 1, tipLength=0.3)
        hist = [px(h[1], h[2]) for h in s.history]
        for a, b in zip(hist[:-1], hist[1:]):
            cv2.line(img, a, b, (0, 120, 0), 1)
    if r.T_fixed_sensor is not None:
        cv2.drawMarker(img, px(r.T_fixed_sensor[0, 3], r.T_fixed_sensor[1, 3]), (255, 0, 255),
                       cv2.MARKER_CROSS, 8)
    img = img[::-1]                   # y up
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for s in r.high_tracks:
        x, y = px(s.x[0], s.x[1])
        cv2.putText(big, str(s.id), (2 * x + 6, 2 * (spec.ny - y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1)
    cv2.putText(big, f't={r.stamp:.1f} blobs={len(r.blobs)} hl={len(r.high_tracks)}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(path, big)


def track_stats(objects_rows):
    """Lifetime statistics of high-level tracks (no ground truth needed)."""
    by = {}
    for fr, tid, x, y, sp, status in objects_rows:
        by.setdefault(tid, []).append((fr, x, y, sp, status))
    lif, disp, obs_frac = [], [], []
    for tid, v in by.items():
        v.sort()
        lif.append(v[-1][0] - v[0][0] + 1)
        disp.append(float(np.hypot(v[-1][1] - v[0][1], v[-1][2] - v[0][2])))
        obs_frac.append(float(np.mean([s == 1 for *_, s in v])))
    lif = np.array(lif)
    disp = np.array(disp)
    return {'n_tracks': len(by), 'lifetime_frames_median': float(np.median(lif)) if len(lif) else 0,
            'lifetime_frames_p90': float(np.percentile(lif, 90)) if len(lif) else 0,
            'n_tracks_lifetime_ge_20': int((lif >= 20).sum()),
            'n_tracks_lifetime_lt_5': int((lif < 5).sum()),
            'n_tracks_displacement_lt_1m': int((disp < 1.0).sum()),
            'n_tracks_displacement_ge_5m': int((disp >= 5.0).sum())}


def run(args):
    cfg = load_config(args.config or None, args.set)
    cloud_topic = args.cloud_topic or cfg.input.cloud_topic
    odom_topic = args.odom_topic if args.odom_topic is not None else cfg.input.odom_topic
    meta = metadata(args.bag)
    bag_t0 = meta['starting_time']['nanoseconds_since_epoch']
    t_start = bag_t0 + int(args.start * 1e9)
    t_end = t_start + int(args.duration * 1e9) if args.duration > 0 else None
    preroll = int(args.odom_preroll * 1e9)
    topics = [cloud_topic] + ([odom_topic] if odom_topic and cfg.registration.mode != 'cpd' else [])
    pipe = TrackingPipeline(cfg, keep_history=args.snapshots > 0)
    W = Writers(args.out)
    if args.snapshots > 0:
        os.makedirs(os.path.join(args.out, 'snapshots'), exist_ok=True)
    try:
        import psutil
        proc = psutil.Process()
    except ImportError:
        proc = None
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    proc_time = 0.0
    n = 0
    t0 = None
    obj_rows = []
    stage_ms = {s: [] for s in STAGES}
    n_valid = 0
    methods = {}
    rss_max = 0
    first_cloud = True
    for topic, msg, trecv in iterate(args.bag, topics, max(bag_t0, t_start - preroll), t_end):
        if topic == odom_topic:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            pipe.add_odom(stamp_sec(msg.header.stamp), trecv * 1e-9, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                          msg.header.frame_id, msg.child_frame_id)
            continue
        if trecv < t_start:
            continue
        if first_cloud and args.verbose:
            print(f'first cloud at bag t={(trecv - bag_t0) * 1e-9:.2f}s')
        first_cloud = False
        xyz, ring, ppr, info = cloud_to_xyz(msg)
        stamp = stamp_sec(msg.header.stamp)
        t0 = stamp if t0 is None else t0
        tp = time.perf_counter()
        r = pipe.process(Scan(stamp, xyz, msg.header.frame_id, ring, ppr, info.get('time')))
        proc_time += time.perf_counter() - tp
        write_frame(W, r, t0, cfg.high.min_heading_speed)
        for s in r.high_tracks:
            obj_rows.append((r.index, s.id, s.x[0], s.x[1], float(np.hypot(s.x[2], s.x[3])), s.status))
        for s in STAGES:
            if s in r.timings:
                stage_ms[s].append(r.timings[s] * 1e3)
        n_valid += int(r.valid)
        if r.registration:
            methods[r.registration.method] = methods.get(r.registration.method, 0) + 1
        if args.snapshots > 0 and n % args.snapshots == 0:
            snapshot(os.path.join(args.out, 'snapshots', f'{n:05d}.png'), r)
        if proc is not None and n % 50 == 0:
            rss_max = max(rss_max, proc.memory_info().rss)
        n += 1
        if args.verbose and n % 100 == 0:
            print(f'{n} frames, t={stamp - t0:.1f}s, blobs={len(r.blobs)} low={len(r.low_tracks)} '
                  f'clusters={len(r.clusters)} high={len(r.high_tracks)} total={r.timings["total"]*1e3:.1f}ms')
    W.close()
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    summ = {
        'bag': os.path.abspath(args.bag), 'label': args.label, 'start_s': args.start,
        'duration_s': args.duration, 'frames': n, 'valid_frames': n_valid, 'invalid_frames': n - n_valid,
        'missed_instants': pipe.n_missed_total, 'registration_methods': methods,
        'angular_density': pipe.delta, 'angular_density_source': pipe.delta_source,
        'odom_clock_offset_s': pipe.registrar.odom.offset,
        'grid': None if pipe.spec is None else {'nx': pipe.spec.nx, 'ny': pipe.spec.ny,
                                                'resolution': pipe.spec.resolution,
                                                'center': [pipe.spec.cx, pipe.spec.cy]},
        'low_level': {'created': pipe.low.n_created, 'confirmed': pipe.low.n_confirmed,
                      'deleted': pipe.low.n_deleted},
        'high_level': {'created': pipe.high.n_created, 'deleted': pipe.high.n_deleted},
        'high_level_track_stats': track_stats(obj_rows),
        'latency_ms': {s: {'mean': float(np.mean(v)), 'p50': float(np.median(v)),
                           'p95': float(np.percentile(v, 95)), 'max': float(np.max(v))}
                       for s, v in stage_ms.items() if v},
        'pipeline_time_s': proc_time, 'wall_time_s': wall,
        'effective_pipeline_rate_hz': n / proc_time if proc_time > 0 else None,
        'process_cpu_time_s': cpu, 'cpu_utilisation_of_one_core': cpu / wall if wall > 0 else None,
        'max_rss_mb': rss_max / 2 ** 20 if rss_max else None,
        'config': to_dict(cfg),
        'note': 'No ground truth: counts and lifetimes are descriptive, not accuracy metrics.'}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    return summ


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--config', default='')
    ap.add_argument('--set', action='append', default=[], help='override, e.g. shadow.enabled=false')
    ap.add_argument('--start', type=float, default=0.0, help='seconds from bag start')
    ap.add_argument('--duration', type=float, default=0.0, help='seconds (0 = to the end)')
    ap.add_argument('--out', default='crossing_tracking_results')
    ap.add_argument('--label', default='')
    ap.add_argument('--cloud-topic', default='')
    ap.add_argument('--odom-topic', default=None)
    ap.add_argument('--odom-preroll', type=float, default=3.0)
    ap.add_argument('--snapshots', type=int, default=0, help='save a BEV PNG every N frames (0 = off)')
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap


def main(argv=None):
    s = run(build_parser().parse_args(argv))
    keys = ['frames', 'valid_frames', 'registration_methods', 'odom_clock_offset_s', 'angular_density',
            'high_level_track_stats', 'effective_pipeline_rate_hz', 'cpu_utilisation_of_one_core', 'max_rss_mb']
    print(json.dumps({k: s[k] for k in keys}, indent=1))
    print('latency ms (mean/p95):', {k: (round(v['mean'], 2), round(v['p95'], 2))
                                     for k, v in s['latency_ms'].items()})


if __name__ == '__main__':
    main()
