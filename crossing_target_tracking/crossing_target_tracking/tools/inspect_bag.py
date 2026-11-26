#!/usr/bin/env python3
"""Inspect a rosbag2 for LiDAR tracking: topics, fields, rates, frames, odometry
clock offset, robot motion segments, ground geometry, point density.

Usage:
  ros2 run crossing_target_tracking inspect_bag --bag <BAG_DIR> [--out report.md]
     [--cloud-topic auto] [--odom-topic auto] [--sample-every 20]
Writes a Markdown report (and a .json next to it). Nothing is assumed: topic
names are discovered from the bag by message type when 'auto'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

from crossing_target_tracking.core.config import GroundConfig
from crossing_target_tracking.core.ground import GroundRemover
from crossing_target_tracking.ros.pointcloud import cloud_to_struct, cloud_to_xyz
from crossing_target_tracking.tools.bag_io import iterate, metadata, open_reader, stamp_sec


def _yaw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def motion_segments(t, xy, yaw, v_thr=0.10, w_thr=0.02, min_len=0.5):
    """Return list of (t_start, t_end, path_m, dyaw_deg) where the robot moves."""
    if len(t) < 3:
        return []
    # resample at 10 Hz to get robust speeds
    tg = np.arange(t[0], t[-1], 0.1)
    x = np.interp(tg, t, xy[:, 0])
    y = np.interp(tg, t, xy[:, 1])
    yw = np.interp(tg, t, np.unwrap(yaw))
    v = np.hypot(np.gradient(x, tg), np.gradient(y, tg))
    w = np.abs(np.gradient(yw, tg))
    k = 10
    v = np.convolve(v, np.ones(k) / k, mode='same')
    w = np.convolve(w, np.ones(k) / k, mode='same')
    mov = (v > v_thr) | (w > w_thr)
    segs = []
    i = 0
    while i < len(mov):
        if mov[i]:
            j = i
            while j < len(mov) and mov[j]:
                j += 1
            if tg[j - 1] - tg[i] >= min_len:
                path = float(np.sum(np.hypot(np.diff(x[i:j]), np.diff(y[i:j]))))
                segs.append((float(tg[i]), float(tg[j - 1]), path,
                             float(np.degrees(yw[j - 1] - yw[i]))))
            i = j
        else:
            i += 1
    # merge segments separated by < 2 s
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 2.0:
            a = merged[-1]
            merged[-1] = (a[0], s[1], a[2] + s[2], a[3] + s[3])
        else:
            merged.append(s)
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--cloud-topic', default='auto')
    ap.add_argument('--odom-topic', default='auto')
    ap.add_argument('--sample-every', type=int, default=20)
    ap.add_argument('--ground-z-prior', type=float, default=-1.0)
    a = ap.parse_args(argv)

    meta = metadata(a.bag)
    _, types = open_reader(a.bag)
    topics = meta['topics_with_message_count']
    clouds = [t for t, ty in types.items() if ty == 'sensor_msgs/msg/PointCloud2']
    odoms = [t for t, ty in types.items() if ty == 'nav_msgs/msg/Odometry']
    tf_t = [t for t, ty in types.items() if ty == 'tf2_msgs/msg/TFMessage']
    imus = [t for t, ty in types.items() if ty == 'sensor_msgs/msg/Imu']
    cloud_topic = clouds[0] if a.cloud_topic == 'auto' else a.cloud_topic
    if a.cloud_topic == 'auto' and len(clouds) != 1:
        print(f'WARNING: {len(clouds)} PointCloud2 topics {clouds}; using {cloud_topic}', file=sys.stderr)
    odom_topic = (odoms[0] if odoms else None) if a.odom_topic == 'auto' else a.odom_topic

    rep = {'bag': os.path.abspath(a.bag), 'storage': meta.get('storage_identifier'),
           'duration_s': meta['duration']['nanoseconds'] * 1e-9,
           'start_ns': meta['starting_time']['nanoseconds_since_epoch'],
           'size_bytes': sum(os.path.getsize(os.path.join(a.bag, f)) for f in os.listdir(a.bag)),
           'topics': {t['topic_metadata']['name']: {'type': t['topic_metadata']['type'],
                                                     'count': t['message_count']} for t in topics},
           'cloud_topic': cloud_topic, 'odom_topic': odom_topic, 'tf_topics': tf_t, 'imu_topics': imus}

    read_topics = [cloud_topic] + ([odom_topic] if odom_topic else []) + tf_t
    c_hdr, c_rcv, npts, nvalid = [], [], [], []
    o_hdr, o_rcv, o_xy, o_z, o_yaw = [], [], [], [], []
    frames, tf_edges = set(), set()
    samples = []
    gr = GroundRemover(GroundConfig(z_prior=a.ground_z_prior))
    fields_desc = None
    k = 0
    for topic, msg, t in iterate(a.bag, read_topics):
        if topic == cloud_topic:
            c_hdr.append(stamp_sec(msg.header.stamp))
            c_rcv.append(t * 1e-9)
            frames.add(msg.header.frame_id)
            if fields_desc is None:
                fields_desc = {'fields': [(f.name, f.offset, f.datatype, f.count) for f in msg.fields],
                               'width': msg.width, 'height': msg.height, 'point_step': msg.point_step,
                               'is_dense': bool(msg.is_dense), 'frame_id': msg.header.frame_id}
            if k % a.sample_every == 0:
                xyz, ring, ppr, info = cloud_to_xyz(msg)
                st = cloud_to_struct(msg)
                npts.append(info['n_raw'])
                nvalid.append(info['n_valid'])
                r = np.linalg.norm(xyz[:, :2], axis=1)
                g = gr(xyz[(r > 0.8) & (r < 60)])
                tm = st['time'][np.isfinite(st['time'])] if 'time' in st.dtype.names else np.array([np.nan])
                samples.append({'t': c_rcv[-1] - rep['start_ns'] * 1e-9, 'n_valid': info['n_valid'],
                                'points_per_ring': ppr,
                                'range_p50': float(np.percentile(r, 50)), 'range_p95': float(np.percentile(r, 95)),
                                'range_max': float(r.max()),
                                'z_p5': float(np.percentile(xyz[:, 2], 5)),
                                'ground_d': float(g.plane.d), 'ground_tilt_deg': g.plane.tilt_deg,
                                'ground_normal': g.plane.normal.round(4).tolist(),
                                'ground_source': g.plane.source, 'ground_inliers': g.plane.inliers,
                                'obstacle_points': int(g.obstacle_mask.sum()),
                                'time_min': float(tm.min()), 'time_max': float(tm.max()),
                                'rings': int(len(np.unique(ring))) if ring is not None else None})
            k += 1
        elif topic == odom_topic:
            o_hdr.append(stamp_sec(msg.header.stamp))
            o_rcv.append(t * 1e-9)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            o_xy.append((p.x, p.y))
            o_z.append(p.z)
            o_yaw.append(_yaw((q.x, q.y, q.z, q.w)))
            if len(o_hdr) == 1:
                rep['odom_frames'] = (msg.header.frame_id, msg.child_frame_id)
        else:
            for tr in msg.transforms:
                tf_edges.add((topic, tr.header.frame_id, tr.child_frame_id))

    c_hdr, c_rcv = np.array(c_hdr), np.array(c_rcv)
    dts = np.diff(c_hdr)
    rep['cloud'] = {
        'count': int(len(c_hdr)), 'frames': sorted(frames), 'layout': fields_desc,
        'rate_hz': float(1.0 / np.median(dts)) if len(dts) else None,
        'dt_median': float(np.median(dts)) if len(dts) else None,
        'dt_std': float(np.std(dts)) if len(dts) else None,
        'dt_max': float(dts.max()) if len(dts) else None,
        'gaps_over_1.5dt': int((dts > 1.5 * np.median(dts)).sum()) if len(dts) else 0,
        'non_monotonic': int((dts <= 0).sum()),
        'recv_minus_header_median': float(np.median(c_rcv - c_hdr)),
        'points_raw_median': float(np.median(npts)), 'points_valid_median': float(np.median(nvalid)),
        'samples': samples}
    rep['tf'] = sorted(tf_edges)
    if o_hdr:
        o_hdr, o_rcv = np.array(o_hdr), np.array(o_rcv)
        off = o_rcv - o_hdr
        tcorr = o_hdr + np.median(off)
        xy = np.array(o_xy)
        yaw = np.array(o_yaw)
        segs = motion_segments(tcorr, xy, yaw)
        t0 = rep['start_ns'] * 1e-9
        rep['odom'] = {
            'count': int(len(o_hdr)), 'rate_hz': float(1.0 / np.median(np.diff(o_hdr))),
            'recv_minus_header_median': float(np.median(off)),
            'recv_minus_header_p5_p95': [float(np.percentile(off, 5)), float(np.percentile(off, 95))],
            'path_length_m': float(np.sum(np.hypot(*np.diff(xy, axis=0).T))),
            'z_range': [float(np.min(o_z)), float(np.max(o_z))],
            'yaw_range_deg': float(np.degrees(np.ptp(np.unwrap(yaw)))),
            'motion_segments_bag_time': [(s[0] - t0, s[1] - t0, s[2], s[3]) for s in segs]}
        # stationary intervals = complement of the motion segments
        bounds = [tcorr[0]] + [v for s in segs for v in (s[0], s[1])] + [tcorr[-1]]
        stat_iv = []
        for a0, a1 in zip(bounds[0::2], bounds[1::2]):
            m = (tcorr >= a0) & (tcorr <= a1)
            if a1 - a0 > 5.0 and m.sum() > 10:
                stat_iv.append((float(a0 - t0), float(a1 - t0),
                                float(np.degrees(np.ptp(np.unwrap(yaw[m])))),
                                float(np.max(np.hypot(*(xy[m] - xy[m][0]).T)))))
        rep['odom']['stationary_intervals'] = stat_iv

    out = a.out or os.path.join(os.getcwd(), 'DATASET_REPORT.md')
    with open(os.path.splitext(out)[0] + '.json', 'w') as f:
        json.dump(rep, f, indent=1, default=str)
    with open(out, 'w') as f:
        f.write(render(rep))
    print(render(rep))
    print(f'\nreport written to {out}')


def render(r: dict) -> str:
    L = ["# Dataset inspection report\n", "Generated by `inspect_bag.py` (read-only).\n",
         f"* bag: `{r['bag']}`", f"* storage: {r['storage']}, size {r['size_bytes'] / 2**30:.2f} GiB, "
         f"duration {r['duration_s']:.1f} s\n", "## Topics\n", "| topic | type | count |", "|---|---|---|"]
    for t, v in sorted(r['topics'].items()):
        L.append(f"| `{t}` | {v['type']} | {v['count']} |")
    c = r['cloud']
    lay = c['layout']
    L += ["", "## LiDAR", f"* topic `{r['cloud_topic']}`, frame(s) {c['frames']}",
          f"* layout width={lay['width']} height={lay['height']} point_step={lay['point_step']} "
          f"is_dense={lay['is_dense']}",
          f"* fields (name, offset, datatype, count): {lay['fields']}",
          f"* rate {c['rate_hz']:.3f} Hz, dt median {c['dt_median']*1e3:.2f} ms, std {c['dt_std']*1e3:.3f} ms, "
          f"max {c['dt_max']*1e3:.1f} ms, gaps>1.5dt: {c['gaps_over_1.5dt']}, non-monotonic: {c['non_monotonic']}",
          f"* receive - header stamp (median): {c['recv_minus_header_median']*1e3:.1f} ms",
          f"* points per scan: raw {c['points_raw_median']:.0f}, valid {c['points_valid_median']:.0f} (median of samples)"]
    s = c['samples']
    if s:
        def med(k):
            return float(np.median([x[k] for x in s if x[k] is not None]))
        L += [f"* range p50 {med('range_p50'):.1f} m, p95 {med('range_p95'):.1f} m, max {med('range_max'):.1f} m",
              f"* rings: {s[0]['rings']}, firings per ring per scan: {med('points_per_ring'):.0f} "
              f"-> angular density delta = {med('points_per_ring') / (2 * np.pi):.1f} points/rad/layer",
              f"* per-point time field range: {med('time_min')*1e3:.1f} .. {med('time_max')*1e3:.1f} ms",
              f"* RANSAC ground (sensor frame): d (sensor height) median {med('ground_d'):.3f} m, "
              f"tilt median {med('ground_tilt_deg'):.2f} deg "
              f"(min {min(x['ground_tilt_deg'] for x in s):.2f}, max {max(x['ground_tilt_deg'] for x in s):.2f}), "
              f"source counts {dict(zip(*np.unique([x['ground_source'] for x in s], return_counts=True)))}",
              f"* non-ground points per scan (median): {med('obstacle_points'):.0f}"]
    L += ["", "## TF", *(f"* `{e[0]}`: {e[1]} -> {e[2]}" for e in r['tf'])]
    if not any(e[0] == '/tf' for e in r['tf']):
        L.append("* **no dynamic /tf in the bag**")
    if 'odom' in r:
        o = r['odom']
        L += ["", "## Odometry", f"* topic `{r['odom_topic']}` frames {r.get('odom_frames')}, {o['rate_hz']:.1f} Hz",
              f"* receive - header stamp: median **{o['recv_minus_header_median']:+.3f} s** "
              f"(p5 {o['recv_minus_header_p5_p95'][0]:+.3f}, p95 {o['recv_minus_header_p5_p95'][1]:+.3f})",
              f"* path length {o['path_length_m']:.2f} m, z range {o['z_range'][0]:.2f}..{o['z_range'][1]:.2f} m, "
              f"yaw range {o['yaw_range_deg']:.1f} deg",
              "* stationary intervals (bag time s, yaw drift deg, max xy drift m): " + ', '.join(
                  f"{a:.1f}-{b:.1f} ({y:.2f} deg, {d:.3f} m)" for a, b, y, d in o['stationary_intervals']),
              "* motion segments (bag time, s): " + (', '.join(
                  f"{a:.1f}-{b:.1f} ({p:.1f} m, {y:+.0f} deg)" for a, b, p, y in o['motion_segments_bag_time'])
                  or 'none')]
    else:
        L += ["", "## Odometry", "* none in bag"]
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
