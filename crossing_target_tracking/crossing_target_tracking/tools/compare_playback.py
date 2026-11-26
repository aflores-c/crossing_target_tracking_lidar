#!/usr/bin/env python3
"""Compare ROS-playback outputs (recorded with ros2 bag record) with offline CSVs.

  ros2 bag record -o rec /crossing_tracking/blobs /crossing_tracking/tracked_objects   # during playback
  ros2 run crossing_target_tracking compare_playback --recorded rec --offline <offline_out_dir>

Matches frames by LiDAR stamp and reports how many stamps are present in both,
the agreement of the per-frame blob count (blobs depend only on the scan, the
background history and the registration) and of the high-level track count.
Differences can come from scans dropped by the subscriber, from the odometry
clock-offset estimate (sim-time /clock vs bag receive time) and from the first
frames (odometry warm-up).
"""
import argparse
import csv
import json
import os
from collections import Counter

from crossing_target_tracking.tools.bag_io import iterate, open_reader, stamp_sec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--recorded', required=True)
    ap.add_argument('--offline', required=True)
    a = ap.parse_args(argv)
    off = {}
    for r in csv.DictReader(open(os.path.join(a.offline, 'frames.csv'))):
        off[round(float(r['stamp']), 4)] = (int(float(r['n_blobs'] or 0)), int(float(r['n_high_tracks'] or 0)),
                                            r['valid'] == '1')
    _, types = open_reader(a.recorded)
    topics = [t for t in ('/crossing_tracking/blobs', '/crossing_tracking/tracked_objects') if t in types]
    rec_b, rec_h = {}, {}
    for topic, msg, _ in iterate(a.recorded, topics):
        k = round(stamp_sec(msg.header.stamp), 4)
        if topic.endswith('blobs'):
            rec_b[k] = len(msg.blobs)
        else:
            rec_h[k] = len(msg.objects)
    common = sorted(set(rec_b) & set(off))
    eq_b = sum(rec_b[k] == off[k][0] for k in common)
    common_h = sorted(set(rec_h) & set(off))
    eq_h = sum(rec_h[k] == off[k][1] for k in common_h)
    first_diff = next((k for k in common if rec_b[k] != off[k][0]), None)
    res = {'offline_frames': len(off), 'recorded_frames': len(rec_b), 'common_frames': len(common),
           'blob_count_equal': eq_b, 'blob_count_equal_fraction': eq_b / len(common) if common else None,
           'high_count_equal_fraction': eq_h / len(common_h) if common_h else None,
           'blob_count_abs_diff_hist': dict(Counter(abs(rec_b[k] - off[k][0]) for k in common)),
           'first_differing_stamp': first_diff}
    print(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    main()
