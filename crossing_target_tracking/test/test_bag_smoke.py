"""Offline processing smoke test on a real bag (skipped unless CTT_BAG is set).

  CTT_BAG="/media/cas/TOSHIBA EXT/bags/bag_go2_20261105_165413" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_bag_smoke.py -s
"""
import csv
import json
import os
import tempfile

import pytest

BAG = os.environ.get('CTT_BAG', '')
pytestmark = pytest.mark.skipif(not BAG or not os.path.isdir(BAG), reason='set CTT_BAG to a bag directory')


def test_offline_on_bag_is_deterministic():
    from crossing_target_tracking.tools import offline_process
    outs = []
    for k in range(2):
        out = tempfile.mkdtemp(prefix=f'ctt_smoke{k}_')
        s = offline_process.run(offline_process.build_parser().parse_args(
            ['--bag', BAG, '--start', os.environ.get('CTT_BAG_START', '40'), '--duration', '12', '--out', out]))
        assert s['frames'] > 100 and s['valid_frames'] >= s['frames'] - 2
        assert s['latency_ms']['total']['p95'] < 100.0           # keeps up with 10 Hz on this machine
        outs.append(out)
    rows = [[r for r in csv.reader(open(os.path.join(o, 'objects.csv')))] for o in outs]
    assert rows[0] == rows[1]                                      # identical tracks on re-run
    json.load(open(os.path.join(outs[0], 'summary.json')))
