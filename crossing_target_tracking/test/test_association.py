"""Gating (Eq. 15) and GNN / Hungarian association (Sec. 3.3.2)."""
import numpy as np

from crossing_target_tracking.core.association import gnn_associate


def test_hungarian_is_global_not_greedy():
    # greedy (track 0 takes its nearest, measurement 0) would leave track 1 with cost 10
    cost = np.array([[1.0, 2.0],
                     [1.5, 10.0]])
    r = gnn_associate(cost, gate=100.0)
    assert sorted(r.matches) == [(0, 1), (1, 0)]       # total 3.5 < 11


def test_gate_rejects_and_reports_unmatched():
    cost = np.array([[1.0, 50.0, 3.0],
                     [40.0, 60.0, 70.0]])
    r = gnn_associate(cost, gate=9.21)
    assert r.matches == [(0, 0)]
    assert r.unmatched_tracks == [1]
    assert r.unmatched_measurements == [1, 2]


def test_empty_inputs():
    r = gnn_associate(np.zeros((0, 3)), 9.0)
    assert r.matches == [] and r.unmatched_measurements == [0, 1, 2]
    r = gnn_associate(np.zeros((2, 0)), 9.0)
    assert r.unmatched_tracks == [0, 1]


def test_nonfinite_cost_is_not_assigned():
    r = gnn_associate(np.array([[np.nan, 1.0], [np.inf, np.inf]]), 9.0)
    assert r.matches == [(0, 1)]
