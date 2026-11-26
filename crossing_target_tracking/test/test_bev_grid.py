"""Grid quantisation (Eq. 3) and angular normalisation / binarisation (Eqs. 4-6)."""
import numpy as np
import pytest

from crossing_target_tracking.core import synth as S
from crossing_target_tracking.core.bev_grid import GridSpec, SensorGeometry, binarize, quantize
from crossing_target_tracking.core.config import GridConfig


def spec(n=40, g=0.5):
    return GridSpec(g, n, n, 0.0, 0.0)


def test_eq3_indexing_and_layout():
    s = spec(4, 1.0)                      # cells cover [-2, 2) on both axes
    xy = np.array([[-2.0, -2.0], [-0.01, 0.0], [1.99, 1.5], [2.0, 0.0], [0.5, -1.5]])
    i, j, ins = s.index(xy)
    assert list(i[:3]) == [0, 1, 3] and list(j[:3]) == [0, 2, 3]
    assert not ins[3]                     # x = 2 is outside (strict upper bound)
    c = quantize(s, xy)
    assert c.shape == (4, 4) and c.sum() == 4
    assert c[0, 2] == 1                   # row = y index, column = x index


def test_alpha_angular_coverage():
    s = spec(40, 0.5)
    g = SensorGeometry(s)
    g.update((0.0, 0.0))
    # cell containing the sensor sees 2 pi
    i, j, _ = s.index(np.array([[0.1, 0.1]]))
    assert g.alpha[j[0], i[0]] == pytest.approx(2 * np.pi)
    # cell centred at (7.25, 0.25): extent from its corners
    i, j, _ = s.index(np.array([[7.25, 0.25]]))
    exp = np.arctan2(0.5, 7.0) - np.arctan2(0.0, 7.5)
    assert g.alpha[j[0], i[0]] == pytest.approx(exp, rel=1e-9)
    # alpha decreases ~1/r
    i2, j2, _ = s.index(np.array([[3.25, 0.25]]))
    assert g.alpha[j2[0], i2[0]] > 2 * g.alpha[j[0], i[0]]
    # cell straddling the -x axis (angle wrap)
    i, j, _ = s.index(np.array([[-7.25, -0.25]]))
    assert 0 < g.alpha[j[0], i[0]] < 0.2


def test_normalisation_makes_detection_range_independent():
    """Same wall at 6 m and 28 m: normalised occupancy is similar; raw counts are not (Eqs. 4-6)."""
    sm = S.SensorModel(range_noise=0.0)
    boxes = [S.Box(6.0, 0.0, 0.0, 0.4, 4.0, 2.0), S.Box(-28.0, 0.0, 0.0, 0.4, 4.0, 2.0)]
    pts, ring, ppr = S.ray_cast(sm, boxes)
    pts = pts[pts[:, 2] > -0.8]           # drop ground hits
    s = GridSpec(0.2, 400, 400, 0.0, 0.0)
    geo = SensorGeometry(s)
    geo.update((0, 0))
    counts = quantize(s, pts[:, :2])
    delta = ppr / (2 * np.pi)
    _, bhat = binarize(counts, geo, GridConfig(tau_bin=0.5), delta)
    near = (s.cell_centers()[0] > 5) & (s.cell_centers()[0] < 7)
    far = (s.cell_centers()[0] < -27) & (s.cell_centers()[0] > -29)
    rows = slice(*s.index(np.array([[0, -1.5], [0, 1.5]]))[1])
    occ_near = bhat[rows][:, near].max(axis=1)
    occ_far = bhat[rows][:, far].max(axis=1)
    raw_near = counts[rows][:, near].max(axis=1)
    raw_far = counts[rows][:, far].max(axis=1)
    raw_ratio = raw_near.mean() / raw_far.mean()
    norm_ratio = occ_near.mean() / occ_far.mean()
    # Eqs. (4)-(5) remove the horizontal (1/r) density decay; what remains is the number
    # of layers hitting the wall (Eq. 5 models a single layer, as the paper notes)
    assert raw_ratio > 3.0 * norm_ratio
    rings_ratio = 10.0 / 2.5                                      # ~layers on a 2 m wall at 6 m vs 28 m
    assert 0.5 * rings_ratio < norm_ratio < 2.0 * rings_ratio
    bt, _ = binarize(counts, geo, GridConfig(tau_bin=0.5), delta)
    bt_raw, _ = binarize(counts, geo, GridConfig(normalization='raw_count', raw_count_threshold=8), delta)
    assert bt[rows][:, far].any(axis=1).mean() > 0.9              # far wall still occupied
    assert bt_raw[rows][:, far].any(axis=1).mean() < 0.5          # fixed count threshold loses it
