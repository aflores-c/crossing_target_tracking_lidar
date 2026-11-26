"""Background model (Eqs. 7-9), morphology (Sec. 3.2.2) and shadow management (Sec. 3.2.3)."""
import cv2
import numpy as np

from crossing_target_tracking.core.bev_grid import GridSpec, SensorGeometry
from crossing_target_tracking.core.config import BackgroundConfig, ShadowConfig
from crossing_target_tracking.core.dynamic import DynamicExtractor


def setup(window=10, tau=0.4, shadow=True, **kw):
    s = GridSpec(0.2, 200, 200, 0.0, 0.0)
    geo = SensorGeometry(s)
    geo.update((0.0, 0.0))
    cfg = BackgroundConfig(window=window, tau_bck=tau, min_observed_frames=3, dilate_kernel=3,
                           close_kernel=kw.get('close', 3), min_blob_area=0.12, max_frame_gap=1.0)
    return s, geo, DynamicExtractor(s, cfg, ShadowConfig(enabled=shadow))


def rect(s, x0, x1, y0, y1):
    g = np.zeros(s.shape, np.uint8)
    i0, j0, _ = s.index(np.array([[x0, y0]]))
    i1, j1, _ = s.index(np.array([[x1, y1]]))
    g[j0[0]:j1[0] + 1, i0[0]:i1[0] + 1] = 1
    return g


def test_static_becomes_background_and_moving_is_dynamic():
    s, geo, dx = setup()
    wall = rect(s, -10, 10, 15, 15.4)
    for k in range(10):
        r = dx.process(wall, geo, 40.0, 0.1 * k)
    assert not r.blobs                                        # static wall is background (Eq. 8)
    car = rect(s, -2, 2, 5, 6.6)
    r = dx.process(wall | car, geo, 40.0, 1.0)
    assert len(r.blobs) == 1                                  # Eq. 9
    b = r.blobs[0]
    assert abs(b.x) < 0.2 and 5.5 < b.y < 6.1 and b.area > 4.0
    assert b.pixels.shape[1] == 2 and b.stamp == 1.0


def test_warmup_suppresses_detections():
    s, geo, dx = setup()
    r = dx.process(rect(s, -2, 2, 5, 6), geo, 40.0, 0.0)
    assert not r.blobs and r.warmup.any()


def test_square_closing_bridges_sparse_trace_and_area_filter():
    s, geo, dx = setup(close=3)
    for k in range(10):
        dx.process(np.zeros(s.shape, np.uint8), geo, 40.0, 0.1 * k)
    g = np.zeros(s.shape, np.uint8)
    j = s.index(np.array([[0, 8.0]]))[1][0]
    i0 = s.index(np.array([[-3.0, 0]]))[0][0]
    g[j, i0:i0 + 30:2] = 1                                    # every other cell (sparse car side)
    g[j + 40, 50] = 1                                         # isolated noise pixel
    r = dx.process(g, geo, 40.0, 1.0)
    assert len(r.blobs) == 1 and r.blobs[0].n_pixels >= 29    # one component, noise removed
    ell = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    assert ell[j, i0:i0 + 29].sum() < 29                       # why an elliptical element is not used


def test_dilated_background_removes_edge_noise():
    s, geo, dx = setup()
    wall = rect(s, -10, 10, 15, 15.4)
    for k in range(10):
        dx.process(wall, geo, 40.0, 0.1 * k)
    jitter = np.roll(wall, 1, axis=0)                         # wall measured one cell off
    r = dx.process(jitter, geo, 40.0, 1.0)
    assert not r.blobs


def _occlusion_sequence(shadow):
    """Paper Sec. 3.2.3 failure case.

    A wall (y = 14 m) is background. An occluder (2.5 m long, y = 4 m) passes at
    0.25 m/frame. Its cone hides the wall for ~9.5 frames (> (1 - tau) W = 8 ->
    without shadow management the wall leaves the background), while each lane
    cell is occupied ~10.8 frames (< tau W = 12 -> the occluder stays dynamic).
    Note: with tau_bck < 0.5 a single occluder can never cause the failure,
    because occlusion time <= lane-cell occupancy time (see PAPER_REPRODUCTION.md).
    """
    s, geo, dx = setup(window=20, tau=0.6, shadow=shadow)
    wall_x = np.arange(-0.2, 0.21, 0.2)
    wall = rect(s, -0.2, 0.2, 14.0, 14.2)
    t = 0.0
    for _ in range(20):
        dx.process(wall, geo, 40.0, t)
        t += 0.1
    xc = -6.0
    while xc < 1.5:                       # stop as soon as the wall is visible again
        occ = rect(s, xc - 1.25, xc + 1.25, 4.0, 4.4)
        lo, hi = np.arctan2(4.0, xc + 1.25), np.arctan2(4.0, xc - 1.25)
        b = np.arctan2(14.0, wall_x)
        visible = np.all((b < lo) | (b > hi))
        dx.process(occ | (wall if visible else 0), geo, 40.0, t)
        t += 0.1
        xc += 0.25
    return dx.process(wall, geo, 40.0, t)


def test_shadow_management_preserves_occluded_background():
    r_on = _occlusion_sequence(True)
    r_off = _occlusion_sequence(False)
    assert not r_on.blobs                 # held background: reappearing wall is not dynamic
    assert len(r_off.blobs) == 1          # without shadow management the wall pops up as a blob


def test_shadow_cone_geometry():
    s, geo, dx = setup()
    for k in range(10):
        dx.process(np.zeros(s.shape, np.uint8), geo, 40.0, 0.1 * k)
    r = dx.process(rect(s, -1, 1, 5, 5.4), geo, 40.0, 1.0)
    sh = r.shadow
    i, j, _ = s.index(np.array([[0, 12.0], [0, 3.0], [8, 12.0], [0, 5.2]]))
    assert sh[j[0], i[0]] and not sh[j[1], i[1]] and not sh[j[2], i[2]]   # behind / in front / beside
    assert not sh[j[3], i[3]]                                              # blob itself stays visible


def test_time_gap_resets_history():
    s, geo, dx = setup()
    wall = rect(s, -10, 10, 15, 15.4)
    for k in range(10):
        dx.process(wall, geo, 40.0, 0.1 * k)
    r = dx.process(wall, geo, 40.0, 5.0)
    assert r.reset and r.frames_in_history == 1
