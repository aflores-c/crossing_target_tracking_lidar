"""Registration: transform direction, odom interpolation + clock offset, rigid CPD (Sec. 3.1.2)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crossing_target_tracking.core.config import RegistrationConfig
from crossing_target_tracking.core.registration import (ClockOffsetEstimator, OdomBuffer, Registrar, T_from_xyz_rpy,
                                                       inv_T, rigid_cpd, transform_points, yaw_of)


def test_transform_direction_convention():
    # sensor 2 m ahead of base, base at (10, 0) rotated 90 deg: point 1 m ahead of the sensor
    T_ob = T_from_xyz_rpy([10, 0, 0], [0, 0, np.pi / 2])
    T_bs = T_from_xyz_rpy([2, 0, 0], [0, 0, 0])
    p = transform_points(T_ob @ T_bs, np.array([[1.0, 0, 0]]))
    np.testing.assert_allclose(p, [[10, 3, 0]], atol=1e-12)
    np.testing.assert_allclose(inv_T(T_ob) @ T_ob, np.eye(4), atol=1e-12)


def test_clock_offset_median_robust():
    est = ClockOffsetEstimator(window=100, min_samples=5)
    rng = np.random.default_rng(0)
    for k in range(50):
        hdr = 100.0 + 0.01 * k
        recv = hdr - 2.73 + abs(rng.normal(0, 0.003)) + (0.5 if k % 17 == 0 else 0)
        est.add(recv, hdr)
    assert est.offset == pytest.approx(-2.73, abs=0.01)


def test_odom_interpolation_with_offset():
    cfg = RegistrationConfig(odom_stamp_mode='auto_offset', odom_offset_min_samples=3)
    buf = OdomBuffer(cfg)
    for k in range(100):                       # header ahead by 2.7 s, robot moving 1 m/s, turning
        t_true = 10.0 + 0.01 * k
        q = Rotation.from_euler('z', 0.1 * (t_true - 10)).as_quat()
        buf.add(t_true + 2.7, t_true, (t_true - 10, 0, 0), q)
    T, d = buf.lookup(10.505)
    assert T is not None and 'odom_error' not in d
    assert T[0, 3] == pytest.approx(0.505, abs=1e-6)
    assert yaw_of(T) == pytest.approx(0.0505, abs=1e-6)
    T, d = buf.lookup(30.0)
    assert T is None and 'odom_error' in d     # no extrapolation far outside the buffer


def _scene(rng, n=600):
    walls = np.r_[np.c_[rng.uniform(-15, 15, n), np.full(n, 8.0), rng.uniform(0, 2, n)],
                  np.c_[np.full(n, -10.0), rng.uniform(-8, 8, n), rng.uniform(0, 2, n)],
                  np.c_[rng.uniform(2, 3, n // 3), rng.uniform(-4, -3, n // 3), rng.uniform(0, 1.5, n // 3)]]
    return walls


@pytest.mark.parametrize('planar', [False, True])
def test_rigid_cpd_recovers_known_motion(planar):
    rng = np.random.default_rng(1)
    X = _scene(rng)
    T_true = T_from_xyz_rpy([0.25, -0.1, 0.0], [0, 0, np.radians(4)])
    # current scan Y expressed in the moved sensor frame: X = T_true Y
    Y = transform_points(inv_T(T_true), X) + rng.normal(0, 0.02, X.shape)
    sel = rng.choice(len(X), 500, replace=False)
    res = rigid_cpd(X[sel], Y[rng.choice(len(Y), 500, replace=False)], w=0.1, max_iter=80, planar=planar)
    assert np.linalg.norm(res.T[:2, 3] - T_true[:2, 3]) < 0.05
    assert abs(yaw_of(res.T) - np.radians(4)) < np.radians(0.5)


def test_registrar_modes_and_stationary_check():
    cfg = RegistrationConfig(mode='stationary', odom_stamp_mode='header')
    reg = Registrar(cfg)
    for k in range(50):
        reg.add_odom(k * 0.01, None, (0.02 * k, 0, 0), (0, 0, 0, 1))
    r = reg.register(0.1, np.zeros((0, 3)), 'lidar')
    assert r.valid and r.method == 'stationary' and r.fixed_frame == 'lidar'
    r = reg.register(0.45, np.zeros((0, 3)), 'lidar')
    assert 'warning' in r.diag                 # moved 0.7 m: assumption violated and reported

    reg = Registrar(RegistrationConfig(mode='odom', odom_stamp_mode='header', extrinsic_xyz=[0, 0, 0]))
    assert not reg.register(1.0, np.zeros((0, 3))).valid      # no odometry -> invalid, not silent
    reg = Registrar(RegistrationConfig(mode='auto', auto_fallback='invalid', auto_odom_wait=0.5))
    assert reg.register(1.0, np.zeros((0, 3))).diag['odom_error'] == 'waiting for odometry'
    assert not reg.register(2.0, np.zeros((0, 3))).valid


def test_odom_jump_rejected():
    reg = Registrar(RegistrationConfig(mode='odom', odom_stamp_mode='header', extrinsic_xyz=[0, 0, 0],
                                       max_speed=3.0))
    for k in range(300):
        x = 0.0 if k < 150 else 5.0            # 5 m jump
        reg.add_odom(k * 0.01, None, (x, 0, 0), (0, 0, 0, 1))
    assert reg.register(1.0, np.zeros((0, 3))).valid
    r = reg.register(1.6, np.zeros((0, 3)))
    assert not r.valid and 'jump' in r.diag['odom_error']


def test_deskew_uses_per_point_pose():
    """Sensor turning at 1 rad/s: a static point observed at different times within one sweep."""
    from crossing_target_tracking.core.config import load_config
    from crossing_target_tracking.core.pipeline import TrackingPipeline
    cfg = load_config(None, ['registration.mode=odom', 'registration.odom_stamp_mode=header',
                             'registration.extrinsic_xyz=[0,0,0]'])
    p = TrackingPipeline(cfg)
    for k in range(200):
        t = k * 0.005
        p.add_odom(t, None, (0, 0, 0), Rotation.from_euler('z', 1.0 * t).as_quat())
    stamp = 0.6
    target = np.array([10.0, 0.0, 0.0])                    # static point in the fixed frame
    ptime = np.linspace(-0.1, 0.0, 11)
    pts = np.stack([Rotation.from_euler('z', -(stamp + dt)).apply(target) for dt in ptime])
    T = p.registrar.odom_pose(stamp)
    naive = transform_points(T, pts)
    desk, used = p._to_fixed(T, pts, ptime, stamp, 'odom')
    assert used
    assert np.max(np.linalg.norm(naive - target, axis=1)) > 0.9    # ~0.1 rad * 10 m
    assert np.max(np.linalg.norm(desk - target, axis=1)) < 0.06    # residual within one time bin
