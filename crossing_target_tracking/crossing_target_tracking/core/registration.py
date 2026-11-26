"""Scan registration into a common fixed frame (paper Sec. 3.1.2).

Conventions
-----------
T_a_b is a 4x4 homogeneous transform that maps coordinates expressed in
frame b into frame a:  p_a = T_a_b @ p_b.  The registration result of a scan
is T_fixed_sensor at the scan stamp.

Modes
-----
stationary : T_fixed_sensor = I (fixed frame == sensor frame). If odometry is
             available it is used only to *check* the assumption.
odom       : T_fixed_sensor(t) = T_odom_base(t) @ T_base_sensor, with the odom
             pose interpolated (linear + slerp) at the scan stamp after
             clock-offset correction of the odometry stamps.
cpd        : LiDAR odometry with rigid Coherent Point Drift
             (Myronenko & Song 2010) between consecutive non-ground scans,
             as in the paper; T_fixed_sensor(t) = T_fixed_sensor(t-1) @ T_prev_curr.
auto       : odom when a valid odom pose exists, otherwise `auto_fallback`.

Each result carries a `valid` flag and diagnostics. Invalid frames must not
be fed to the trackers (the pipeline treats them as missed instants).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .config import RegistrationConfig


# ── geometry helpers ───────────────────────────────────────────────────────

def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    return make_T(Rotation.from_euler('xyz', rpy).as_matrix(), xyz)


def T_from_xyz_quat(xyz, q_xyzw) -> np.ndarray:
    return make_T(Rotation.from_quat(q_xyzw).as_matrix(), xyz)


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return make_T(R.T, -R.T @ T[:3, 3])


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


# ── odometry clock offset ──────────────────────────────────────────────────

class ClockOffsetEstimator:
    """offset = median(receive_time - header_stamp) over a sliding window.

    Inspection of the Go2 bags showed /utlidar/robot_odom header stamps
    ~2.7 s ahead of the LiDAR clock (unsynchronised Unitree board).
    Same estimator as go2_semantic_perception/nodes/odom_tf_node.py, so the
    offline tool (receive time = bag time) and the node (receive time = sim
    clock) obtain the same correction.
    """

    def __init__(self, window: int = 300, min_samples: int = 20, jump_reset: float = 1.0):
        self.d: Deque[float] = deque(maxlen=window)
        self.min_samples = min_samples
        self.jump_reset = jump_reset
        self.offset: Optional[float] = None
        self.resets = 0

    def add(self, recv: float, header: float) -> Optional[float]:
        diff = recv - header
        if self.d and abs(diff - float(np.median(self.d))) > self.jump_reset:
            self.d.clear()
            self.resets += 1
        self.d.append(diff)
        if len(self.d) >= self.min_samples:
            self.offset = float(np.median(self.d))
        return self.offset


class OdomBuffer:
    """Time-ordered buffer of T_odom_base samples with interpolation."""

    def __init__(self, cfg: RegistrationConfig):
        self.cfg = cfg
        self.t: Deque[float] = deque()
        self.p: Deque[np.ndarray] = deque()
        self.q: Deque[np.ndarray] = deque()
        self.clock = ClockOffsetEstimator(cfg.odom_offset_window, cfg.odom_offset_min_samples)
        self.frame_id = ''
        self.child_frame_id = ''
        self.n_received = 0
        self.n_out_of_order = 0

    def add(self, header_stamp: float, recv_stamp: Optional[float], xyz, q_xyzw,
            frame_id: str = '', child_frame_id: str = ''):
        self.n_received += 1
        c = self.cfg
        if c.odom_stamp_mode == 'header':
            ts = header_stamp
        elif c.odom_stamp_mode == 'fixed_offset':
            ts = header_stamp + c.odom_fixed_offset
        else:
            if recv_stamp is None:
                raise ValueError('auto_offset needs the receive time')
            off = self.clock.add(recv_stamp, header_stamp)
            if off is None:
                return
            ts = header_stamp + off
        if self.t and ts <= self.t[-1]:
            self.n_out_of_order += 1
            return
        self.frame_id = frame_id or self.frame_id
        self.child_frame_id = child_frame_id or self.child_frame_id
        self.t.append(ts)
        self.p.append(np.asarray(xyz, float))
        self.q.append(np.asarray(q_xyzw, float))
        while self.t and self.t[-1] - self.t[0] > c.odom_buffer_s:
            self.t.popleft(); self.p.popleft(); self.q.popleft()

    @property
    def offset(self) -> Optional[float]:
        if self.cfg.odom_stamp_mode == 'header':
            return 0.0
        if self.cfg.odom_stamp_mode == 'fixed_offset':
            return self.cfg.odom_fixed_offset
        return self.clock.offset

    def available(self) -> bool:
        return len(self.t) > 0

    def lookup(self, stamp: float) -> Tuple[Optional[np.ndarray], dict]:
        """Return (T_odom_base at stamp or None, diagnostics)."""
        c = self.cfg
        diag = {'odom_samples': len(self.t)}
        if not self.t:
            diag['odom_error'] = 'no odom'
            return None, diag
        ts = np.fromiter(self.t, float, len(self.t))
        i = int(np.searchsorted(ts, stamp))
        if i == 0 or i == len(ts):
            j = 0 if i == 0 else len(ts) - 1
            gap = abs(stamp - ts[j])
            diag['odom_extrapolation'] = gap
            if gap > c.odom_max_extrapolation:
                diag['odom_error'] = f'stamp outside odom buffer by {gap:.3f}s'
                return None, diag
            return T_from_xyz_quat(self.p[j], self.q[j]), diag
        t0, t1 = ts[i - 1], ts[i]
        diag['odom_bracket'] = t1 - t0
        if t1 - t0 > c.odom_max_gap:
            diag['odom_error'] = f'odom gap {t1 - t0:.3f}s'
            return None, diag
        a = (stamp - t0) / (t1 - t0)
        p = (1 - a) * self.p[i - 1] + a * self.p[i]
        rots = Rotation.from_quat(np.stack([self.q[i - 1], self.q[i]]))
        R = Slerp([0.0, 1.0], rots)([a]).as_matrix()[0]
        return make_T(R, p), diag


# ── rigid CPD ──────────────────────────────────────────────────────────────

@dataclass
class CPDResult:
    T: np.ndarray                # maps source (current) points into target (previous) frame
    sigma2: float
    iterations: int
    matched_fraction: float
    converged: bool


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0 or voxel <= 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def rigid_cpd(X: np.ndarray, Y: np.ndarray, w: float = 0.2, max_iter: int = 40,
              tol: float = 1e-5, T0: Optional[np.ndarray] = None, planar: bool = False,
              match_dist: float = 0.4) -> CPDResult:
    """Rigid CPD (no scaling). X target (N,D), Y source (M,D); finds R,t with X ~ R Y + t.

    Myronenko & Song, "Point Set Registration: Coherent Point Drift", TPAMI 2010,
    Fig. 2 (rigid), with s = 1. `planar` restricts R to a rotation about z
    and t_z = 0 (useful for a ground vehicle with negligible pitch/roll).
    """
    N, D = X.shape
    M = Y.shape[0]
    T0 = np.eye(4) if T0 is None else T0
    R = T0[:3, :3].copy()
    t = T0[:3, 3].copy()
    TY = Y @ R.T + t
    xx = (X * X).sum(axis=1)

    def sqdist(TY):                                                     # (M, N), BLAS
        return np.maximum((TY * TY).sum(axis=1)[:, None] + xx[None, :] - 2.0 * TY @ X.T, 0.0)
    sigma2 = float(sqdist(TY).sum()) / (D * M * N)
    prev_q = np.inf
    it = 0
    converged = False
    P = None
    for it in range(1, max_iter + 1):
        d2 = sqdist(TY)
        Pm = np.exp(-d2 / (2.0 * sigma2))
        c = (2.0 * np.pi * sigma2) ** (D / 2.0) * w / (1.0 - w) * M / N
        den = Pm.sum(axis=0) + c
        P = Pm / den[None, :]                                            # (M, N)
        Np = P.sum()
        if Np < 1e-9:
            break
        P1 = P.sum(axis=1)
        Pt1 = P.sum(axis=0)
        mu_x = (Pt1 @ X) / Np
        mu_y = (P1 @ Y) / Np
        Xh = X - mu_x
        Yh = Y - mu_y
        A = Xh.T @ P.T @ Yh
        if planar:
            A2 = A[:2, :2]
            U, _, Vt = np.linalg.svd(A2)
            C = np.diag([1.0, np.linalg.det(U @ Vt)])
            R = np.eye(3)
            R[:2, :2] = U @ C @ Vt
        else:
            U, _, Vt = np.linalg.svd(A)
            C = np.eye(D)
            C[-1, -1] = np.linalg.det(U @ Vt)
            R = U @ C @ Vt
        t = mu_x - R @ mu_y
        if planar:
            t[2] = 0.0
        TY = Y @ R.T + t
        # s = 1 fixed, so the full objective is used (TPAMI 2010, Fig. 2 with s=1)
        q_val = float(np.trace(Xh.T * Pt1 @ Xh) - 2.0 * np.trace(A.T @ R)
                      + np.trace(Yh.T * P1 @ Yh))
        sigma2 = max(q_val / (Np * D), 1e-8)
        if abs(prev_q - q_val) < tol * max(1.0, abs(q_val)):
            converged = True
            break
        prev_q = q_val
    # quality: fraction of source points whose nearest target point (after alignment) is
    # closer than match_dist (the subsampling voxel); posterior-based measures are not
    # meaningful when sigma^2 collapses below the voxel size
    matched = float((np.sqrt(sqdist(TY).min(axis=1)) < match_dist).mean())
    return CPDResult(make_T(R, t), sigma2, it, matched, converged)


# ── registration front-end ─────────────────────────────────────────────────

@dataclass
class RegistrationResult:
    T_fixed_sensor: Optional[np.ndarray]
    valid: bool
    method: str
    fixed_frame: str
    diag: dict = field(default_factory=dict)


class Registrar:
    def __init__(self, cfg: RegistrationConfig):
        self.cfg = cfg
        self.odom = OdomBuffer(cfg)
        self.T_base_sensor = T_from_xyz_rpy(cfg.extrinsic_xyz, cfg.extrinsic_rpy)
        self.rng = np.random.default_rng(cfg.seed)
        self._cpd_prev_pts: Optional[np.ndarray] = None
        self._cpd_T: Optional[np.ndarray] = None
        self._cpd_prev_inc = np.eye(4)
        self._first_odom_T: Optional[np.ndarray] = None
        self._last_valid_T: Optional[np.ndarray] = None
        self._last_valid_t: Optional[float] = None
        self._last_odom = None
        self._first_scan_t: Optional[float] = None
        self.sensor_frame = ''

    # -- odom input
    def add_odom(self, header_stamp, recv_stamp, xyz, q_xyzw, frame_id='', child_frame_id=''):
        self.odom.add(header_stamp, recv_stamp, xyz, q_xyzw, frame_id, child_frame_id)

    def _fixed_frame_name(self, method: str) -> str:
        if method == 'stationary':
            return self.cfg.stationary_frame or self.sensor_frame or 'sensor'
        if method == 'odom':
            return self.odom.frame_id or self.cfg.fixed_frame
        return self.cfg.fixed_frame

    def _cpd_points(self, pts: np.ndarray, stamp: Optional[float] = None) -> np.ndarray:
        c = self.cfg
        if stamp is not None:
            self.rng = np.random.default_rng([c.seed, int(round(stamp * 1e6))])
        r = np.linalg.norm(pts[:, :2], axis=1)
        p = pts[(r > c.cpd_min_range) & (r < c.cpd_max_range)]
        if c.cpd_project_2d:
            # BEV projection: removes the ring-height sampling pattern of a 16-layer LiDAR,
            # which otherwise biases point-to-point registration on vertical surfaces
            p = np.c_[p[:, :2], np.zeros(len(p))]
        p = voxel_downsample(p, c.cpd_voxel)
        if len(p) > c.cpd_max_points:
            p = p[self.rng.choice(len(p), c.cpd_max_points, replace=False)]
        return p

    def _cpd_step(self, stamp: float, obstacle_pts: np.ndarray) -> RegistrationResult:
        c = self.cfg
        cur = self._cpd_points(obstacle_pts, stamp)
        if self._cpd_T is None:
            self._cpd_T = np.eye(4) if self._last_valid_T is None else self._last_valid_T.copy()
            self._cpd_prev_pts = cur
            return RegistrationResult(self._cpd_T.copy(), True, 'cpd', self._fixed_frame_name('cpd'),
                                      {'cpd_init': True, 'cpd_points': len(cur)})
        if len(cur) < 30 or len(self._cpd_prev_pts) < 30:
            self._cpd_prev_pts = cur
            return RegistrationResult(None, False, 'cpd', self._fixed_frame_name('cpd'),
                                      {'cpd_error': 'too few points', 'cpd_points': len(cur)})
        res = rigid_cpd(self._cpd_prev_pts, cur, w=c.cpd_w, max_iter=c.cpd_max_iter,
                        tol=c.cpd_tol, T0=self._cpd_prev_inc, planar=c.cpd_planar or c.cpd_project_2d,
                        match_dist=c.cpd_voxel)
        diag = {'cpd_sigma2': res.sigma2, 'cpd_iterations': res.iterations,
                'cpd_matched': res.matched_fraction, 'cpd_points': len(cur),
                'cpd_dx': float(res.T[0, 3]), 'cpd_dy': float(res.T[1, 3]),
                'cpd_dyaw_deg': float(np.degrees(yaw_of(res.T)))}
        dt = None if self._last_valid_t is None else stamp - self._last_valid_t
        step = float(np.linalg.norm(res.T[:2, 3]))
        ok = res.sigma2 <= c.cpd_max_sigma2 and res.matched_fraction >= c.cpd_min_matched
        if dt and dt > 0 and step / dt > c.max_speed:
            ok = False
            diag['cpd_error'] = f'implausible speed {step / dt:.2f} m/s'
        self._cpd_prev_pts = cur
        if not ok:
            diag.setdefault('cpd_error', 'quality gate')
            self._cpd_prev_inc = np.eye(4)
            return RegistrationResult(None, False, 'cpd', self._fixed_frame_name('cpd'), diag)
        self._cpd_prev_inc = res.T
        self._cpd_T = self._cpd_T @ res.T
        return RegistrationResult(self._cpd_T.copy(), True, 'cpd', self._fixed_frame_name('cpd'), diag)

    def _odom_step(self, stamp: float) -> RegistrationResult:
        T_ob, diag = self.odom.lookup(stamp)
        diag['odom_offset'] = self.odom.offset
        if T_ob is None:
            return RegistrationResult(None, False, 'odom', self._fixed_frame_name('odom'), diag)
        T = T_ob @ self.T_base_sensor
        prev = self._last_odom
        self._last_odom = (stamp, T)        # compare consecutive odom poses only
        if prev is not None:
            dt = stamp - prev[0]
            if 0 < dt < 1.0:
                v = np.linalg.norm(T[:2, 3] - prev[1][:2, 3]) / dt
                diag['ego_speed'] = float(v)
                if v > self.cfg.max_speed:
                    diag['odom_error'] = f'implausible ego speed {v:.2f} m/s (odom jump)'
                    return RegistrationResult(None, False, 'odom', self._fixed_frame_name('odom'), diag)
        return RegistrationResult(T, True, 'odom', self._fixed_frame_name('odom'), diag)

    def odom_pose(self, t: float) -> Optional[np.ndarray]:
        """T_fixed_sensor from odometry at time t (None if unavailable)."""
        T_ob, _ = self.odom.lookup(t)
        return None if T_ob is None else T_ob @ self.T_base_sensor

    def register(self, stamp: float, obstacle_pts_sensor: np.ndarray, sensor_frame: str = '') \
            -> RegistrationResult:
        self.sensor_frame = sensor_frame or self.sensor_frame
        mode = self.cfg.mode
        if mode == 'stationary':
            res = RegistrationResult(np.eye(4), True, 'stationary', self._fixed_frame_name('stationary'))
            T_ob, d = self.odom.lookup(stamp) if self.odom.available() else (None, {})
            if T_ob is not None:
                if self._first_odom_T is None:
                    self._first_odom_T = T_ob
                moved = float(np.linalg.norm(T_ob[:2, 3] - self._first_odom_T[:2, 3]))
                dyaw = abs(yaw_of(inv_T(self._first_odom_T) @ T_ob))
                res.diag['odom_moved_m'] = moved
                res.diag['odom_dyaw_deg'] = float(np.degrees(dyaw))
                if moved > self.cfg.stationary_check_translation:
                    res.diag['warning'] = 'stationary assumption violated by odometry'
        elif mode == 'odom':
            res = self._odom_step(stamp)
        elif mode == 'cpd':
            res = self._cpd_step(stamp, obstacle_pts_sensor)
        else:  # auto
            if self._first_scan_t is None:
                self._first_scan_t = stamp
            if self.odom.n_received > 0:
                # odometry exists: it is the only ego-motion source (no silent switching,
                # which would change the fixed frame mid-run)
                res = self._odom_step(stamp)
            elif stamp - self._first_scan_t < self.cfg.auto_odom_wait:
                res = RegistrationResult(None, False, 'odom', self._fixed_frame_name('odom'),
                                         {'odom_error': 'waiting for odometry'})
            elif self.cfg.auto_fallback == 'cpd':
                res = self._cpd_step(stamp, obstacle_pts_sensor)
                res.diag['auto'] = 'no odometry -> cpd'
            elif self.cfg.auto_fallback == 'stationary':
                res = RegistrationResult(np.eye(4), True, 'stationary', self._fixed_frame_name('stationary'),
                                         {'auto': 'no odometry -> stationary (UNVERIFIED assumption)'})
            else:
                res = RegistrationResult(None, False, 'odom', self._fixed_frame_name('odom'),
                                         {'odom_error': 'no odometry and auto_fallback=invalid'})
        if self.cfg.cpd_diagnostic and res.method == 'odom' and res.valid:
            prevT = self._last_valid_T
            cpd = self._cpd_diag(obstacle_pts_sensor, stamp)
            if cpd is not None and prevT is not None:
                inc_odom = inv_T(prevT) @ res.T_fixed_sensor
                res.diag['cpd_vs_odom_dxy'] = float(np.linalg.norm(cpd.T[:2, 3] - inc_odom[:2, 3]))
                res.diag['cpd_vs_odom_dyaw_deg'] = float(np.degrees(
                    abs(yaw_of(inv_T(inc_odom) @ cpd.T))))
                res.diag['cpd_sigma2'] = cpd.sigma2
        if res.valid:
            self._last_valid_T = res.T_fixed_sensor
            self._last_valid_t = stamp
        return res

    def _cpd_diag(self, pts, stamp=None):
        cur = self._cpd_points(pts, stamp)
        prev = self._cpd_prev_pts
        self._cpd_prev_pts = cur
        if prev is None or len(prev) < 30 or len(cur) < 30:
            return None
        c = self.cfg
        return rigid_cpd(prev, cur, w=c.cpd_w, max_iter=c.cpd_max_iter, tol=c.cpd_tol,
                         planar=c.cpd_planar or c.cpd_project_2d, match_dist=c.cpd_voxel)
