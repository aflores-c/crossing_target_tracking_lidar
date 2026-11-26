"""Complete per-scan pipeline (shared by the ROS node and the offline tool).

    scan (sensor frame)
      -> range filter -> RANSAC ground removal (3.1.1)
      -> registration T_fixed_sensor (3.1.2)
      -> BEV quantisation + angular normalisation + binarisation (3.1.3-3.1.4)
      -> background / shadow / morphology -> blobs (3.2)
      -> low-level CT-EKF + GNN + M/N on blob centroids (3.3)
      -> position-velocity clustering of confirmed tracks (3.4)
      -> high-level CT-EKF + GNN + M/N on clusters (3.5)

No ROS imports: unit-testable with NumPy arrays.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .bev_grid import GridSpec, SensorGeometry, binarize, estimate_angular_density, quantize
from .clustering import Cluster, cluster_tracks
from .config import Config
from .ct_ekf import H_POS, H_POS_VEL
from .dynamic import Blob, DynamicExtractor, DynamicResult
from .ground import GroundRemover, GroundResult
from .registration import Registrar, RegistrationResult, transform_points
from .tracker import MultiTargetTracker, Track

VLP16_POINTS_PER_RING_10HZ = 1800   # fallback only (0.2 deg at 600 rpm)


@dataclass
class Scan:
    stamp: float                         # LiDAR header stamp [s]
    xyz: np.ndarray                      # (N, 3) sensor frame, finite
    frame_id: str = ''
    ring: Optional[np.ndarray] = None
    points_per_ring: Optional[float] = None   # firings per revolution (incl. no-return)
    point_time: Optional[np.ndarray] = None   # per-point time offset w.r.t. stamp [s] (deskew)


@dataclass
class TrackSnapshot:
    id: int
    status: int
    x: np.ndarray
    P: np.ndarray
    first_stamp: float
    last_obs_stamp: float
    hits: int
    consecutive_misses: int
    member_ids: List[int]
    history: List[tuple]

    @staticmethod
    def of(t: Track, with_history: bool) -> 'TrackSnapshot':
        return TrackSnapshot(t.id, t.status, t.x.copy(), t.P.copy(), t.first_stamp, t.last_obs_stamp,
                             t.hits, t.consecutive_misses, list(t.meta.get('member_ids', [])),
                             list(t.history) if with_history else [])

    @property
    def age(self) -> float:
        return (self.history[-1][0] if self.history else self.last_obs_stamp) - self.first_stamp


@dataclass
class FrameResult:
    stamp: float
    index: int
    valid: bool
    reason: str
    fixed_frame: str
    sensor_frame: str
    T_fixed_sensor: Optional[np.ndarray]
    registration: Optional[RegistrationResult]
    ground: Optional[GroundResult]
    obstacle_points_fixed: Optional[np.ndarray]
    ground_points_sensor: Optional[np.ndarray]
    spec: Optional[GridSpec]
    btilde: Optional[np.ndarray]
    dyn: Optional[DynamicResult]
    blobs: List[Blob]
    low_tracks: List[TrackSnapshot]
    clusters: List[Cluster]
    high_tracks: List[TrackSnapshot]
    missed_instants: int
    timings: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, float] = field(default_factory=dict)


class TrackingPipeline:
    def __init__(self, cfg: Config, keep_history: bool = True):
        self.cfg = cfg
        self.keep_history = keep_history
        self.ground = GroundRemover(cfg.ground)
        self.registrar = Registrar(cfg.registration)
        self.spec: Optional[GridSpec] = None
        self.geom: Optional[SensorGeometry] = None
        self.dyn: Optional[DynamicExtractor] = None
        self.delta: Optional[float] = None
        self.delta_source = ''
        self.fixed_frame: Optional[str] = None
        lo = cfg.low_ekf
        self.low = MultiTargetTracker(lo, cfg.low_manager, H_POS, np.eye(2) * lo.sigma_r ** 2, 'low')
        hi = cfg.high
        R2 = np.diag([hi.sigma_r_pos ** 2] * 2 + [hi.sigma_r_vel ** 2] * 2)
        self.high = MultiTargetTracker(hi.ekf, hi.manager, H_POS_VEL, R2, 'high')
        self.n_frames = 0
        self.n_invalid = 0
        self.n_missed_total = 0
        self._dts: deque = deque(maxlen=50)
        self._last_stamp: Optional[float] = None

    # ── inputs
    def add_odom(self, header_stamp, recv_stamp, xyz, q_xyzw, frame_id='', child_frame_id=''):
        self.registrar.add_odom(header_stamp, recv_stamp, xyz, q_xyzw, frame_id, child_frame_id)

    # ── helpers
    def _missed(self, stamp: float) -> int:
        if self._last_stamp is None:
            return 0
        dt = stamp - self._last_stamp
        nominal = float(np.median(self._dts)) if len(self._dts) >= 5 else None
        if nominal is not None and nominal > 0 and dt > 1.5 * nominal:
            missed = int(round(dt / nominal)) - 1
        else:
            missed = 0
            if dt > 0:
                self._dts.append(dt)
        return max(0, missed)

    def _angular_density(self, scan: Scan) -> float:
        if self.delta is not None:
            return self.delta
        d = self.cfg.grid.angular_density
        if d != 'auto':
            self.delta, self.delta_source = float(d), 'config'
        elif scan.points_per_ring:
            self.delta = estimate_angular_density(scan.points_per_ring)
            self.delta_source = 'cloud_layout'
        elif scan.ring is not None and len(scan.ring):
            counts = np.bincount(scan.ring.astype(np.int64))
            self.delta = estimate_angular_density(counts.max())
            self.delta_source = 'max_points_per_ring (lower bound)'
        else:
            self.delta = estimate_angular_density(VLP16_POINTS_PER_RING_10HZ)
            self.delta_source = 'VLP-16 10 Hz nominal (fallback)'
        return self.delta

    def _setup_grid(self, T_fs: np.ndarray, fixed_frame: str):
        g = self.cfg.grid
        if g.origin_mode == 'first_pose':
            cx, cy = float(T_fs[0, 3]), float(T_fs[1, 3])
        else:
            cx, cy = float(g.origin_xy[0]), float(g.origin_xy[1])
        self.spec = GridSpec.from_config(g, cx, cy)
        self.geom = SensorGeometry(self.spec)
        self.dyn = DynamicExtractor(self.spec, self.cfg.background, self.cfg.shadow)
        self.fixed_frame = fixed_frame

    def _to_fixed(self, T, pts, ptime, stamp, method):
        """Transform to the fixed frame; with odometry, each point uses the pose at its own time."""
        rc = self.cfg.registration
        if not (rc.deskew and method == 'odom' and ptime is not None and len(pts)):
            return transform_points(T, pts), False
        lo, hi = float(ptime.min()), float(ptime.max())
        if hi - lo < 1e-3:
            return transform_points(T, pts), False
        nb = max(1, rc.deskew_bins)
        edges = np.linspace(lo, hi + 1e-9, nb + 1)
        b = np.clip(np.searchsorted(edges, ptime, side='right') - 1, 0, nb - 1)
        out = np.empty_like(pts)
        for k in range(nb):
            m = b == k
            if not m.any():
                continue
            Tk = self.registrar.odom_pose(stamp + 0.5 * (edges[k] + edges[k + 1]))
            out[m] = transform_points(T if Tk is None else Tk, pts[m])
        return out, True

    def _snap(self, tracker: MultiTargetTracker, only_confirmed: bool) -> List[TrackSnapshot]:
        return [TrackSnapshot.of(t, self.keep_history) for t in tracker.tracks
                if (t.confirmed or not only_confirmed)]

    # ── main
    def process(self, scan: Scan) -> FrameResult:
        cfg = self.cfg
        tm: Dict[str, float] = {}
        t_all = time.perf_counter()
        idx = self.n_frames
        self.n_frames += 1
        missed = self._missed(scan.stamp)
        self.n_missed_total += missed
        self._last_stamp = scan.stamp

        t0 = time.perf_counter()
        xyz = scan.xyz
        r = np.linalg.norm(xyz[:, :2], axis=1)
        sel = (r >= cfg.input.min_range) & (r <= cfg.input.max_range)
        xyz = xyz[sel]
        ptime = scan.point_time[sel] if scan.point_time is not None else None
        gr = self.ground(xyz, scan.stamp)
        obst = xyz[gr.obstacle_mask]
        obst_time = ptime[gr.obstacle_mask] if ptime is not None else None
        tm['ground'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        reg = self.registrar.register(scan.stamp, obst, scan.frame_id)
        tm['registration'] = time.perf_counter() - t0
        counters = {'n_points': float(len(scan.xyz)), 'n_obstacle_points': float(len(obst)),
                    'ground_inliers': float(gr.plane.inliers), 'ground_tilt_deg': gr.plane.tilt_deg,
                    'ground_d': float(gr.plane.d)}
        reason = ''
        valid = reg.valid
        if valid and self.fixed_frame is not None and reg.fixed_frame != self.fixed_frame:
            valid, reason = False, f'fixed frame changed {self.fixed_frame} -> {reg.fixed_frame}'
        if not valid:
            self.n_invalid += 1
            reason = reason or 'registration invalid: ' + str(
                {k: v for k, v in reg.diag.items() if 'error' in k})
            self.low.step(scan.stamp, np.zeros((0, 2)), missed_instants=missed)
            self.high.step(scan.stamp, np.zeros((0, 4)), missed_instants=missed)
            tm['total'] = time.perf_counter() - t_all
            return FrameResult(scan.stamp, idx, False, reason, self.fixed_frame or reg.fixed_frame,
                               scan.frame_id, None, reg, gr, None, xyz[gr.ground_mask], self.spec,
                               None, None, [], self._snap(self.low, False), [],
                               self._snap(self.high, False), missed, tm, counters)

        T = reg.T_fixed_sensor
        if self.spec is None:
            self._setup_grid(T, reg.fixed_frame)

        t0 = time.perf_counter()
        obst_f, deskewed = self._to_fixed(T, obst, obst_time, scan.stamp, reg.method)
        counters['deskewed'] = float(deskewed)
        sensor_xy = T[:2, 3]
        self.geom.update(sensor_xy)
        counts = quantize(self.spec, obst_f[:, :2])
        delta = self._angular_density(scan)
        btilde, _ = binarize(counts, self.geom, cfg.grid, delta)
        tm['grid'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dres = self.dyn.process(btilde, self.geom, cfg.grid.roi_max_range, scan.stamp)
        tm['dynamic'] = time.perf_counter() - t0
        if dres.reset:
            reason = 'background reset after time gap'

        t0 = time.perf_counter()
        Z1 = np.array([[b.x, b.y] for b in dres.blobs]).reshape(-1, 2)
        self.low.step(scan.stamp, Z1, [{'blob_id': b.id} for b in dres.blobs], missed)
        tm['low_level'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        conf = self.low.confirmed_tracks(include_predicted=cfg.clustering.include_predicted)
        if conf:
            X = np.stack([t.x for t in conf])
            # per-scan seeded generator: reproducible random seed selection (Sec. 3.4)
            rng = np.random.default_rng([cfg.clustering.seed, int(round(scan.stamp * 1e6))])
            clusters = cluster_tracks(X[:, :2], X[:, 2:4], X[:, 4], [t.id for t in conf],
                                      cfg.clustering, rng)
        else:
            clusters = []
        tm['clustering'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        Z2 = np.array([np.r_[c.position, c.velocity] for c in clusters]).reshape(-1, 4)
        self.high.step(scan.stamp, Z2, [{'member_ids': c.member_ids, 'cluster_id': c.id}
                                        for c in clusters], missed)
        tm['high_level'] = time.perf_counter() - t0
        tm['total'] = time.perf_counter() - t_all

        counters.update({'n_blobs': float(len(dres.blobs)), 'n_clusters': float(len(clusters)),
                         'n_low_tracks': float(len(self.low.tracks)),
                         'n_low_confirmed': float(len(self.low.confirmed_tracks())),
                         'n_high_tracks': float(len(self.high.tracks)),
                         'frames_in_history': float(dres.frames_in_history),
                         'angular_density': float(delta)})
        counters.update({k: float(v) for k, v in dres.stats.items()})
        return FrameResult(scan.stamp, idx, True, reason, self.fixed_frame, scan.frame_id, T, reg, gr,
                           obst_f, xyz[gr.ground_mask], self.spec, btilde, dres, dres.blobs,
                           self._snap(self.low, False), clusters, self._snap(self.high, False),
                           missed, tm, counters)
