"""Multi-target tracker = CT-EKF + Mahalanobis/GNN + M/N track management.

Used for both levels (paper Sec. 3.3 and 3.5):

  low level : z = [px, py]            H  (Eq. 13), M_c/N_c confirmation, M_e/N_e elimination
  high level: z = [px, py, vx, vy]    H2 (Eq. 18), NO confirmation (input clusters are built from
                                      confirmed tracks only), M_e/N_e elimination

M/N logic (Sec. 3.3.3):
  * a new track is TENTATIVE; it is confirmed as soon as it has received
    >= M_c measurements within its first N_c instants (the creating
    measurement counts as the first one), otherwise it is deleted;
  * a track is removed if it received < M_e measurements in the last N_e
    instants (evaluated once N_e instants are available).
An "instant" is a processed LiDAR frame. Dropped scans (time gap) and frames
with invalid registration are counted as instants without measurements.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

import numpy as np

from .association import gnn_associate
from .config import EKFConfig, TrackManagerConfig
from .ct_ekf import CTEKF, batch_mahalanobis2

TENTATIVE, CONFIRMED, PREDICTED = 0, 1, 2
STATUS_NAMES = {TENTATIVE: 'tentative', CONFIRMED: 'confirmed', PREDICTED: 'predicted'}


@dataclass
class Track:
    id: int
    ekf: CTEKF
    first_stamp: float
    last_stamp: float              # stamp the state refers to
    last_obs_stamp: float
    confirmed: bool
    hits: int = 1
    instants: int = 1
    consecutive_misses: int = 0
    flags: Deque[bool] = field(default_factory=deque)
    history: Deque[tuple] = field(default_factory=deque)
    updated: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def status(self) -> int:
        if not self.confirmed:
            return TENTATIVE
        return CONFIRMED if self.updated else PREDICTED

    @property
    def x(self):
        return self.ekf.x

    @property
    def P(self):
        return self.ekf.P

    @property
    def age(self) -> float:
        return self.last_stamp - self.first_stamp


@dataclass
class TrackerStepResult:
    matches: list
    created: List[int]
    deleted: List[int]
    n_gated_pairs: int


class MultiTargetTracker:
    def __init__(self, ekf_cfg: EKFConfig, mgr: TrackManagerConfig, H: np.ndarray, R: np.ndarray,
                 name: str = 'tracker'):
        self.ecfg = ekf_cfg
        self.mgr = mgr
        self.H = H
        self.R = R
        self.dim_z = H.shape[0]
        self.name = name
        self.tracks: List[Track] = []
        self.next_id = 1
        self.n_created = 0
        self.n_deleted = 0
        self.n_confirmed = 0
        self.last_stamp: Optional[float] = None

    # ── helpers
    def _new_track(self, z: np.ndarray, stamp: float, meta: Optional[dict]) -> Track:
        e = self.ecfg
        x0 = np.zeros(5)
        x0[:self.dim_z] = z
        P0 = np.diag([0.0, 0.0, e.init_sigma_v ** 2, e.init_sigma_v ** 2, e.init_sigma_omega ** 2])
        P0[:self.dim_z, :self.dim_z] = self.R
        ekf = CTEKF(x0, P0, e.sigma_a, e.sigma_omega, e.omega_max, e.omega_eps)
        maxlen = max(self.mgr.confirm_n, self.mgr.delete_n)
        tr = Track(self.next_id, ekf, stamp, stamp, stamp, not self.mgr.use_confirmation,
                   flags=deque([True], maxlen=maxlen), history=deque(maxlen=self.mgr.history_length),
                   meta=dict(meta or {}))
        if tr.confirmed:
            self.n_confirmed += 1
        self.next_id += 1
        self.n_created += 1
        return tr

    def _record(self, tr: Track, stamp: float):
        x = tr.ekf.x
        tr.history.append((stamp, x[0], x[1], x[2], x[3], x[4], tr.updated))

    def _miss(self, tr: Track):
        tr.instants += 1
        tr.flags.append(False)
        tr.consecutive_misses += 1
        tr.updated = False

    def _should_delete(self, tr: Track) -> bool:
        m = self.mgr
        if not tr.confirmed:
            remaining = m.confirm_n - tr.instants
            return tr.hits + max(remaining, 0) < m.confirm_m
        if len(tr.flags) >= m.delete_n:
            last = list(tr.flags)[-m.delete_n:]
            return sum(last) < m.delete_m
        return False

    # ── main step
    def step(self, stamp: float, Z: np.ndarray, metas: Optional[Sequence[dict]] = None,
             missed_instants: int = 0) -> TrackerStepResult:
        """Process one instant. Z: (K, dim_z) measurements (non-finite rows are dropped)."""
        Z = np.asarray(Z, float).reshape(-1, self.dim_z)
        metas = list(metas) if metas is not None else [{} for _ in range(len(Z))]
        finite = np.all(np.isfinite(Z), axis=1)
        Z = Z[finite]
        metas = [m for m, ok in zip(metas, finite) if ok]
        deleted: List[int] = []
        # dropped scans: instants without measurements before this one
        for _ in range(max(0, missed_instants)):
            for tr in self.tracks:
                self._miss(tr)
            keep = []
            for tr in self.tracks:
                (deleted.append(tr.id) if self._should_delete(tr) else keep.append(tr))
            self.tracks = keep
        # prediction to the current stamp
        for tr in self.tracks:
            tr.ekf.predict(stamp - tr.last_stamp)
            tr.last_stamp = stamp
        # gating + GNN (Eqs. 14-15)
        if self.tracks and len(Z):
            xs = np.stack([t.ekf.x for t in self.tracks])
            Ps = np.stack([t.ekf.P for t in self.tracks])
            cost = batch_mahalanobis2(xs, Ps, Z, self.H, self.R)
        else:
            cost = np.zeros((len(self.tracks), len(Z)))
        assoc = gnn_associate(cost, self.mgr.gate)
        n_gated = int((cost <= self.mgr.gate).sum())
        for j, i in assoc.matches:
            tr = self.tracks[j]
            ok = tr.ekf.update(Z[i], self.H, self.R)
            if not ok:
                self._miss(tr)
                continue
            tr.instants += 1
            tr.hits += 1
            tr.flags.append(True)
            tr.consecutive_misses = 0
            tr.updated = True
            tr.last_obs_stamp = stamp
            tr.meta = dict(metas[i])
            if not tr.confirmed and tr.hits >= self.mgr.confirm_m:
                tr.confirmed = True
                self.n_confirmed += 1
        for j in assoc.unmatched_tracks:
            self._miss(self.tracks[j])
        keep = []
        for tr in self.tracks:
            if self._should_delete(tr):
                deleted.append(tr.id)
            else:
                keep.append(tr)
        self.tracks = keep
        created = []
        for i in assoc.unmatched_measurements:
            tr = self._new_track(Z[i], stamp, metas[i])
            self.tracks.append(tr)
            created.append(tr.id)
        for tr in self.tracks:
            self._record(tr, stamp)
        self.n_deleted += len(deleted)
        self.last_stamp = stamp
        return TrackerStepResult(assoc.matches, created, deleted, n_gated)

    def confirmed_tracks(self, include_predicted: bool = True) -> List[Track]:
        return [t for t in self.tracks if t.confirmed and (include_predicted or t.updated)]
