"""Ground removal with RANSAC plane fitting (paper Sec. 3.1.1).

The paper only states "RANSAC algorithm implemented for plane fitting".
The plane is fitted in the *sensor* frame and is NOT assumed horizontal:
any normal within `max_tilt_deg` of the sensor z axis is accepted, so a
sloped road or a pitched/rolled robot is handled. Hypotheses are evaluated
in one vectorised batch, followed by a least-squares (SVD) refinement.

Engineering additions (documented in PAPER_REPRODUCTION.md):
  * candidate pre-selection by a height window around a prior (robustness
    against walls / car sides dominating the RANSAC count),
  * temporal fallback to the previous plane, then to the prior plane,
  * optional height band [min_height, max_height] above the plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GroundConfig


@dataclass
class GroundPlane:
    normal: np.ndarray        # unit normal (3,), n_z > 0
    d: float                  # plane: n . p + d = 0
    inliers: int
    source: str               # 'ransac' | 'previous' | 'prior'

    def height(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.normal[2], -1.0, 1.0))))


@dataclass
class GroundResult:
    plane: GroundPlane
    ground_mask: np.ndarray      # points within distance_threshold of the plane
    obstacle_mask: np.ndarray    # non-ground points kept for the BEV grid


def _prior_plane(z_prior: float) -> GroundPlane:
    return GroundPlane(np.array([0.0, 0.0, 1.0]), -z_prior, 0, 'prior')


class GroundRemover:
    def __init__(self, cfg: GroundConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.prev: Optional[GroundPlane] = None

    def fit(self, pts: np.ndarray) -> GroundPlane:
        c = self.cfg
        fallback = self.prev if (c.temporal_fallback and self.prev is not None) else _prior_plane(c.z_prior)
        if fallback is self.prev:
            fallback = GroundPlane(self.prev.normal, self.prev.d, 0, 'previous')
        if len(pts) < 3:
            return fallback
        r = np.hypot(pts[:, 0], pts[:, 1])
        sel = r < c.candidate_max_range
        if c.z_window > 0:
            sel &= np.abs(pts[:, 2] - c.z_prior) < c.z_window
        cand = pts[sel]
        if len(cand) < max(3, c.min_inliers):
            return fallback
        if len(cand) > c.max_candidates:
            cand = cand[self.rng.choice(len(cand), c.max_candidates, replace=False)]
        k = c.iterations
        idx = self.rng.integers(0, len(cand), size=(k, 3))
        a, b, cc = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
        n = np.cross(b - a, cc - a)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-9
        n = n[ok] / nn[ok, None]
        a = a[ok]
        n *= np.sign(n[:, 2:3] + 1e-12)
        cos_max = np.cos(np.radians(c.max_tilt_deg))
        ok = n[:, 2] >= cos_max
        n, a = n[ok], a[ok]
        if len(n) == 0:
            return fallback
        d = -np.einsum('ij,ij->i', n, a)
        dist = np.abs(cand @ n.T + d[None, :])          # (Ncand, K)
        counts = (dist < c.distance_threshold).sum(axis=0)
        best = int(np.argmax(counts))
        if counts[best] < c.min_inliers:
            return fallback
        inl = cand[dist[:, best] < c.distance_threshold]
        # least-squares refinement
        cen = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
        nr = vt[2] * np.sign(vt[2][2] + 1e-12)
        if nr[2] < cos_max:
            nr = n[best]
        dr = -float(nr @ cen)
        cnt = int((np.abs(cand @ nr + dr) < c.distance_threshold).sum())
        if cnt < counts[best]:            # refinement worse: keep the RANSAC model
            nr, dr, cnt = n[best], float(d[best]), int(counts[best])
        return GroundPlane(nr, dr, cnt, 'ransac')

    def __call__(self, pts: np.ndarray, stamp: Optional[float] = None) -> GroundResult:
        c = self.cfg
        if stamp is not None:      # per-scan seed: identical result for the same scan online/offline
            self.rng = np.random.default_rng([c.seed, int(round(stamp * 1e6))])
        if not c.enabled:
            plane = _prior_plane(c.z_prior)
            h = plane.height(pts)
            keep = np.ones(len(pts), bool)
            if c.max_height is not None:
                keep &= h < c.max_height
            return GroundResult(plane, np.zeros(len(pts), bool), keep)
        plane = self.fit(pts)
        if plane.source == 'ransac':
            self.prev = plane
        h = plane.height(pts)
        ground = np.abs(h) < c.distance_threshold
        hmin = c.distance_threshold if c.min_height is None else c.min_height
        obstacle = h >= hmin
        if c.max_height is not None:
            obstacle &= h < c.max_height
        return GroundResult(plane, ground, obstacle)
