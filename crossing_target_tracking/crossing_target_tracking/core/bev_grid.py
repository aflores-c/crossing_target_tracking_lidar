"""BEV grid quantisation and binarisation (paper Sec. 3.1.3-3.1.4, Eqs. 1-6).

Array convention
----------------
The grid B = {b_{n,m}} has N cells along x and M cells along y (both even)
of size gamma, centred on `center` (paper: the origin of the world-fixed
frame = the initial standing point).  Eq. (3) with 1-based n, m:

    n - 1 <= x/gamma + N/2 < n,     m - 1 <= y/gamma + M/2 < m

We use 0-based indices i = n - 1, j = m - 1, i.e.
    i = floor((x - cx)/gamma + N/2),   j = floor((y - cy)/gamma + M/2)
and store arrays as `grid[j, i]` (row = y, column = x), which is exactly
the row-major layout of nav_msgs/OccupancyGrid with
origin = (cx - N*gamma/2, cy - M*gamma/2).

Normalisation (Eqs. 4-5):  b_hat = b / b_bar,  b_bar = alpha(n,m) * delta
  alpha(n, m): horizontal angle subtended by the cell as seen from the
               sensor (exact, from its four corners; 2*pi for the cell that
               contains the sensor),
  delta:       angular point density of one LiDAR layer [points/rad].
Binarisation (Eq. 6): b_tilde = 1 if b_hat > tau_bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GridConfig


@dataclass
class GridSpec:
    resolution: float
    nx: int                 # N (cells along x)
    ny: int                 # M (cells along y)
    cx: float               # grid centre in the fixed frame [m]
    cy: float

    @staticmethod
    def from_config(cfg: GridConfig, cx: float, cy: float) -> 'GridSpec':
        nx = int(round(cfg.size_x / cfg.resolution))
        ny = int(round(cfg.size_y / cfg.resolution))
        nx += nx % 2
        ny += ny % 2
        return GridSpec(cfg.resolution, nx, ny, cx, cy)

    @property
    def origin(self):
        return (self.cx - self.nx * self.resolution / 2.0, self.cy - self.ny * self.resolution / 2.0)

    @property
    def shape(self):
        return (self.ny, self.nx)

    def index(self, xy: np.ndarray):
        """Eq. (3), 0-based. Returns (i, j, inside)."""
        g = self.resolution
        i = np.floor((xy[:, 0] - self.cx) / g + self.nx / 2).astype(np.int64)
        j = np.floor((xy[:, 1] - self.cy) / g + self.ny / 2).astype(np.int64)
        inside = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return i, j, inside

    def cell_centers(self):
        ox, oy = self.origin
        g = self.resolution
        xs = ox + (np.arange(self.nx) + 0.5) * g
        ys = oy + (np.arange(self.ny) + 0.5) * g
        return xs, ys

    def to_metric(self, j: np.ndarray, i: np.ndarray):
        """Centre of cell (row j, column i) in the fixed frame."""
        ox, oy = self.origin
        g = self.resolution
        return ox + (np.asarray(i) + 0.5) * g, oy + (np.asarray(j) + 0.5) * g


def quantize(spec: GridSpec, xy: np.ndarray) -> np.ndarray:
    """b_{n,m}: number of points in each vertical column (Eq. 3). int32 (ny, nx)."""
    i, j, ins = spec.index(xy)
    flat = j[ins] * spec.nx + i[ins]
    return np.bincount(flat, minlength=spec.nx * spec.ny).reshape(spec.shape).astype(np.int32)


class SensorGeometry:
    """Per-cell polar quantities w.r.t. the sensor position (cached).

    alpha  : angular coverage of each cell [rad]  (Eq. 5)
    theta  : bearing of each cell centre [rad]
    rng    : range of each cell centre [m]
    """

    def __init__(self, spec: GridSpec, recompute_dist: float = 0.02):
        self.spec = spec
        self.recompute_dist = recompute_dist
        self.sensor_xy: Optional[np.ndarray] = None
        self.alpha = self.theta = self.rng = None
        self.n_recomputed = 0

    def update(self, sensor_xy) -> bool:
        s = np.asarray(sensor_xy, float)[:2]
        if self.sensor_xy is not None and np.hypot(*(s - self.sensor_xy)) < self.recompute_dist:
            return False
        self.sensor_xy = s
        spec = self.spec
        ox, oy = spec.origin
        g = spec.resolution
        xe = ox + np.arange(spec.nx + 1) * g - s[0]
        ye = oy + np.arange(spec.ny + 1) * g - s[1]
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        th = np.arctan2(yc[:, None], xc[None, :])                 # (ny, nx)
        corners = np.arctan2(ye[:, None], xe[None, :])           # (ny+1, nx+1)
        d = np.stack([corners[:-1, :-1], corners[:-1, 1:], corners[1:, :-1], corners[1:, 1:]])
        d = np.angle(np.exp(1j * (d - th[None])))                # wrap to (-pi, pi]
        alpha = d.max(axis=0) - d.min(axis=0)
        # the cell containing the sensor sees all directions
        inside = (xe[:-1][None, :] <= 0) & (xe[1:][None, :] > 0) & \
                 (ye[:-1][:, None] <= 0) & (ye[1:][:, None] > 0)
        alpha[inside] = 2.0 * np.pi
        self.alpha = alpha
        self.theta = th
        self.rng = np.hypot(yc[:, None], xc[None, :])
        self.n_recomputed += 1
        return True


def binarize(counts: np.ndarray, geom: SensorGeometry, cfg: GridConfig, delta: float):
    """Eqs. (4)-(6). Returns (b_tilde uint8, b_hat float32 or None)."""
    if cfg.normalization == 'raw_count':        # ablation baseline, NOT the paper
        return (counts >= cfg.raw_count_threshold).astype(np.uint8), None
    b_bar = geom.alpha * delta                   # Eq. (5)
    b_hat = counts / b_bar                       # Eq. (4)
    return (b_hat > cfg.tau_bin).astype(np.uint8), b_hat.astype(np.float32)   # Eq. (6)


def estimate_angular_density(n_points_per_ring: Optional[float]) -> float:
    """delta = points per revolution per layer / (2 pi)."""
    if n_points_per_ring is None or n_points_per_ring <= 0:
        raise ValueError('cannot estimate angular density')
    return float(n_points_per_ring) / (2.0 * np.pi)
