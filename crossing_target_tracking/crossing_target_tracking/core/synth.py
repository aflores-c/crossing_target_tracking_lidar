"""Synthetic spinning-LiDAR scenes for deterministic tests.

Ray-casts a VLP-16-like sensor (16 layers -15..+15 deg, configurable
azimuth resolution) against a flat ground plane z = 0 and oriented boxes.
Ground truth (box centres and velocities) is known exactly.

World frame: z up, ground at z = 0. The sensor is at (sx, sy, height) with
yaw `syaw`; returned points are in the sensor frame (x forward, z up).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class Box:
    cx: float
    cy: float
    yaw: float
    length: float
    width: float
    height: float
    z0: float = 0.0


@dataclass
class SensorModel:
    elevations_deg: Sequence[float] = tuple(np.arange(-15.0, 15.1, 2.0))
    n_azimuth: int = 1800
    max_range: float = 100.0
    min_range: float = 0.5
    range_noise: float = 0.01
    height: float = 1.0


def ray_directions(sm: SensorModel) -> np.ndarray:
    el = np.radians(np.asarray(sm.elevations_deg))
    az = np.linspace(0.0, 2 * np.pi, sm.n_azimuth, endpoint=False)
    E, A = np.meshgrid(el, az, indexing='ij')
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1)
    return d.reshape(-1, 3)


def ray_cast(sm: SensorModel, boxes: List[Box], sensor_xy=(0.0, 0.0), sensor_yaw: float = 0.0,
             rng: Optional[np.random.Generator] = None, dirs: Optional[np.ndarray] = None):
    """Return (points in sensor frame (N,3), ring (N,), points_per_ring)."""
    D = ray_directions(sm) if dirs is None else dirs
    n_el = len(sm.elevations_deg)
    ring = np.repeat(np.arange(n_el), sm.n_azimuth)
    cy, sy = np.cos(sensor_yaw), np.sin(sensor_yaw)
    Rw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Dw = D @ Rw.T
    o = np.array([sensor_xy[0], sensor_xy[1], sm.height])
    t = np.full(len(D), np.inf)
    down = Dw[:, 2] < -1e-9
    t[down] = -o[2] / Dw[down, 2]
    for b in boxes:
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        # into box frame
        ox, oy = o[0] - b.cx, o[1] - b.cy
        lo = np.array([c * ox + s * oy, -s * ox + c * oy, o[2] - b.z0])
        ld = np.stack([c * Dw[:, 0] + s * Dw[:, 1], -s * Dw[:, 0] + c * Dw[:, 1], Dw[:, 2]], axis=1)
        half = np.array([b.length / 2, b.width / 2])
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / ld
            t1 = np.stack([(-half[0] - lo[0]) * inv[:, 0], (-half[1] - lo[1]) * inv[:, 1],
                           (0.0 - lo[2]) * inv[:, 2]], 1)
            t2 = np.stack([(half[0] - lo[0]) * inv[:, 0], (half[1] - lo[1]) * inv[:, 1],
                           (b.height - lo[2]) * inv[:, 2]], 1)
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmin <= tmax) & (tmax > 0) & (tmin > 0)
        t = np.where(hit & (tmin < t), tmin, t)
    ok = np.isfinite(t) & (t < sm.max_range) & (t > sm.min_range)
    rng = rng or np.random.default_rng(0)
    tt = t[ok] + rng.normal(0, sm.range_noise, ok.sum())
    pts = D[ok] * tt[:, None]
    return pts, ring[ok], float(sm.n_azimuth)


@dataclass
class SynthObject:
    """Moving box: state(t) -> (cx, cy, yaw, vx, vy)."""
    length: float
    width: float
    height: float
    state: Callable[[float], tuple]
    name: str = ''

    def box(self, t: float) -> Box:
        cx, cy, yaw, _, _ = self.state(t)
        return Box(cx, cy, yaw, self.length, self.width, self.height)


def constant_velocity(x0, y0, vx, vy):
    yaw = float(np.arctan2(vy, vx))
    return lambda t: (x0 + vx * t, y0 + vy * t, yaw, vx, vy)


def coordinated_turn(x0, y0, speed, heading0, omega):
    def f(t):
        h = heading0 + omega * t
        if abs(omega) < 1e-9:
            x, y = x0 + speed * np.cos(heading0) * t, y0 + speed * np.sin(heading0) * t
        else:
            x = x0 + speed / omega * (np.sin(h) - np.sin(heading0))
            y = y0 - speed / omega * (np.cos(h) - np.cos(heading0))
        return x, y, h, speed * np.cos(h), speed * np.sin(h)
    return f


def generate(objects: List[SynthObject], static: List[Box], duration: float, dt: float = 0.1,
             sm: Optional[SensorModel] = None, sensor_pose: Optional[Callable[[float], tuple]] = None,
             seed: int = 0, t0: float = 1000.0):
    """Yield dicts {stamp, xyz, ring, points_per_ring, sensor (x,y,yaw), gt: [(name,x,y,vx,vy)]}."""
    sm = sm or SensorModel()
    rng = np.random.default_rng(seed)
    dirs = ray_directions(sm)
    n = int(round(duration / dt))
    for k in range(n):
        t = k * dt
        sx, sy, syaw = sensor_pose(t) if sensor_pose else (0.0, 0.0, 0.0)
        boxes = list(static) + [o.box(t) for o in objects]
        pts, ring, ppr = ray_cast(sm, boxes, (sx, sy), syaw, rng, dirs)
        gt = [(o.name, *o.state(t)[:2], *o.state(t)[3:]) for o in objects]
        yield {'stamp': t0 + t, 't': t, 'xyz': pts, 'ring': ring, 'points_per_ring': ppr,
               'sensor': (sx, sy, syaw), 'gt': gt}
