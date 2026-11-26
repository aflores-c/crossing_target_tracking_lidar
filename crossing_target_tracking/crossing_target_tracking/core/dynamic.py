"""Dynamic-object extraction (paper Sec. 3.2, Eqs. 7-9).

Per frame t (B~(t) = binary grid from Eq. 6):

  Eq. (7)  G(t) = mean of the W most recent (shadow-corrected) grids,
           T = 0 .. W-1 (current frame included).  NOTE: the printed Eq. (7)
           sums T = 0 .. W and divides by W (W+1 terms); we read this as an
           index typo and average exactly W frames so that G is in [0, 1].
  Eq. (8)  g~ = 1 if G > tau_bck
  Sec. 3.2.2  g~ is dilated BEFORE the subtraction
  Eq. (9)  f = b~ AND NOT dilate(g~)
  Sec. 3.2.2  closing, then removal of components with area < threshold,
              then connected-component extraction (blobs).
  Sec. 3.2.3  shadow cones of the detected blobs freeze B~ in the shadow:
              the value pushed into the history for a shadowed cell is the
              last value observed while it was visible.

Engineering additions (named, documented, switchable):
  * Cells farther than grid.roi_max_range from the sensor are treated like
    shadowed cells (not observed -> value held).
  * Per-cell warm-up: a cell produces dynamic pixels only after it has been
    observed in `min_observed_frames` of the W frames. For a stationary
    sensor this reduces to the usual global warm-up of the moving average;
    for a moving sensor it prevents newly visible static structure from being
    reported as dynamic.
  * A time gap larger than max_frame_gap resets the history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .bev_grid import GridSpec, SensorGeometry
from .config import BackgroundConfig, ShadowConfig
from .shadow import shadow_mask


@dataclass
class Blob:
    id: int
    x: float                       # geometric centre [m] (fixed frame)
    y: float
    area: float                    # [m^2]
    bbox: tuple                    # (x_min, y_min, x_max, y_max) [m]
    pixels: np.ndarray             # (K, 2) int (row j, col i)
    stamp: float

    @property
    def n_pixels(self) -> int:
        return int(len(self.pixels))


@dataclass
class DynamicResult:
    blobs: List[Blob]
    background: np.ndarray         # g~ (uint8)
    dynamic: np.ndarray            # final F after morphology / area filter (uint8)
    dynamic_raw: np.ndarray        # Eq. (9) before morphology (uint8)
    shadow: np.ndarray             # bool
    warmup: np.ndarray             # bool, cells not yet observed enough
    frames_in_history: int
    reset: bool = False
    stats: dict = field(default_factory=dict)


class _Ring:
    """Fixed-capacity ring of uint8 grids with an O(1) running sum."""

    def __init__(self, cap: int, shape):
        self.cap = cap
        self.buf = np.zeros((max(cap, 1),) + tuple(shape), np.uint8)
        self.sum = np.zeros(shape, np.int32)
        self.n = 0
        self.head = 0

    def push(self, g: np.ndarray):
        if self.cap == 0:
            return
        if self.n == self.cap:
            self.sum -= self.buf[self.head]
        else:
            self.n += 1
        self.buf[self.head] = g
        self.sum += g
        self.head = (self.head + 1) % self.cap

    def clear(self):
        self.buf[:] = 0
        self.sum[:] = 0
        self.n = 0
        self.head = 0


class DynamicExtractor:
    def __init__(self, spec: GridSpec, cfg: BackgroundConfig, shadow_cfg: ShadowConfig):
        self.spec = spec
        self.cfg = cfg
        self.shadow_cfg = shadow_cfg
        shape = spec.shape
        self.hist_b = _Ring(cfg.window - 1, shape)      # previous W-1 corrected grids
        self.hist_v = _Ring(cfg.window - 1, shape)      # previous W-1 visibility masks
        self.last = np.zeros(shape, np.uint8)            # last visible value of each cell
        self.last_stamp: Optional[float] = None
        self.k_dil = np.ones((cfg.dilate_kernel,) * 2, np.uint8) if cfg.dilate_kernel > 1 else None
        # square element: an elliptical one cannot bridge gaps in 1-pixel-thick
        # LiDAR surface traces (its tips fail the erosion), see test_dynamic.py
        self.k_close = np.ones((cfg.close_kernel,) * 2, np.uint8) if cfg.close_kernel > 1 else None
        self.n_resets = 0

    def reset(self):
        self.hist_b.clear()
        self.hist_v.clear()
        self.last[:] = 0
        self.n_resets += 1

    def process(self, btilde: np.ndarray, geom: SensorGeometry, roi_max_range: float,
                stamp: float) -> DynamicResult:
        c = self.cfg
        reset = False
        if self.last_stamp is not None and (stamp - self.last_stamp > c.max_frame_gap
                                            or stamp < self.last_stamp):
            self.reset()
            reset = True
        self.last_stamp = stamp
        in_range = geom.rng <= roi_max_range
        # current contribution before shadows are known (only cells out of range held)
        b_cur = np.where(in_range, btilde, self.last).astype(np.uint8)
        k = self.hist_b.n + 1
        G = (self.hist_b.sum + b_cur) / float(k)                       # Eq. (7)
        g = (G > c.tau_bck).astype(np.uint8)                           # Eq. (8)
        obs = self.hist_v.sum + in_range
        warm = obs < min(c.min_observed_frames, c.window)
        g_d = cv2.dilate(g, self.k_dil) if self.k_dil is not None else g
        f_raw = (btilde.astype(bool) & ~g_d.astype(bool) & in_range & ~warm).astype(np.uint8)  # Eq. (9)
        f = cv2.morphologyEx(f_raw, cv2.MORPH_CLOSE, self.k_close) if self.k_close is not None else f_raw
        blobs, f_final = self._components(f, stamp)
        if self.shadow_cfg.enabled:
            sh = shadow_mask([b.pixels for b in blobs], geom, self.shadow_cfg, self.spec.shape)
        else:
            sh = np.zeros(self.spec.shape, bool)
        visible = in_range & ~sh
        b_corr = np.where(visible, btilde, self.last).astype(np.uint8)
        self.last = b_corr
        self.hist_b.push(b_corr)
        self.hist_v.push(visible.astype(np.uint8))
        return DynamicResult(blobs, g, f_final, f_raw, sh, warm, k, reset,
                             {'shadow_cells': int(sh.sum()), 'warmup_cells': int((warm & in_range).sum()),
                              'background_cells': int(g.sum()), 'dynamic_raw_cells': int(f_raw.sum())})

    def _components(self, f: np.ndarray, stamp: float):
        c = self.cfg
        g = self.spec.resolution
        n, labels, stats, cents = cv2.connectedComponentsWithStats(f, connectivity=c.connectivity)
        if n <= 1:
            return [], np.zeros_like(f)
        area = stats[1:, cv2.CC_STAT_AREA] * g * g
        keep = np.flatnonzero((area >= c.min_blob_area) & (area <= c.max_blob_area)) + 1
        if len(keep) == 0:
            return [], np.zeros_like(f)
        lut = np.zeros(n, np.int32)
        lut[keep] = np.arange(1, len(keep) + 1)
        lab = lut[labels]
        f_final = (lab > 0).astype(np.uint8)
        nz = np.flatnonzero(lab)
        lv = lab.ravel()[nz]
        order = np.argsort(lv, kind='stable')
        nz, lv = nz[order], lv[order]
        splits = np.flatnonzero(np.diff(lv)) + 1
        groups = np.split(nz, splits)
        ox, oy = self.spec.origin
        nx = self.spec.nx
        blobs = []
        for bid, (L, flat) in enumerate(zip(keep, groups)):
            rows, cols = flat // nx, flat % nx
            px = np.stack([rows, cols], axis=1)
            xs = ox + (cols + 0.5) * g
            ys = oy + (rows + 0.5) * g
            bbox = (ox + cols.min() * g, oy + rows.min() * g,
                    ox + (cols.max() + 1) * g, oy + (rows.max() + 1) * g)
            blobs.append(Blob(bid, float(xs.mean()), float(ys.mean()), float(len(flat) * g * g),
                              bbox, px, stamp))
        return blobs, f_final
