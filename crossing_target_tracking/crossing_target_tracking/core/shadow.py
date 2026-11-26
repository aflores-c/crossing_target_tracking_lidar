"""Shadow-cone estimation (paper Sec. 3.2.3).

"First, estimated shadow cones are created for each dynamic object detected
in the scene. Then, the pixels in B~ that fall under any shadow cone are kept
constant until they are visible again."

For every dynamic blob, the cone is the set of cells whose bearing (from the
sensor position at the scan stamp) lies within the blob's angular interval
(widened by `angular_margin_deg`) and whose range is larger than the blob's
closest pixel (+ `range_margin`). The blob's own pixels (dilated) are
excluded, so the object itself stays observable.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .bev_grid import SensorGeometry
from .config import ShadowConfig


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def shadow_mask(blob_pixels: List[np.ndarray], geom: SensorGeometry, cfg: ShadowConfig,
                shape) -> np.ndarray:
    """blob_pixels: list of (K, 2) arrays of (row j, col i). Returns bool (ny, nx)."""
    sh = np.zeros(shape, bool)
    if not blob_pixels:
        return sh
    th, rg = geom.theta, geom.rng
    margin = np.radians(cfg.angular_margin_deg)
    own = np.zeros(shape, np.uint8)
    for px in blob_pixels:
        j, i = px[:, 0], px[:, 1]
        own[j, i] = 1
        t = th[j, i]
        # angular interval around the circular mean (blobs subtend < pi)
        c = float(np.arctan2(np.sin(t).mean(), np.cos(t).mean()))
        d = _wrap(t - c)
        half = 0.5 * geom.alpha[j, i]               # each pixel's own angular half-width
        lo = (d - half).min() - margin
        hi = (d + half).max() + margin
        if hi - lo >= np.pi:                        # blob around the sensor: no meaningful cone
            continue
        r0 = rg[j, i].min() + cfg.range_margin
        dd = _wrap(th - c)
        sh |= (rg > r0) & (dd >= lo) & (dd <= hi)
    if cfg.exclude_blob_dilate > 0:
        k = 2 * cfg.exclude_blob_dilate + 1
        own = cv2.dilate(own, np.ones((k, k), np.uint8))
    sh &= own == 0
    return sh
