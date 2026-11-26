"""PointCloud2 <-> NumPy (vectorised, no per-point Python loops).

Only depends on the message *fields* (works with a deserialised ROS message
from rclpy or from rosbag2_py), so the offline tool and the node share it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# sensor_msgs/PointField datatypes
_PF = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
       7: np.float32, 8: np.float64}


def cloud_to_struct(msg, fields=('x', 'y', 'z', 'intensity', 'ring', 'time')) -> np.ndarray:
    n = msg.width * msg.height
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.name in fields and f.datatype in _PF:
            names.append(f.name)
            formats.append(np.dtype(_PF[f.datatype]).newbyteorder('>' if msg.is_bigendian else '<'))
            offsets.append(f.offset)
    dt = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': msg.point_step})
    if n == 0:
        return np.zeros(0, dt)
    buf = np.frombuffer(msg.data, dtype=np.uint8)        # array.array / bytes: zero-copy
    if msg.row_step != msg.width * msg.point_step:       # row padding
        buf = np.ascontiguousarray(
            buf.reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]).reshape(-1)
    return np.frombuffer(buf, dtype=dt, count=n)


def cloud_to_xyz(msg) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float], dict]:
    """Return (xyz finite (N,3) float64, ring (N,) or None, points_per_ring or None, info)."""
    a = cloud_to_struct(msg)
    if len(a) == 0:
        return np.zeros((0, 3)), None, None, {'n_raw': 0}
    xyz = np.stack([a['x'], a['y'], a['z']], axis=1).astype(np.float64)
    ok = np.all(np.isfinite(xyz), axis=1) & (np.abs(xyz).sum(axis=1) > 1e-6)
    ring = a['ring'].astype(np.int32) if 'ring' in a.dtype.names else None
    ppr = None
    if ring is not None:
        n_rings = int(ring.max()) + 1 if len(ring) else 0
        if n_rings > 0:
            if (not ok.all()) or (msg.height > 1 and msg.width > 1):
                # organised cloud incl. no-return points: total firings / layers
                ppr = len(a) / n_rings
            else:
                ppr = float(np.bincount(ring[ok]).max())   # lower bound
    info = {'n_raw': int(len(a)), 'n_valid': int(ok.sum()), 'fields': list(a.dtype.names)}
    if 'time' in a.dtype.names:
        tt = a['time'][ok].astype(np.float64)
        info['time'] = np.where(np.isfinite(tt), tt, 0.0)
    return xyz[ok], (ring[ok] if ring is not None else None), ppr, info


def xyz_to_cloud_fields():
    """(name, offset, datatype) for an x y z float32 cloud."""
    return [('x', 0, 7), ('y', 4, 7), ('z', 8, 7)]
