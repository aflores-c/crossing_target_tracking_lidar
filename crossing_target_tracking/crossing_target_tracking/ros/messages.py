"""Core results -> ROS messages (grids, clouds, tracks, markers)."""

from __future__ import annotations

import array
from typing import List

import numpy as np
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from crossing_target_tracking_msgs.msg import (Blob as BlobMsg, BlobArray, Cluster as ClusterMsg,
                                               ClusterArray, TrackedObject, TrackedObjectArray)

from ..core.bev_grid import GridSpec
from ..core.pipeline import TrackSnapshot
from ..core.tracker import CONFIRMED, PREDICTED, TENTATIVE

BLUE = (0.2, 0.45, 1.0, 1.0)
LIGHT_BLUE = (0.5, 0.75, 1.0, 0.9)
YELLOW = (1.0, 0.85, 0.0, 1.0)
GREEN = (0.1, 0.9, 0.2, 1.0)
RED = (1.0, 0.15, 0.15, 1.0)


def sec_to_time(t: float) -> Time:
    s = int(np.floor(t))
    return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000)


def header(stamp: float, frame: str) -> Header:
    return Header(stamp=sec_to_time(stamp), frame_id=frame)


def grid_msg(grid: np.ndarray, spec: GridSpec, hdr: Header, values=(0, 100), mask_unknown=None) -> OccupancyGrid:
    m = OccupancyGrid()
    m.header = hdr
    m.info.map_load_time = hdr.stamp
    m.info.resolution = float(spec.resolution)
    m.info.width = spec.nx
    m.info.height = spec.ny
    ox, oy = spec.origin
    m.info.origin = Pose(position=Point(x=ox, y=oy, z=0.0), orientation=Quaternion(w=1.0))
    d = np.where(grid > 0, values[1], values[0]).astype(np.int8)
    if mask_unknown is not None:
        d[mask_unknown] = -1
    a = array.array('b')
    a.frombytes(d.tobytes())
    m.data = a
    return m


def cloud_msg(xyz: np.ndarray, hdr: Header) -> PointCloud2:
    m = PointCloud2()
    m.header = hdr
    m.height = 1
    m.width = int(len(xyz))
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate('xyz')]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 12 * m.width
    m.is_dense = True
    a = array.array('B')
    a.frombytes(np.ascontiguousarray(xyz, dtype=np.float32).tobytes())
    m.data = a
    return m


def blobs_msg(blobs, hdr: Header) -> BlobArray:
    out = BlobArray(header=hdr)
    for b in blobs:
        out.blobs.append(BlobMsg(id=int(b.id), centroid=Point(x=b.x, y=b.y, z=0.0), area=float(b.area),
                                 x_min=float(b.bbox[0]), y_min=float(b.bbox[1]), x_max=float(b.bbox[2]),
                                 y_max=float(b.bbox[3]), n_pixels=b.n_pixels))
    return out


def clusters_msg(clusters, hdr: Header) -> ClusterArray:
    out = ClusterArray(header=hdr)
    for c in clusters:
        out.clusters.append(ClusterMsg(
            id=int(c.id), position=Point(x=float(c.position[0]), y=float(c.position[1]), z=0.0),
            velocity=Vector3(x=float(c.velocity[0]), y=float(c.velocity[1]), z=0.0),
            turn_rate=float(c.omega), member_track_ids=[int(i) for i in c.member_ids],
            n_members=c.n_members))
    return out


def tracks_msg(tracks: List[TrackSnapshot], hdr: Header, level: int, stamp: float,
               min_heading_speed: float, include_tentative: bool = True) -> TrackedObjectArray:
    out = TrackedObjectArray(header=hdr)
    for t in tracks:
        if t.status == TENTATIVE and not include_tentative:
            continue
        x = t.x
        sp = float(np.hypot(x[2], x[3]))
        o = TrackedObject(level=level, status=int(t.status), id=int(t.id),
                          position=Point(x=float(x[0]), y=float(x[1]), z=0.0),
                          velocity=Vector3(x=float(x[2]), y=float(x[3]), z=0.0),
                          speed=sp, heading=float(np.arctan2(x[3], x[2])),
                          heading_valid=bool(sp >= min_heading_speed), turn_rate=float(x[4]),
                          age=float(stamp - t.first_stamp), hits=int(t.hits),
                          consecutive_misses=int(t.consecutive_misses),
                          first_stamp=sec_to_time(t.first_stamp),
                          last_observed_stamp=sec_to_time(t.last_obs_stamp),
                          member_ids=[int(i) for i in t.member_ids])
        o.covariance = [float(v) for v in t.P.reshape(-1)]
        out.objects.append(o)
    return out


def _c(rgba) -> ColorRGBA:
    return ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))


def _m(hdr, ns, mid, mtype, color, scale, lifetime):
    m = Marker(header=hdr, ns=ns, id=mid, type=mtype, action=Marker.ADD)
    m.pose.orientation.w = 1.0
    m.color = _c(color)
    m.scale = Vector3(x=float(scale[0]), y=float(scale[1]), z=float(scale[2]))
    m.lifetime = Duration(sec=int(lifetime), nanosec=int((lifetime % 1) * 1e9))
    m.frame_locked = False
    return m


def markers(hdr: Header, blobs, low: List[TrackSnapshot], clusters, high: List[TrackSnapshot],
            history: bool, ellipses: bool, lifetime: float) -> MarkerArray:
    ma = MarkerArray()
    clear = Marker(header=hdr, action=Marker.DELETEALL)
    ma.markers.append(clear)
    z = 0.3
    # blobs (raw measurements): light blue cubes at centroids
    if blobs:
        m = _m(hdr, 'blobs', 0, Marker.CUBE_LIST, LIGHT_BLUE, (0.25, 0.25, 0.05), lifetime)
        m.points = [Point(x=b.x, y=b.y, z=0.05) for b in blobs]
        ma.markers.append(m)
    # low-level confirmed tracks: blue spheres + velocity lines
    lc = [t for t in low if t.status != TENTATIVE]
    if lc:
        m = _m(hdr, 'low_level_tracks', 0, Marker.SPHERE_LIST, BLUE, (0.3, 0.3, 0.3), lifetime)
        m.points = [Point(x=float(t.x[0]), y=float(t.x[1]), z=z) for t in lc]
        ma.markers.append(m)
        v = _m(hdr, 'low_level_velocity', 0, Marker.LINE_LIST, BLUE, (0.05, 0, 0), lifetime)
        for t in lc:
            v.points += [Point(x=float(t.x[0]), y=float(t.x[1]), z=z),
                         Point(x=float(t.x[0] + t.x[2]), y=float(t.x[1] + t.x[3]), z=z)]
        ma.markers.append(v)
    # clusters: yellow centre + lines to members
    if clusters:
        pos = {t.id: t.x for t in low}
        cm = _m(hdr, 'clusters', 0, Marker.SPHERE_LIST, YELLOW, (0.45, 0.45, 0.1), lifetime)
        cl = _m(hdr, 'cluster_membership', 0, Marker.LINE_LIST, YELLOW, (0.04, 0, 0), lifetime)
        for c in clusters:
            p = Point(x=float(c.position[0]), y=float(c.position[1]), z=z + 0.2)
            cm.points.append(p)
            for mid in c.member_ids:
                if mid in pos:
                    cl.points += [p, Point(x=float(pos[mid][0]), y=float(pos[mid][1]), z=z)]
        ma.markers.append(cm)
        if cl.points:
            ma.markers.append(cl)
    # high-level objects
    for t in high:
        col = GREEN if t.status == CONFIRMED else RED
        x = t.x
        zz = z + 0.5
        s = _m(hdr, 'objects', t.id, Marker.CYLINDER, col, (0.6, 0.6, 1.0), lifetime)
        s.pose.position = Point(x=float(x[0]), y=float(x[1]), z=0.5)
        ma.markers.append(s)
        a = _m(hdr, 'object_velocity', t.id, Marker.ARROW, col, (0.12, 0.25, 0.25), lifetime)
        a.points = [Point(x=float(x[0]), y=float(x[1]), z=zz),
                    Point(x=float(x[0] + x[2]), y=float(x[1] + x[3]), z=zz)]
        ma.markers.append(a)
        txt = _m(hdr, 'object_ids', t.id, Marker.TEXT_VIEW_FACING, (1, 1, 1, 1), (0, 0, 0.6), lifetime)
        txt.pose.position = Point(x=float(x[0]), y=float(x[1]), z=1.6)
        txt.text = f'{t.id}  {np.hypot(x[2], x[3]):.1f} m/s' + ('' if t.status == CONFIRMED else ' (pred)')
        ma.markers.append(txt)
        if history and len(t.history) > 1:
            h = _m(hdr, 'object_history', t.id, Marker.LINE_STRIP, (col[0] * .6, col[1] * .6, col[2] * .6, .8),
                   (0.06, 0, 0), lifetime)
            h.points = [Point(x=float(e[1]), y=float(e[2]), z=0.1) for e in t.history]
            ma.markers.append(h)
        if ellipses:
            P = t.P[:2, :2]
            w, V = np.linalg.eigh(0.5 * (P + P.T))
            w = np.maximum(w, 1e-6)
            e = _m(hdr, 'object_covariance', t.id, Marker.CYLINDER, (col[0], col[1], col[2], 0.25),
                   (2 * 2 * np.sqrt(w[1]), 2 * 2 * np.sqrt(w[0]), 0.02), lifetime)   # 2-sigma
            yaw = float(np.arctan2(V[1, 1], V[0, 1]))
            e.pose.position = Point(x=float(x[0]), y=float(x[1]), z=0.02)
            e.pose.orientation = Quaternion(z=float(np.sin(yaw / 2)), w=float(np.cos(yaw / 2)))
            ma.markers.append(e)
    return ma


__all__ = ['grid_msg', 'cloud_msg', 'blobs_msg', 'clusters_msg', 'tracks_msg', 'markers', 'header',
           'sec_to_time', 'CONFIRMED', 'PREDICTED', 'TENTATIVE']
