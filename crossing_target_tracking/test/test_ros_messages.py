"""Core results -> ROS messages (requires a sourced ROS 2 + built crossing_target_tracking_msgs)."""
import numpy as np
import pytest

M = pytest.importorskip('crossing_target_tracking.ros.messages', reason='ROS 2 messages not available')

from crossing_target_tracking.core import synth as S  # noqa: E402
from crossing_target_tracking.core.pipeline import Scan, TrackingPipeline  # noqa: E402
from crossing_target_tracking.tools.scenarios import base_config  # noqa: E402


def test_all_messages_from_pipeline():
    p = TrackingPipeline(base_config(['grid.size_x=40', 'grid.size_y=40', 'background.window=20']))
    car = S.SynthObject(4.5, 1.8, 1.5, S.constant_velocity(-15, 8, 6, 0), 'car')
    for fr in S.generate([car], [S.Box(0, 15, 0, 30, 1, 3)], 4.0):
        r = p.process(Scan(fr['stamp'], fr['xyz'], 'lidar', fr['ring'], fr['points_per_ring']))
    hdr = M.header(r.stamp, r.fixed_frame)
    g = M.grid_msg(r.btilde, r.spec, hdr)
    assert g.info.width == r.spec.nx and len(g.data) == r.spec.nx * r.spec.ny
    assert g.info.origin.position.x == pytest.approx(r.spec.origin[0])
    assert set(np.unique(np.frombuffer(g.data, np.int8))) <= {0, 100}
    c = M.cloud_msg(r.obstacle_points_fixed, hdr)
    assert c.width == len(r.obstacle_points_fixed) and c.point_step == 12
    o = M.tracks_msg(r.high_tracks, hdr, 1, r.stamp, 0.5)
    assert len(o.objects) == len(r.high_tracks) >= 1
    ob = o.objects[0]
    assert len(ob.covariance) == 25 and ob.level == 1
    assert ob.speed == pytest.approx(np.hypot(ob.velocity.x, ob.velocity.y))
    ma = M.markers(hdr, r.blobs, r.low_tracks, r.clusters, r.high_tracks, True, True, 0.3)
    assert ma.markers[0].action == ma.markers[0].DELETEALL
    assert all(m.header.frame_id == r.fixed_frame for m in ma.markers)
    assert M.sec_to_time(12.9999999999).nanosec < 1000000000


def test_pointcloud_roundtrip():
    from sensor_msgs.msg import PointCloud2
    from crossing_target_tracking.ros.pointcloud import cloud_to_xyz
    xyz = np.random.default_rng(0).normal(size=(100, 3))
    m = M.cloud_msg(xyz, M.header(1.0, 'f'))
    assert isinstance(m, PointCloud2)
    back, ring, ppr, info = cloud_to_xyz(m)
    np.testing.assert_allclose(back, xyz.astype(np.float32), rtol=1e-6)
    assert ring is None and info['n_valid'] == 100
