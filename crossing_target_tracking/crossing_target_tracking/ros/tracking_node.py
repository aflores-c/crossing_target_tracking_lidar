#!/usr/bin/env python3
"""ROS 2 node: two-level extended-target tracking (Pieroni et al. 2024, Sec. 3).

Parameters (ROS):
  config_file (string)  YAML with the algorithm configuration (same file as the offline tool)
  overrides  (string[]) 'section.key=value' overrides applied after the file
  use_sim_time          set true for `ros2 bag play --clock`

Subscribes: input.cloud_topic (PointCloud2), input.odom_topic (Odometry; used by
odom/auto registration; its receive time on the node clock drives the
clock-offset estimate exactly like the offline tool uses the bag time).
Publishes under output.topic_prefix (default /crossing_tracking):
  non_ground_cloud, ground_cloud (opt.), bev_grid, background_grid, dynamic_grid, shadow_grid,
  blobs, low_level_tracks, clusters, tracked_objects, markers, diagnostics
and TF fixed_frame -> output.tf_child_frame (sensor pose used for the scan).
"""

from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import time  # noqa: E402
from collections import deque  # noqa: E402

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from sensor_msgs.msg import PointCloud2  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402
from visualization_msgs.msg import MarkerArray  # noqa: E402

from crossing_target_tracking_msgs.msg import BlobArray, ClusterArray, TrackedObjectArray  # noqa: E402

from ..core.config import load_config  # noqa: E402
from ..core.pipeline import Scan, TrackingPipeline  # noqa: E402
from . import messages as M  # noqa: E402
from .pointcloud import cloud_to_xyz  # noqa: E402

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass


def _sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class CrossingTrackingNode(Node):
    def __init__(self):
        super().__init__('crossing_target_tracking')
        self.declare_parameter('config_file', '')
        self.declare_parameter('overrides', [''])
        path = self.get_parameter('config_file').value
        ovr = [o for o in self.get_parameter('overrides').value if o]
        self.cfg = load_config(path or None, ovr)
        self.pipe = TrackingPipeline(self.cfg, keep_history=self.cfg.output.marker_history)
        c = self.cfg
        p = c.output.topic_prefix.rstrip('/')
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                                depth=2, durability=DurabilityPolicy.VOLATILE)
        out_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = {
            'non_ground_cloud': self.create_publisher(PointCloud2, f'{p}/non_ground_cloud', out_qos),
            'ground_cloud': self.create_publisher(PointCloud2, f'{p}/ground_cloud', out_qos),
            'bev_grid': self.create_publisher(OccupancyGrid, f'{p}/bev_grid', out_qos),
            'background_grid': self.create_publisher(OccupancyGrid, f'{p}/background_grid', out_qos),
            'dynamic_grid': self.create_publisher(OccupancyGrid, f'{p}/dynamic_grid', out_qos),
            'shadow_grid': self.create_publisher(OccupancyGrid, f'{p}/shadow_grid', out_qos),
            'blobs': self.create_publisher(BlobArray, f'{p}/blobs', out_qos),
            'low': self.create_publisher(TrackedObjectArray, f'{p}/low_level_tracks', out_qos),
            'clusters': self.create_publisher(ClusterArray, f'{p}/clusters', out_qos),
            'objects': self.create_publisher(TrackedObjectArray, f'{p}/tracked_objects', out_qos),
            'markers': self.create_publisher(MarkerArray, f'{p}/markers', out_qos),
            'diag': self.create_publisher(DiagnosticArray, f'{p}/diagnostics', 10),
        }
        self.tf = TransformBroadcaster(self) if c.output.publish_tf else None
        self.create_subscription(PointCloud2, c.input.cloud_topic, self._on_cloud, sensor_qos)
        if c.input.odom_topic and c.registration.mode in ('odom', 'auto', 'stationary'):
            self.create_subscription(Odometry, c.input.odom_topic, self._on_odom,
                                     QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                                history=HistoryPolicy.KEEP_LAST, depth=200))
        self._stats = deque(maxlen=200)
        self._last_proc = None
        self._n_skipped_rate = 0
        self._n_frames = 0
        self._last = None
        self.create_timer(c.output.diagnostics_period, self._publish_diag)
        self.get_logger().info(
            f'crossing_target_tracking: cloud={c.input.cloud_topic} odom={c.input.odom_topic} '
            f'registration={c.registration.mode} config={path or "<defaults>"} overrides={ovr}')

    # ── callbacks
    def _on_odom(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds * 1e-9
        pp, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pipe.add_odom(_sec(msg.header.stamp), now, (pp.x, pp.y, pp.z), (q.x, q.y, q.z, q.w),
                           msg.header.frame_id, msg.child_frame_id)

    def _on_cloud(self, msg: PointCloud2):
        c = self.cfg
        stamp = _sec(msg.header.stamp)
        if c.output.max_rate > 0 and self._last_proc is not None and \
                stamp - self._last_proc < 1.0 / c.output.max_rate - 1e-3:
            self._n_skipped_rate += 1
            return
        self._last_proc = stamp
        t0 = time.perf_counter()
        xyz, ring, ppr, info = cloud_to_xyz(msg)
        frame = c.input.sensor_frame or msg.header.frame_id
        r = self.pipe.process(Scan(stamp, xyz, frame, ring, ppr, info.get('time')))
        t1 = time.perf_counter()
        self._publish(r, msg)
        t2 = time.perf_counter()
        self._n_frames += 1
        self._last = r
        self._stats.append({'wall': time.time(), 'pipeline': t1 - t0,
                            'publish': t2 - t1, 'valid': r.valid, **{f'ms_{k}': v * 1e3
                                                                      for k, v in r.timings.items()}})
        if not r.valid:
            self.get_logger().warn(f'frame {r.index}: {r.reason}', throttle_duration_sec=2.0)

    # ── publishing
    def _publish(self, r, msg):
        c = self.cfg
        hdr = M.header(r.stamp, r.fixed_frame or '')
        if r.valid and self.tf is not None:
            T = r.T_fixed_sensor
            ts = TransformStamped()
            ts.header = hdr
            ts.child_frame_id = c.output.tf_child_frame
            ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = \
                (float(v) for v in T[:3, 3])
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w = \
                (float(v) for v in q)
            self.tf.sendTransform(ts)
        if r.valid and c.output.publish_clouds and r.obstacle_points_fixed is not None:
            self.pub['non_ground_cloud'].publish(M.cloud_msg(r.obstacle_points_fixed, hdr))
        if c.output.publish_ground_cloud and r.ground_points_sensor is not None:
            self.pub['ground_cloud'].publish(M.cloud_msg(r.ground_points_sensor,
                                                         M.header(r.stamp, r.sensor_frame)))
        if r.valid and c.output.publish_grids and r.dyn is not None:
            self.pub['bev_grid'].publish(M.grid_msg(r.btilde, r.spec, hdr))
            self.pub['background_grid'].publish(M.grid_msg(r.dyn.background, r.spec, hdr))
            self.pub['dynamic_grid'].publish(M.grid_msg(r.dyn.dynamic, r.spec, hdr))
            self.pub['shadow_grid'].publish(M.grid_msg(r.dyn.shadow, r.spec, hdr, values=(0, 60)))
        self.pub['blobs'].publish(M.blobs_msg(r.blobs, hdr))
        mh = c.high.min_heading_speed
        self.pub['low'].publish(M.tracks_msg(r.low_tracks, hdr, 0, r.stamp, mh,
                                             c.output.publish_low_level_tentative))
        self.pub['clusters'].publish(M.clusters_msg(r.clusters, hdr))
        self.pub['objects'].publish(M.tracks_msg(r.high_tracks, hdr, 1, r.stamp, mh))
        self.pub['markers'].publish(M.markers(hdr, r.blobs, r.low_tracks, r.clusters, r.high_tracks,
                                              c.output.marker_history, c.output.marker_ellipses,
                                              c.output.marker_lifetime))

    def _publish_diag(self):
        st = DiagnosticStatus(name='crossing_target_tracking', hardware_id=self.cfg.input.cloud_topic)
        if not self._stats:
            st.level = DiagnosticStatus.WARN
            st.message = 'no LiDAR scans received'
        else:
            s = list(self._stats)
            span = s[-1]['wall'] - s[0]['wall']
            rate = (len(s) - 1) / span if span > 0 and len(s) > 1 else 0.0
            inval = sum(not x['valid'] for x in s)
            st.level = DiagnosticStatus.OK if inval == 0 else DiagnosticStatus.WARN
            st.message = f'{rate:.1f} Hz, {inval} invalid of last {len(s)}'
            kv = {'rate_hz': rate, 'frames_total': self._n_frames, 'invalid_recent': inval,
                  'skipped_by_rate_limit': self._n_skipped_rate,
                  'missed_instants_total': self.pipe.n_missed_total,
                  'pipeline_ms_mean': 1e3 * np.mean([x['pipeline'] for x in s]),
                  'publish_ms_mean': 1e3 * np.mean([x['publish'] for x in s]),
                  'odom_clock_offset_s': self.pipe.registrar.odom.offset,
                  'angular_density': self.pipe.delta}
            for k in ('ground', 'registration', 'grid', 'dynamic', 'low_level', 'clustering', 'high_level',
                      'total'):
                v = [x[f'ms_{k}'] for x in s if f'ms_{k}' in x]
                if v:
                    kv[f'ms_{k}_mean'] = float(np.mean(v))
                    kv[f'ms_{k}_max'] = float(np.max(v))
            r = self._last
            if r is not None:
                kv['registration_method'] = r.registration.method if r.registration else ''
                kv['last_reason'] = r.reason
                kv.update({k: v for k, v in r.counters.items()})
                if r.registration:
                    kv.update({f'reg_{k}': v for k, v in r.registration.diag.items()})
            st.values = [KeyValue(key=k, value=str(v)) for k, v in kv.items()]
        da = DiagnosticArray()
        da.header.stamp = self.get_clock().now().to_msg()
        da.status = [st]
        self.pub['diag'].publish(da)


def main(args=None):
    rclpy.init(args=args)
    node = CrossingTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
