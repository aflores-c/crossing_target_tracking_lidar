# crossing_target_tracking

LiDAR-only, class-agnostic **two-level extended-target tracking** for robotic street crossing,
reproducing Section 3 of Pieroni, Corno, Parravicini, Savaresi (2024), *Design of an automated street
crossing management module for a delivery robot*, Control Engineering Practice 153:106095
(extending Parravicini et al. 2020).

```
PointCloud2 ─► range filter ─► RANSAC ground removal ─► registration (odom | CPD | stationary)
   ─► BEV counts b(n,m) ─► angular normalisation b̂ = b/(α·δ) ─► binary B̃           (Sec. 3.1, Eqs. 1-6)
   ─► moving-average background (W, τ_bck) + dilation ─► F = B̃ ∧ ¬G̃ ─► closing ─► area filter
   ─► connected components = blobs        ◄── shadow cones freeze occluded background (Sec. 3.2, Eqs. 7-9)
   ─► LOW level: CT-EKF per blob, z=[x,y], Mahalanobis gate + GNN (Hungarian), M_c/N_c, M_e/N_e (Sec. 3.3)
   ─► position–velocity clustering of confirmed tracks (δ_d = min dist, δ_v = max vel. diff.)  (Sec. 3.4)
   ─► HIGH level: CT-EKF per cluster, z=[x,y,vx,vy] (H₂), GNN, M_e/N_e only        (Sec. 3.5, Eq. 18)
   ─► tracked obstacles: id, position, velocity, speed, heading, turn rate, 5×5 covariance, history
```

Out of scope (separate extensions): crossing decision logic / FSM, intention or vehicle
classification, cameras, multi-robot communication, learned detectors.
See **[PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md)** for the equation → code → test map and every
deviation, **[docs/DATASET_REPORT.md](docs/DATASET_REPORT.md)** for the dataset inspection and
**[docs/RESULTS.md](docs/RESULTS.md)** for what was tested and what remains unverified.

## 1. Dependencies

Tested on this workstation: Ubuntu 22.04, **ROS 2 Humble**, system Python 3.10.12, NumPy 1.21.5,
SciPy 1.8.0, OpenCV 4.5.4, matplotlib 3.5.1, PyYAML, psutil, pytest 6.2.5 (all apt/system packages;
no virtualenv, no GPU). ROS packages: rclpy, sensor_msgs, nav_msgs, visualization_msgs,
diagnostic_msgs, tf2_ros, rosbag2_py, rviz2.

## 2. Build

```bash
cd ~/go2_bag_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select crossing_target_tracking_msgs crossing_target_tracking
source install/setup.bash
```

Packages: `crossing_target_tracking_msgs` (ament_cmake, messages) and `crossing_target_tracking`
(ament_python). No other package of the workspace is modified.

## 3. Running on the dataset

Bag used for development: `/media/cas/TOSHIBA EXT/bags/bag_go2_20261105_165413` (paths are always
arguments; nothing is hard-coded). Set `BAG=<path>` below.

**Inspect a bag** (topics, fields, rates, odometry clock offset, motion segments, ground):
```bash
ros2 run crossing_target_tracking inspect_bag --bag "$BAG" --out docs/DATASET_REPORT.md
```

**Mode A – rosbag playback + RViz** (sim time, only LiDAR + odometry replayed):
```bash
ros2 launch crossing_target_tracking tracking_bag.launch.py bag:="$BAG" start_offset:=40
# options: rate:=0.5 rviz:=false config:=<yaml> overrides:="['shadow.enabled=false']"
```
or with your own player:
```bash
ros2 launch crossing_target_tracking tracking.launch.py use_sim_time:=true
ros2 bag play "$BAG" --clock 100 --topics /velodyne_points /utlidar/robot_odom
```

**Mode B – offline processing** (same core code, CSV + JSON output):
```bash
ros2 run crossing_target_tracking offline_process --bag "$BAG" --start 40 --duration 97 \
    --out results/stationary [--set shadow.enabled=false] [--snapshots 20] -v
ros2 run crossing_target_tracking evaluate --runs results/stationary --out results/plots
```

**All real-bag experiments + ablations** (≈ 3 min on this machine):
```bash
ros2 run crossing_target_tracking run_experiments --bag "$BAG" --out docs/results/bag_go2_20261105_165413
```
**Synthetic experiments 1–8 with ground truth** (+ ablations):
```bash
ros2 run crossing_target_tracking synthetic_experiments --out docs/results/synthetic
```
**Check playback ≡ offline**:
```bash
ros2 bag record -o /tmp/rec /crossing_tracking/blobs /crossing_tracking/tracked_objects &   # during Mode A
ros2 run crossing_target_tracking compare_playback --recorded /tmp/rec --offline results/stationary
```

**Tests**:
```bash
cd ~/go2_bag_ws/src/crossing_target_tracking_lidar/crossing_target_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test/                 # unit + synthetic integration
CTT_BAG="$BAG" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test/test_bag_smoke.py
```

## 4. Topics (prefix `output.topic_prefix`, default `/crossing_tracking`)

| topic | type | content |
|---|---|---|
| `non_ground_cloud` | sensor_msgs/PointCloud2 | non-ground points, **fixed frame** |
| `ground_cloud` (opt.) | sensor_msgs/PointCloud2 | RANSAC ground inliers, sensor frame |
| `bev_grid` | nav_msgs/OccupancyGrid | B̃ (Eq. 6): 100 occupied / 0 free |
| `background_grid` | nav_msgs/OccupancyGrid | G̃ (Eq. 8) |
| `dynamic_grid` | nav_msgs/OccupancyGrid | F after morphology and area filter |
| `shadow_grid` | nav_msgs/OccupancyGrid | shadow cones (60) |
| `blobs` | crossing_target_tracking_msgs/BlobArray | id, centroid, area, bbox, n_pixels |
| `low_level_tracks` | crossing_target_tracking_msgs/TrackedObjectArray (level 0) | confirmed blob tracks (tentative optional) |
| `clusters` | crossing_target_tracking_msgs/ClusterArray | mean position/velocity/turn rate, member low-level ids |
| `tracked_objects` | crossing_target_tracking_msgs/TrackedObjectArray (level 1) | **final obstacles** |
| `markers` | visualization_msgs/MarkerArray | see §6 |
| `diagnostics` | diagnostic_msgs/DiagnosticArray | rate, per-stage latency, registration diagnostics, counts |
| TF `fixed_frame → crossing_sensor` | tf2 | sensor pose used for each scan |

`TrackedObject`: `level`, `status` (TENTATIVE / CONFIRMED = measured at this stamp / PREDICTED =
coasting), persistent `id` (independent counters per level), `position`, `velocity`, `speed`,
`heading` + `heading_valid` (speed ≥ `high.min_heading_speed`), `turn_rate`, `covariance[25]`
(row-major, state `[px py vx vy ω]`), `age`, `hits`, `consecutive_misses`, `first_stamp`,
`last_observed_stamp`, `member_ids` (low-level ids of the last cluster). No semantic label.

Inputs: `input.cloud_topic` (default `/velodyne_points`, best-effort QoS, compatible with reliable
publishers) and `input.odom_topic` (`/utlidar/robot_odom`).

## 5. Coordinate frames

* **Sensor frame**: the PointCloud2 `frame_id` (`velodyne_go2`). Ground removal runs here.
* **Fixed frame**: `odom` (from the odometry header) in `odom`/`auto` mode; the configurable
  `registration.fixed_frame` in `cpd` mode (initial sensor pose = identity); the sensor frame in
  `stationary` mode. Grids, blobs, tracks, clusters and markers are all in the fixed frame, z = 0.
* `T_fixed_sensor(t) = T_odom_base(t) · T_base_sensor`, odometry interpolated at the **scan stamp**
  after the clock-offset correction (`/utlidar/robot_odom` stamps are ≈ 2.73 s ahead of the LiDAR).
  `T_base_sensor` = `registration.extrinsic_xyz/rpy` (hand-measured 0.055, 0, 0.39).
* Grid: N×M cells of γ m centred on the first sensor position (paper: "initial standing point");
  `OccupancyGrid` row = y index, column = x index.

## 6. RViz

`rviz/crossing_tracking.rviz` (fixed frame `odom`, top-down). Colours: **light blue** blob
centroids, **blue** low-level tracks + velocity, **yellow** cluster centres and membership lines,
**green** high-level objects (cylinder, velocity arrow, id + speed text, trajectory, 2σ ellipse),
**red** high-level objects predicted without a measurement. Grids: B̃ (map), F (costmap), background
and shadow (disabled by default). For `stationary` mode set the fixed frame to the sensor frame.

## 7. Parameters

All parameters are in `config/crossing_tracking.yaml`, each tagged `[paper]` (structure defined by the
paper) or `[eng]` (our value: **the paper gives no numeric values**). Main groups:
`input` (topics, ranges), `ground` (RANSAC threshold/iterations/min inliers/tilt/prior/height band),
`registration` (mode, extrinsic, odometry clock offset, CPD subsampling and quality gates),
`grid` (γ, size, origin, normalisation, δ, τ_bin, ROI), `background` (W, τ_bck, warm-up, kernels,
blob area limits), `shadow`, `low_ekf` / `low_manager` (Q via σ_a, σ_ω; R via σ_r; τ_G; M/N),
`clustering` (τ_d, τ_v, ablation switch, seed), `high` (EKF, R₂, τ_G, M_e/N_e), `output`
(topics, markers, TF, rate limit). Override from the CLI with `--set section.key=value` (offline) or
`overrides:="['section.key=value']"` (launch).

## 8. Troubleshooting

* **All frames "registration invalid: no odom / waiting for odometry"** – the odometry topic is not
  replayed or not received; replay it, or use `registration.mode=stationary` only if the sensor really
  did not move (the node then warns if odometry contradicts it).
* **"implausible ego speed … (odom jump)"** – odometry discontinuity; those frames are skipped and
  counted as missed instants.
* **Tracks displaced by ~2.7 s of robot motion** – wrong odometry stamping; keep
  `odom_stamp_mode: auto_offset` (receive-time based); with `ros2 bag play` use `--clock` and
  `use_sim_time:=true`.
* **Everything is dynamic for the first seconds** – expected warm-up (`background.min_observed_frames`, W).
* **Slow / stopped vehicles disappear** – the moving-average background absorbs objects occupying a
  cell for more than τ_bck·W frames (method property; see RESULTS.md §s9). Increase W·τ_bck.
* **Node falls behind** – check `diagnostics` (`ms_*`); set `output.max_rate`, disable
  `publish_clouds` / `marker_ellipses`. Thread pools are pinned to 1 (OMP/OpenBLAS/OpenCV).
* **Nothing in RViz** – fixed frame must be `odom` (or the sensor frame in stationary mode); QoS of
  the displays is reliable/volatile.
