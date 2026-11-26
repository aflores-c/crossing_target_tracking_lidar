# crossing_target_tracking_lidar

LiDAR-only, class-agnostic two-level extended-target tracker for robotic street crossing (ROS 2 Humble).
It reproduces Section 3 of Pieroni, Corno, Parravicini, Savaresi (2024), *Design of an automated street
crossing management module for a delivery robot*, Control Engineering Practice 153:106095.

Developed and evaluated on a Unitree Go2 with a Velodyne VLP-16.

| package | type | content |
|---|---|---|
| [`crossing_target_tracking`](crossing_target_tracking/README.md) | ament_python | pipeline core, ROS node, launch files, RViz config, offline tools, tests, docs |
| `crossing_target_tracking_msgs` | ament_cmake | `Blob(Array)`, `Cluster(Array)`, `TrackedObject(Array)` |

## Quick start

```bash
cd ~/ros2_ws/src
git clone <this repository> crossing_target_tracking_lidar
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select crossing_target_tracking_msgs crossing_target_tracking
source install/setup.bash
ros2 launch crossing_target_tracking tracking_bag.launch.py bag:=<path/to/bag> start_offset:=40
```

Documentation:

- [README](crossing_target_tracking/README.md): usage, topics, frames, parameters, troubleshooting.
- [PAPER_REPRODUCTION.md](crossing_target_tracking/PAPER_REPRODUCTION.md): maps each paper equation to its code and test, and lists the deviations.
- [DATASET_REPORT.md](crossing_target_tracking/docs/DATASET_REPORT.md): inspection of the dataset bag.
- [RESULTS.md](crossing_target_tracking/docs/RESULTS.md): measured results, ablations and timing; it also lists what is still unverified because the bag has no ground truth.

Not included: the rosbag dataset, the paper PDF, and the per-frame CSV result tables. The CSVs can be regenerated with `run_experiments`.
