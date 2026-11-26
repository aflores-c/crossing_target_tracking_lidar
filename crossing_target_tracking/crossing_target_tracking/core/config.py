"""Configuration dataclasses (pure Python, no ROS).

The ROS node and the offline tool load the *same* YAML file through
`load_config`, so both run the core algorithms with identical parameters.

Every field is tagged in config/crossing_tracking.yaml as either
[paper] (structure / quantity defined by Pieroni et al. 2024) or
[engineering default] (value not given in the paper, chosen by us).
The paper gives NO numeric values for any tracking parameter.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml


@dataclass
class InputConfig:
    cloud_topic: str = '/velodyne_points'
    odom_topic: str = '/utlidar/robot_odom'
    sensor_frame: str = ''            # '' = use the PointCloud2 header frame_id
    min_range: float = 0.8            # [m] drop self-hits on the robot body
    max_range: float = 60.0           # [m]


@dataclass
class GroundConfig:
    enabled: bool = True
    distance_threshold: float = 0.15  # [m] RANSAC inlier distance
    iterations: int = 100
    min_inliers: int = 300
    max_tilt_deg: float = 20.0        # max angle between plane normal and sensor z axis
    z_prior: float = -0.7             # [m] expected ground height in the sensor frame (Go2 bag: -0.69)
    z_window: float = 0.5             # candidates: |z - z_prior| < z_window (null-equivalent: <=0 disables)
    candidate_max_range: float = 25.0
    max_candidates: int = 5000
    temporal_fallback: bool = True    # reuse previous plane when RANSAC fails
    min_height: Optional[float] = None   # [m] above plane kept as obstacle; None = distance_threshold
    max_height: Optional[float] = 3.0    # [m] above plane; None disables the height filter
    seed: int = 0


@dataclass
class RegistrationConfig:
    mode: str = 'auto'                # stationary | odom | cpd | auto
    fixed_frame: str = 'odom'         # frame of grid / tracks for odom & cpd modes
    stationary_frame: str = ''        # '' = sensor frame (stationary mode)
    # sensor pose in the odometry child frame (base_link -> velodyne_go2)
    extrinsic_xyz: List[float] = field(default_factory=lambda: [0.055, 0.0, 0.39])
    extrinsic_rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    odom_stamp_mode: str = 'auto_offset'   # auto_offset | header | fixed_offset
    odom_fixed_offset: float = 0.0    # [s] added to odom header stamps (fixed_offset)
    odom_offset_window: int = 300
    odom_offset_min_samples: int = 20
    odom_max_gap: float = 0.1         # [s] max bracket width for interpolation
    odom_max_extrapolation: float = 0.05   # [s]
    odom_buffer_s: float = 10.0
    max_speed: float = 3.0            # [m/s] plausibility check on ego speed
    stationary_check_translation: float = 0.10  # [m] warn if odom moved this much in stationary mode
    auto_fallback: str = 'cpd'        # cpd | stationary | invalid  (when no odom)
    auto_odom_wait: float = 2.0       # [s] auto: wait this long for odometry before falling back
    cpd_diagnostic: bool = False      # also run CPD and report its disagreement with odom
    # CPD (rigid, Myronenko & Song 2010)
    cpd_voxel: float = 0.4
    cpd_max_points: int = 700
    cpd_min_range: float = 1.5
    cpd_max_range: float = 30.0
    cpd_w: float = 0.2                # outlier weight
    cpd_max_iter: int = 40
    cpd_tol: float = 1e-5
    cpd_planar: bool = False          # True: estimate only x, y, yaw
    cpd_project_2d: bool = False      # True: register BEV-projected points (implies planar)
    deskew: bool = True               # odom mode: per-point motion compensation with the 'time' field
    deskew_bins: int = 10             # time slices per sweep (vectorised)
    cpd_max_sigma2: float = 0.25      # [m^2] quality gate on the final sigma^2
    cpd_min_matched: float = 0.5      # fraction of source points with posterior > 0.5
    seed: int = 0


@dataclass
class GridConfig:
    resolution: float = 0.2           # gamma [m/cell]
    size_x: float = 80.0              # N * gamma [m]
    size_y: float = 80.0              # M * gamma [m]
    origin_mode: str = 'first_pose'   # first_pose (paper: initial standing point) | fixed
    origin_xy: List[float] = field(default_factory=lambda: [0.0, 0.0])
    normalization: str = 'angular'    # angular (paper Eq. 4-6) | raw_count (ablation)
    angular_density: Any = 'auto'     # delta [points/rad/layer] or 'auto'
    tau_bin: float = 0.5              # Eq. 6 threshold on normalized count
    raw_count_threshold: int = 2      # ablation 'raw_count' only
    roi_max_range: float = 40.0       # [m] cells farther than this from the sensor are not observed


@dataclass
class BackgroundConfig:
    window: int = 100                 # W frames (Eq. 7)
    tau_bck: float = 0.4              # Eq. 8
    min_observed_frames: int = 10     # per-cell warm-up (generalises the global warm-up)
    dilate_kernel: int = 3            # Sec. 3.2.2 dilation of the background [cells]
    close_kernel: int = 5             # Sec. 3.2.2 closing of the dynamic image [cells]
    min_blob_area: float = 0.12       # [m^2] Sec. 3.2.2 area filter
    max_blob_area: float = 40.0       # [m^2] engineering: reject huge components (registration jumps)
    connectivity: int = 8
    max_frame_gap: float = 0.5        # [s] larger gaps reset the background


@dataclass
class ShadowConfig:
    enabled: bool = True
    angular_margin_deg: float = 1.0   # widen each cone on both sides
    range_margin: float = 0.0         # [m] cone starts this far behind the closest blob pixel
    exclude_blob_dilate: int = 1      # blob pixels (dilated) are not part of their own shadow


@dataclass
class EKFConfig:
    sigma_a: float = 2.0              # [m/s^2] white acceleration (CT model process noise)
    sigma_omega: float = 0.5          # [rad/s^2] white turn-rate acceleration
    sigma_r: float = 0.25             # [m] measurement std (low level)
    init_sigma_v: float = 5.0         # [m/s]
    init_sigma_omega: float = 0.5     # [rad/s]
    omega_max: float = 2.0            # [rad/s] numerical clamp
    omega_eps: float = 1e-4           # Taylor branch threshold


@dataclass
class TrackManagerConfig:
    gate: float = 9.21                # tau_G on squared Mahalanobis distance (chi2 2 dof, 99 %)
    confirm_m: int = 5                # M_c
    confirm_n: int = 7                # N_c
    delete_m: int = 1                 # M_e
    delete_n: int = 8                 # N_e
    history_length: int = 100
    use_confirmation: bool = True


@dataclass
class ClusteringConfig:
    tau_d: float = 3.0                # [m] position threshold (Eq. 16)
    tau_v: float = 2.0                # [m/s] velocity threshold (Eq. 17)
    use_velocity: bool = True         # False = 'position-only' ablation
    growth: str = 'fixpoint'          # fixpoint | single_pass
    include_predicted: bool = False   # cluster confirmed tracks without a measurement this frame
    seed: int = 0


@dataclass
class HighLevelConfig:
    ekf: EKFConfig = field(default_factory=lambda: EKFConfig(sigma_a=1.5, sigma_omega=0.3, sigma_r=0.3,
                                                             init_sigma_v=1.0))
    sigma_r_pos: float = 1.0          # [m]
    sigma_r_vel: float = 1.5          # [m/s]
    manager: TrackManagerConfig = field(default_factory=lambda: TrackManagerConfig(
        gate=13.28, confirm_m=1, confirm_n=1, delete_m=1, delete_n=15, history_length=300,
        use_confirmation=False))
    min_heading_speed: float = 0.5    # [m/s] heading reported only above this


@dataclass
class OutputConfig:
    publish_clouds: bool = True
    publish_ground_cloud: bool = False
    publish_grids: bool = True
    publish_tf: bool = True
    tf_child_frame: str = 'crossing_sensor'
    publish_low_level_tentative: bool = False
    marker_history: bool = True
    marker_ellipses: bool = True
    marker_lifetime: float = 0.3
    topic_prefix: str = '/crossing_tracking'
    diagnostics_period: float = 1.0
    max_rate: float = 0.0             # [Hz] 0 = process every scan


@dataclass
class Config:
    input: InputConfig = field(default_factory=InputConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    low_ekf: EKFConfig = field(default_factory=EKFConfig)
    low_manager: TrackManagerConfig = field(default_factory=TrackManagerConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    high: HighLevelConfig = field(default_factory=HighLevelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge(obj, data: dict, path: str = ''):
    if data is None:
        return obj
    if not isinstance(data, dict):
        raise ValueError(f'config section {path or "<root>"} must be a mapping')
    names = {f.name: f for f in dataclasses.fields(obj)}
    for k, v in data.items():
        if k not in names:
            raise ValueError(f'unknown config key: {path}{k}')
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            _merge(cur, v, f'{path}{k}.')
        else:
            if isinstance(cur, float) and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            setattr(obj, k, v)
    return obj


def _coerce(s: str):
    return yaml.safe_load(s)


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply 'section.key=value' overrides (value parsed as YAML)."""
    for ov in overrides or []:
        if '=' not in ov:
            raise ValueError(f'override must be key=value: {ov}')
        key, val = ov.split('=', 1)
        parts = key.strip().split('.')
        d: dict = {}
        cur = d
        for p in parts[:-1]:
            cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = _coerce(val)
        _merge(cfg, d)
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # allow a ROS-style wrapper  node_name: {ros__parameters: {...}} to be ignored
        data.pop('ros__parameters', None)
        _merge(cfg, data)
    apply_overrides(cfg, overrides or [])
    validate(cfg)
    return cfg


def validate(cfg: Config):
    if cfg.registration.mode not in ('stationary', 'odom', 'cpd', 'auto'):
        raise ValueError(f'registration.mode invalid: {cfg.registration.mode}')
    if cfg.registration.auto_fallback not in ('cpd', 'stationary', 'invalid'):
        raise ValueError('registration.auto_fallback must be cpd|stationary|invalid')
    if cfg.grid.normalization not in ('angular', 'raw_count'):
        raise ValueError('grid.normalization must be angular|raw_count')
    if cfg.grid.resolution <= 0:
        raise ValueError('grid.resolution must be > 0')
    if cfg.background.window < 1:
        raise ValueError('background.window must be >= 1')
    if not 0.0 < cfg.background.tau_bck < 1.0:
        raise ValueError('background.tau_bck must be in (0, 1)')
    if cfg.clustering.growth not in ('fixpoint', 'single_pass'):
        raise ValueError('clustering.growth must be fixpoint|single_pass')
    for m in (cfg.low_manager, cfg.high.manager):
        if m.confirm_m > m.confirm_n or m.delete_m > m.delete_n:
            raise ValueError('M/N logic requires M <= N')
        if m.delete_m < 1:
            raise ValueError('delete_m must be >= 1')


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
