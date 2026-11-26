"""Synthetic experiments 1-8 (deterministic, with exact ground truth).

Each scenario returns a `ScenarioSpec`; `run_scenario` feeds the ray-cast
scans through the full TrackingPipeline and `evaluate_scenario` computes
metrics against ground truth.

NOTE on positions: the tracker estimates the centre of the *visible* blob
(paper: "the position of its geometric center is the measurement"), while
the synthetic ground truth is the box centre. The difference (up to ~half a
vehicle width/length) is an inherent bias of the method, so position errors
must be read together with the object size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from crossing_target_tracking.core import synth as S
from crossing_target_tracking.core.config import Config, load_config
from crossing_target_tracking.core.pipeline import Scan, TrackingPipeline
from crossing_target_tracking.core.tracker import CONFIRMED, TENTATIVE

OUT = 1e4   # "far away" coordinate for objects that are not in the scene yet


def appear(t_on: float, state: Callable, t_off: float = 1e9):
    """Object present only in [t_on, t_off)."""
    def f(t):
        if t < t_on or t >= t_off:
            return OUT, OUT, 0.0, 0.0, 0.0
        return state(t - t_on)
    return f


def straight_then_turn(x0, y0, speed, t_turn, omega):
    """Straight along +x for t_turn seconds, then a coordinated turn with rate omega."""
    turn = S.coordinated_turn(x0 + speed * t_turn, y0, speed, 0.0, omega)

    def f(t):
        if t < t_turn:
            return x0 + speed * t, y0, 0.0, speed, 0.0
        return turn(t - t_turn)
    return f


def stop_at(state: Callable, t_stop: float):
    def f(t):
        if t < t_stop:
            return state(t)
        x, y, yaw, _, _ = state(t_stop)
        return x, y, yaw, 0.0, 0.0
    return f


STATIC_SCENE = [S.Box(0.0, 18.0, 0.0, 40.0, 1.0, 3.0),        # building facade
                S.Box(-12.0, -14.0, 0.0, 20.0, 1.0, 2.5),     # wall
                S.Box(8.0, -3.5, 0.2, 4.4, 1.8, 1.5),         # parked car
                S.Box(-6.0, 3.0, 0.0, 0.3, 0.3, 3.0),         # pole
                S.Box(14.0, 4.0, 0.0, 0.6, 0.6, 3.0)]         # tree trunk


@dataclass
class ScenarioSpec:
    name: str
    description: str
    objects: List[S.SynthObject]
    static: List[S.Box]
    duration: float
    sensor_pose: Optional[Callable] = None        # t -> (x, y, yaw) in world
    overrides: List[str] = field(default_factory=list)
    eval_from: float = 0.0                        # ignore metrics before this time [s]


def scenarios() -> Dict[str, ScenarioSpec]:
    sc = {}
    sc['s1_static'] = ScenarioSpec(
        's1_static', 'Static scene, no moving objects', [], STATIC_SCENE, 8.0)
    sc['s2_constant_speed'] = ScenarioSpec(
        's2_constant_speed', 'One car at 10 m/s on a straight lane (y = 8 m)',
        [S.SynthObject(4.5, 1.8, 1.5, appear(0.5, S.constant_velocity(-48, 8, 10, 0)), 'car')],
        STATIC_SCENE, 10.0)
    sc['s3_turning'] = ScenarioSpec(
        's3_turning', 'One car: straight at 7 m/s, then a right turn (omega = -0.3 rad/s, R = 23 m)',
        [S.SynthObject(4.5, 1.8, 1.5, appear(0.5, straight_then_turn(-45, 8, 7.0, 4.0, -0.3)), 'car')],
        STATIC_SCENE, 10.5)
    sc['s4_two_vehicles'] = ScenarioSpec(
        's4_two_vehicles', 'Car (8 m/s) and motorcycle (12 m/s) on adjacent lanes, overtaking',
        [S.SynthObject(4.5, 1.8, 1.5, appear(0.5, S.constant_velocity(-44, 6.0, 8, 0)), 'car'),
         S.SynthObject(2.0, 0.8, 1.4, appear(0.5, S.constant_velocity(-62, 8.6, 12, 0)), 'motorcycle')],
        STATIC_SCENE, 9.0)
    sc['s4b_two_pedestrians'] = ScenarioSpec(
        's4b_two_pedestrians', 'Two pedestrians passing each other 0.8 m apart (+-1.4 m/s)',
        [S.SynthObject(0.5, 0.5, 1.7, appear(2.0, S.constant_velocity(-6, 5.0, 1.4, 0)), 'ped_a'),
         S.SynthObject(0.5, 0.5, 1.7, appear(2.0, S.constant_velocity(6, 5.8, -1.4, 0)), 'ped_b')],
        STATIC_SCENE, 10.0)
    sc['s5_occlusion'] = ScenarioSpec(
        's5_occlusion', 'Cyclist (5 m/s, y = 10 m) passes behind a parked van (static occluder at y = 4 m)',
        [S.SynthObject(1.8, 0.6, 1.6, appear(2.0, S.constant_velocity(-14, 10, 5, 0)), 'cyclist')],
        STATIC_SCENE + [S.Box(0.0, 4.0, 0.0, 2.0, 1.9, 2.2)], 9.0)
    sc['s6_multi'] = ScenarioSpec(
        's6_multi', 'Four road users crossing the field of view at different times',
        [S.SynthObject(4.5, 1.8, 1.5, appear(0.5, S.constant_velocity(-52, 6, 11, 0)), 'car_east'),
         S.SynthObject(4.8, 1.9, 1.6, appear(1.0, S.constant_velocity(53, 9.5, -9, 0)), 'car_west'),
         S.SynthObject(1.8, 0.6, 1.6, appear(0.5, S.constant_velocity(-30, 12.5, 5, 0)), 'cyclist'),
         S.SynthObject(0.5, 0.5, 1.7, appear(2.0, S.constant_velocity(4, -2, 0, 1.3)), 'pedestrian')],
        STATIC_SCENE, 10.0)
    sc['s7_background_change'] = ScenarioSpec(
        's7_background_change', 'Car drives in and parks; a new static box appears; static clutter',
        [S.SynthObject(4.5, 1.8, 1.5, appear(0.5, stop_at(S.constant_velocity(-45, 6, 8, 0), 5.0)), 'parking_car'),
         S.SynthObject(1.0, 1.0, 1.0, appear(4.0, lambda t: (5.0, -8.0, 0.0, 0.0, 0.0)), 'new_box')],
        STATIC_SCENE + [S.Box(-3 + 1.5 * k, -6.0, 0.0, 0.2, 0.2, 1.0) for k in range(6)], 22.0)
    sc['s9_shadow'] = ScenarioSpec(
        's9_shadow', 'Slow bus (12 m, 3 m/s) on the near lane occludes the facade for > (1 - tau_bck) W frames',
        [S.SynthObject(12.0, 2.5, 3.0, appear(0.5, S.constant_velocity(-45, 5, 3.0, 0)), 'bus')],
        STATIC_SCENE, 26.0)
    sc['s8_moving_sensor'] = ScenarioSpec(
        's8_moving_sensor', 'Sensor walks at 0.8 m/s while turning (0.08 rad/s); one car at 9 m/s; odom given',
        [S.SynthObject(4.5, 1.8, 1.5, appear(0.5, S.constant_velocity(-52, 9, 9, 0)), 'car')],
        STATIC_SCENE, 9.0, sensor_pose=lambda t: (-2 + 0.8 * t, 0.0, 0.08 * t),
        overrides=['registration.mode=odom'])
    return sc


def base_config(extra: Optional[List[str]] = None) -> Config:
    ov = ['registration.mode=stationary', 'ground.z_prior=-1.0', 'grid.size_x=90', 'grid.size_y=90',
          'registration.odom_stamp_mode=header', 'registration.extrinsic_xyz=[0,0,0]',
          'registration.fixed_frame=world']
    return load_config(None, ov + list(extra or []))


@dataclass
class ScenarioRun:
    spec: ScenarioSpec
    frames: list          # per frame: dict(t, gt, high, low_conf, blobs, clusters, sensor)


def run_scenario(spec: ScenarioSpec, extra_overrides: Optional[List[str]] = None, seed: int = 0,
                 sm: Optional[S.SensorModel] = None) -> ScenarioRun:
    cfg = base_config(spec.overrides + list(extra_overrides or []))
    pipe = TrackingPipeline(cfg, keep_history=False)
    sm = sm or S.SensorModel()
    frames = []
    odom_k = 0                       # next synthetic odometry sample index (100 Hz)
    for fr in S.generate(spec.objects, spec.static, spec.duration, sm=sm, sensor_pose=spec.sensor_pose,
                         seed=seed):
        if cfg.registration.mode in ('odom', 'auto'):
            # exact synthetic odometry at 100 Hz up to slightly beyond the scan stamp
            t0 = fr['stamp'] - fr['t']
            while odom_k * 0.01 <= fr['t'] + 0.02:
                tt = odom_k * 0.01
                sx, sy, syaw = spec.sensor_pose(tt) if spec.sensor_pose else (0, 0, 0)
                q = (0.0, 0.0, np.sin(syaw / 2), np.cos(syaw / 2))
                pipe.add_odom(t0 + tt, None, (sx, sy, sm.height), q, 'world', 'base')
                odom_k += 1
        r = pipe.process(Scan(fr['stamp'], fr['xyz'], 'lidar', fr['ring'], fr['points_per_ring']))
        # tracks are in the fixed frame; for stationary mode the fixed frame is the sensor frame
        # at the origin of the world (sensor at (0,0,yaw 0)), so world == fixed in xy
        frames.append({'t': fr['t'], 'gt': [g for g in fr['gt'] if abs(g[1]) < OUT / 2],
                       'high': [(s.id, s.x.copy(), s.status) for s in r.high_tracks],
                       'low_conf': [(s.id, s.x.copy()) for s in r.low_tracks if s.status != TENTATIVE],
                       'n_blobs': len(r.blobs), 'n_clusters': len(r.clusters), 'sensor': fr['sensor'],
                       'valid': r.valid, 'ms': r.timings.get('total', np.nan) * 1e3})
    return ScenarioRun(spec, frames)


def evaluate_scenario(run: ScenarioRun, gate: float = 4.0, range_limit: float = 30.0,
                      warm_in: float = 1.0) -> dict:
    """Match high-level tracks to GT per frame (Hungarian on distance, gate in m).

    position/velocity errors after `warm_in` seconds of each GT object's visibility
    (filter convergence), ID switches, fragmentation, false tracks, precision/recall.
    """
    first_seen: Dict[str, float] = {}
    assigned: Dict[str, int] = {}
    ids_per_gt: Dict[str, set] = {}
    switches = 0
    pos_err, vel_err, pos_err_all = [], [], []
    tp = fp = fn = 0
    false_ids = set()
    for fr in run.frames:
        t = fr['t']
        if t < run.spec.eval_from:
            continue
        sx, sy, _ = fr['sensor']
        gts = [g for g in fr['gt'] if np.hypot(g[1] - sx, g[2] - sy) < range_limit]
        hs = [h for h in fr['high'] if h[2] == CONFIRMED]
        if gts and hs:
            D = np.array([[np.hypot(g[1] - h[1][0], g[2] - h[1][1]) for h in hs] for g in gts])
            rr, cc = linear_sum_assignment(np.where(D < gate, D, 1e6))
            pairs = [(i, j) for i, j in zip(rr, cc) if D[i, j] < gate]
        else:
            pairs = []
        mg = {i for i, _ in pairs}
        mh = {j for _, j in pairs}
        tp += len(pairs)
        fn += len(gts) - len(mg)
        fp += len(hs) - len(mh)
        for j, h in enumerate(hs):
            if j not in mh:
                false_ids.add(h[0])
        for i, j in pairs:
            name = gts[i][0]
            tid = hs[j][0]
            first_seen.setdefault(name, t)
            ids_per_gt.setdefault(name, set()).add(tid)
            if name in assigned and assigned[name] != tid:
                switches += 1
            assigned[name] = tid
            e = np.hypot(gts[i][1] - hs[j][1][0], gts[i][2] - hs[j][1][1])
            pos_err_all.append(e)
            if t - first_seen[name] >= warm_in:
                pos_err.append(e)
                vel_err.append(np.hypot(gts[i][3] - hs[j][1][2], gts[i][4] - hs[j][1][3]))
    n_gt = len({g[0] for fr in run.frames for g in fr['gt']})

    def rmse(v):
        return float(np.sqrt(np.mean(np.square(v)))) if v else float('nan')
    return {'scenario': run.spec.name, 'gt_objects': n_gt,
            'position_rmse_m': rmse(pos_err), 'velocity_rmse_mps': rmse(vel_err),
            'position_rmse_all_m': rmse(pos_err_all),
            'id_switches': switches,
            'track_ids_per_gt': {k: len(v) for k, v in ids_per_gt.items()},
            'false_track_ids': len(false_ids), 'false_track_frames': fp,
            'precision': tp / (tp + fp) if tp + fp else float('nan'),
            'recall': tp / (tp + fn) if tp + fn else float('nan'),
            'max_high_tracks': max((len(f['high']) for f in run.frames), default=0),
            'final_high_tracks': len(run.frames[-1]['high']) if run.frames else 0,
            'mean_ms_per_frame': float(np.nanmean([f['ms'] for f in run.frames])),
            'invalid_frames': sum(not f['valid'] for f in run.frames)}
