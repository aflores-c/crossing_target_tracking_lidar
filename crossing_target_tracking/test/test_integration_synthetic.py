"""End-to-end pipeline on synthetic VLP-16 sequences (experiments 1-8 of the spec).

Thresholds are deliberately looser than the values reported in
docs/results/synthetic/synthetic_results.md; they guard against regressions.
"""
import numpy as np
import pytest

from crossing_target_tracking.tools.scenarios import evaluate_scenario, run_scenario, scenarios

_cache = {}


def result(name, extra=()):
    key = (name, tuple(extra))
    if key not in _cache:
        run = run_scenario(scenarios()[name], list(extra))
        _cache[key] = (run, evaluate_scenario(run))
    return _cache[key]


def test_1_static_scene_no_tracks():
    run, m = result('s1_static')
    assert m['max_high_tracks'] == 0
    assert all(f['n_blobs'] == 0 for f in run.frames[15:])


def test_2_constant_speed():
    run, m = result('s2_constant_speed')
    assert m['recall'] >= 0.9 and m['track_ids_per_gt']['car'] <= 2
    assert m['velocity_rmse_mps'] < 1.6
    speeds = [np.hypot(*h[1][2:4]) for f in run.frames if 4 < f['t'] < 7 for h in f['high'] if h[2] == 1
              and abs(h[1][1] - 7.5) < 1.5]
    assert np.median(speeds) == pytest.approx(10.0, abs=1.0)


def test_3_turning_vehicle_turn_rate_sign():
    run, m = result('s3_turning')
    assert m['recall'] >= 0.6 and m['track_ids_per_gt']['car'] <= 4
    om = [h[1][4] for f in run.frames if f['t'] > 6.5 for h in f['high'] if h[2] == 1 and np.hypot(*h[1][2:4]) > 3]
    assert np.mean(om) < -0.05                       # right turn (truth -0.3 rad/s)


def test_4_two_vehicles_different_speed_both_tracked():
    _, m = result('s4_two_vehicles')
    assert set(m['track_ids_per_gt']) == {'car', 'motorcycle'} and m['recall'] >= 0.5


def test_4b_velocity_clustering_vs_position_only_ablation():
    _, m = result('s4b_two_pedestrians')
    assert m['track_ids_per_gt'] == {'ped_a': 1, 'ped_b': 1} and m['id_switches'] == 0
    assert m['velocity_rmse_mps'] < 0.3
    _, ab = result('s4b_two_pedestrians', ['clustering.use_velocity=false'])
    assert ab['id_switches'] > m['id_switches'] or ab['velocity_rmse_mps'] > 2 * m['velocity_rmse_mps']


def test_5_occlusion_and_redetection():
    _, m = result('s5_occlusion')
    assert m['recall'] >= 0.6 and m['track_ids_per_gt']['cyclist'] <= 2


def test_6_multiple_road_users():
    _, m = result('s6_multi')
    assert set(m['track_ids_per_gt']) == {'car_east', 'car_west', 'cyclist', 'pedestrian'}
    assert m['recall'] >= 0.7 and m['precision'] >= 0.8


def test_7_background_change_absorbed():
    run, m = result('s7_background_change')
    assert m['final_high_tracks'] == 0               # parked car and new box became background
    assert run.frames[-1]['n_blobs'] == 0


def test_8_moving_sensor_with_odometry():
    _, m = result('s8_moving_sensor')
    assert m['invalid_frames'] == 0
    assert m['track_ids_per_gt']['car'] == 1 and m['recall'] >= 0.9
    assert m['velocity_rmse_mps'] < 1.6
    assert m['false_track_ids'] <= 4                 # static scene seen from a moving sensor


def test_8b_moving_sensor_without_compensation_is_worse():
    """Ignoring ego-motion (stationary mode on moving data) must degrade the result."""
    _, good = result('s8_moving_sensor')
    _, bad = result('s8_moving_sensor', ['registration.mode=stationary'])
    assert bad['false_track_ids'] > good['false_track_ids'] or bad['velocity_rmse_mps'] > good['velocity_rmse_mps']
