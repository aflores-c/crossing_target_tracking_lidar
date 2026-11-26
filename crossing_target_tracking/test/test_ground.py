"""RANSAC ground removal (Sec. 3.1.1), not assuming a horizontal plane."""
import numpy as np
import pytest

from crossing_target_tracking.core.config import GroundConfig
from crossing_target_tracking.core.ground import GroundRemover


def plane_cloud(n_ground=4000, tilt_deg=6.0, h=0.7, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-20, 20, (n_ground, 2))
    t = np.radians(tilt_deg)
    z = -h + np.tan(t) * xy[:, 0] + rng.normal(0, 0.02, n_ground)
    ground = np.c_[xy, z]
    box = np.c_[rng.uniform(5, 6, 600), rng.uniform(-1, 1, 600), rng.uniform(-0.4, 1.0, 600)]
    return ground, box


def test_tilted_plane_found_and_obstacles_kept():
    ground, box = plane_cloud()
    gr = GroundRemover(GroundConfig(z_prior=-0.7, z_window=3.0))
    res = gr(np.r_[ground, box])
    assert res.plane.source == 'ransac'
    assert res.plane.tilt_deg == pytest.approx(6.0, abs=0.5)
    assert res.ground_mask[:len(ground)].mean() > 0.97
    assert res.obstacle_mask[len(ground):].mean() > 0.6
    assert res.obstacle_mask[:len(ground)].mean() < 0.02


def test_tilt_constraint_rejects_walls():
    rng = np.random.default_rng(1)
    wall = np.c_[np.full(3000, 4.0), rng.uniform(-10, 10, 3000), rng.uniform(-0.7, 3, 3000)]
    gr = GroundRemover(GroundConfig(z_prior=-0.7, z_window=0, min_inliers=100, max_tilt_deg=20))
    res = gr(wall)
    assert res.plane.source == 'prior'          # never accept a vertical plane


def test_sparse_cloud_falls_back_to_previous_plane():
    ground, _ = plane_cloud(tilt_deg=3.0)
    gr = GroundRemover(GroundConfig(z_prior=-0.7))
    p1 = gr(ground).plane
    res = gr(ground[:20])
    assert res.plane.source == 'previous'
    np.testing.assert_allclose(res.plane.normal, p1.normal)


def test_height_band():
    ground, _ = plane_cloud(tilt_deg=0.0)
    high = np.array([[5.0, 0.0, 4.0], [5.0, 0.0, 1.0]])     # 4.7 m and 1.7 m above ground
    res = GroundRemover(GroundConfig(z_prior=-0.7, max_height=3.0))(np.r_[ground, high])
    assert not res.obstacle_mask[-2] and res.obstacle_mask[-1]
