"""Position-velocity clustering (Sec. 3.4, Eqs. 16-17)."""
import numpy as np

from crossing_target_tracking.core.clustering import cluster_tracks
from crossing_target_tracking.core.config import ClusteringConfig


def run(pos, vel, **kw):
    cfg = ClusteringConfig(**{'tau_d': 1.5, 'tau_v': 1.5, **kw})
    return cluster_tracks(np.array(pos, float), np.array(vel, float), np.zeros(len(pos)),
                          list(range(10, 10 + len(pos))), cfg, np.random.default_rng(0))


def as_sets(cl):
    return sorted(sorted(c.member_ids) for c in cl)


def test_blobs_of_one_vehicle_merge_and_average():
    cl = run([[0, 0], [1.0, 0], [2.0, 0.1]], [[8, 0], [8.4, 0.1], [7.8, 0]])
    assert as_sets(cl) == [[10, 11, 12]]
    np.testing.assert_allclose(cl[0].position, [1.0, 0.1 / 3])
    np.testing.assert_allclose(cl[0].velocity, [8.0666667, 1 / 30], rtol=1e-6)
    assert cl[0].n_members == 3


def test_adjacent_objects_with_different_velocity_split():
    pos = [[0, 0], [0.8, 0]]
    vel = [[1.4, 0], [-1.4, 0]]
    assert as_sets(run(pos, vel)) == [[10], [11]]
    assert as_sets(run(pos, vel, use_velocity=False)) == [[10, 11]]   # position-only ablation merges


def test_min_distance_chains_and_max_velocity_is_against_all_members():
    # chain 0-1-2 with 1.2 m spacing (min distance, Eq. 16)
    assert as_sets(run([[0, 0], [1.2, 0], [2.4, 0]], [[5, 0]] * 3)) == [[10, 11, 12]]
    # 2 is close in velocity to 1 but not to 0 -> Eq. 17 (max) excludes it from {0, 1}
    cl = run([[0, 0], [1.0, 0], [2.0, 0]], [[5, 0], [6.2, 0], [7.4, 0]])
    assert all(max(c.velocity) for c in cl)
    for c in cl:
        v = np.array([[5, 0], [6.2, 0], [7.4, 0]])[c.members]
        assert np.max(np.linalg.norm(v[:, None] - v[None], axis=2)) < 1.5


def test_every_track_in_exactly_one_cluster_and_reproducible():
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 20, (30, 2))
    vel = rng.normal(0, 3, (30, 2))
    a = run(pos, vel)
    b = run(pos, vel)
    ids = sorted(i for c in a for i in c.member_ids)
    assert ids == list(range(10, 40))
    assert as_sets(a) == as_sets(b)


def test_fixpoint_vs_single_pass():
    # seed order matters for single_pass when the chain is built from the far end
    pos = [[0, 0], [1.2, 0], [2.4, 0], [3.6, 0]]
    vel = [[5, 0]] * 4
    for seed in range(5):
        cfg = ClusteringConfig(tau_d=1.5, tau_v=1.5, growth='fixpoint')
        cl = cluster_tracks(np.array(pos, float), np.array(vel, float), np.zeros(4), [0, 1, 2, 3], cfg,
                            np.random.default_rng(seed))
        assert len(cl) == 1


def test_empty():
    assert run(np.zeros((0, 2)), np.zeros((0, 2))) == []
