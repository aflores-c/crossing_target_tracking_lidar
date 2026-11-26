"""CT-EKF (paper Eqs. 10-13, 18)."""
import numpy as np
import pytest

from crossing_target_tracking.core.ct_ekf import (CTEKF, H_POS, H_POS_VEL, batch_mahalanobis2, ct_jacobian,
                                                  ct_process_noise, ct_transition)


def numeric_jacobian(x, T, h=1e-6):
    J = np.zeros((5, 5))
    for k in range(5):
        d = np.zeros(5)
        d[k] = h
        J[:, k] = (ct_transition(x + d, T) - ct_transition(x - d, T)) / (2 * h)
    return J


@pytest.mark.parametrize('omega', [0.0, 1e-9, 5e-5, 1e-3, 0.5, -1.2])
def test_jacobian_matches_finite_differences(omega):
    x = np.array([3.0, -2.0, 7.0, -1.5, omega])
    for T in (0.05, 0.1, 0.5):
        np.testing.assert_allclose(ct_jacobian(x, T), numeric_jacobian(x, T), atol=2e-5)


def test_transition_equation_12_explicit():
    x = np.array([1.0, 2.0, 3.0, 4.0, 0.7])
    T = 0.3
    w = 0.7
    s, c = np.sin(w * T), np.cos(w * T)
    expected = [1 + 3 / w * s - 4 / w * (1 - c), 2 + 3 / w * (1 - c) + 4 / w * s, 3 * c - 4 * s, 3 * s + 4 * c, w]
    np.testing.assert_allclose(ct_transition(x, T), expected, rtol=1e-12)


def test_omega_zero_limit_is_constant_velocity():
    x = np.array([1.0, 2.0, 3.0, -4.0, 0.0])
    np.testing.assert_allclose(ct_transition(x, 0.5), [2.5, 0.0, 3.0, -4.0, 0.0], atol=1e-12)
    # Taylor branch == exact formula at the same small omega (state and Jacobian)
    xs = np.r_[x[:4], 0.9e-4]
    np.testing.assert_allclose(ct_transition(xs, 0.5, eps=1e-4), ct_transition(xs, 0.5, eps=1e-9), atol=1e-11)
    np.testing.assert_allclose(ct_jacobian(xs, 0.5, eps=1e-4), ct_jacobian(xs, 0.5, eps=1e-9), atol=1e-8)


def test_speed_preserved_by_turn():
    x = np.array([0.0, 0.0, 5.0, 0.0, 0.4])
    for _ in range(50):
        x = ct_transition(x, 0.1)
    assert np.hypot(x[2], x[3]) == pytest.approx(5.0, rel=1e-12)


def test_tracks_a_circle_and_estimates_turn_rate():
    rng = np.random.default_rng(1)
    v, w, T = 6.0, 0.3, 0.1
    f = CTEKF(np.array([0, 0, 0, 0, 0.0]), np.diag([0.1, 0.1, 25, 25, 0.25]), sigma_a=1.0, sigma_w=0.3)
    R = np.eye(2) * 0.1 ** 2
    for k in range(1, 120):
        t = k * T
        z = np.array([v / w * np.sin(w * t), v / w * (1 - np.cos(w * t))]) + rng.normal(0, 0.1, 2)
        f.predict(T)
        assert f.update(z, H_POS, R)
    assert f.x[4] == pytest.approx(w, abs=0.05)
    assert np.hypot(f.x[2], f.x[3]) == pytest.approx(v, abs=0.3)
    assert np.all(np.linalg.eigvalsh(f.P) > 0)


def test_position_velocity_update_eq18():
    f = CTEKF(np.zeros(5), np.eye(5), 1.0, 0.1)
    assert H_POS_VEL.shape == (4, 5) and np.allclose(H_POS_VEL[:, :4], np.eye(4)) and not H_POS_VEL[:, 4].any()
    f.update(np.array([1.0, 1.0, 2.0, 0.0]), H_POS_VEL, np.eye(4) * 1e-4)
    np.testing.assert_allclose(f.x[:4], [1, 1, 2, 0], atol=1e-3)


def test_nonfinite_measurement_rejected():
    f = CTEKF(np.zeros(5), np.eye(5), 1.0, 0.1)
    x0, P0 = f.x.copy(), f.P.copy()
    assert not f.update(np.array([np.nan, 1.0]), H_POS, np.eye(2))
    assert not f.update(np.array([np.inf, 1.0]), H_POS, np.eye(2))
    np.testing.assert_array_equal(f.x, x0)
    np.testing.assert_array_equal(f.P, P0)


def test_process_noise_psd_and_batch_mahalanobis():
    Q = ct_process_noise(0.1, 2.0, 0.5)
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-15)
    xs = np.array([[0, 0, 1, 0, 0], [5, 5, 0, 0, 0.0]])
    Ps = np.stack([np.eye(5), 2 * np.eye(5)])
    zs = np.array([[1.0, 0.0], [5.0, 7.0]])
    R = np.eye(2) * 0.5
    D = batch_mahalanobis2(xs, Ps, zs, H_POS, R)
    for j in range(2):
        f = CTEKF(xs[j], Ps[j], 1, 1)
        for i in range(2):
            assert D[j, i] == pytest.approx(f.mahalanobis2(zs[i], H_POS, R))
