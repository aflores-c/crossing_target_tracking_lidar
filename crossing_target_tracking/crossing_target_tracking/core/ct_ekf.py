"""Coordinated-Turn Extended Kalman Filter (paper Sec. 3.3.1 and 3.5).

State (Eq. 10):  x = [px, py, vx, vy, omega]^T
Transition (Eq. 12), T = dt:
    px' = px + vx/w sin(wT) - vy/w (1 - cos(wT))
    py' = py + vx/w (1 - cos(wT)) + vy/w sin(wT)
    vx' = vx cos(wT) - vy sin(wT)
    vy' = vx sin(wT) + vy cos(wT)
    w'  = w
Measurement: low level  H  (Eq. 13): z = [px, py]
             high level H2 (Eq. 18): z = [px, py, vx, vy]

For |w| < omega_eps the terms sin(wT)/w and (1-cos(wT))/w and their
w-derivatives are evaluated with Taylor series (continuous at w -> 0, the
model reduces to constant velocity). The Jacobian is analytic and is
checked against finite differences in test/test_ct_ekf.py.

Process noise (not specified by the paper; engineering choice): white
acceleration sigma_a on (vx, vy) discretised as in the CV model, and white
turn-rate acceleration sigma_omega on omega:
    Q = blockdiag over axes of sigma_a^2 [[T^4/4, T^3/2], [T^3/2, T^2]],
        Q_ww = sigma_omega^2 T^2.
Correction uses the Joseph form to keep P symmetric positive definite.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

H_POS = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]], float)                       # Eq. (13)
H_POS_VEL = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0]], float)                   # Eq. (18)


def _ab(w: float, T: float, eps: float):
    """A = sin(wT)/w, B = (1-cos(wT))/w and dA/dw, dB/dw."""
    if abs(w) < eps:
        A = T - w * w * T ** 3 / 6.0
        B = w * T * T / 2.0 - w ** 3 * T ** 4 / 24.0
        dA = -w * T ** 3 / 3.0
        dB = T * T / 2.0 - w * w * T ** 4 / 8.0
    else:
        s, c = np.sin(w * T), np.cos(w * T)
        A = s / w
        B = (1.0 - c) / w
        dA = (w * T * c - s) / (w * w)
        dB = (w * T * s - (1.0 - c)) / (w * w)
    return A, B, dA, dB


def ct_transition(x: np.ndarray, T: float, eps: float = 1e-4) -> np.ndarray:
    """f(x) of Eq. (12)."""
    px, py, vx, vy, w = x
    A, B, _, _ = _ab(w, T, eps)
    s, c = np.sin(w * T), np.cos(w * T)
    return np.array([px + A * vx - B * vy,
                     py + B * vx + A * vy,
                     c * vx - s * vy,
                     s * vx + c * vy,
                     w])


def ct_jacobian(x: np.ndarray, T: float, eps: float = 1e-4) -> np.ndarray:
    _, _, vx, vy, w = x
    A, B, dA, dB = _ab(w, T, eps)
    s, c = np.sin(w * T), np.cos(w * T)
    return np.array([
        [1, 0, A, -B, dA * vx - dB * vy],
        [0, 1, B, A, dB * vx + dA * vy],
        [0, 0, c, -s, -T * s * vx - T * c * vy],
        [0, 0, s, c, T * c * vx - T * s * vy],
        [0, 0, 0, 0, 1]], float)


def ct_process_noise(T: float, sigma_a: float, sigma_w: float) -> np.ndarray:
    q = sigma_a ** 2
    Q = np.zeros((5, 5))
    Q[0, 0] = Q[1, 1] = q * T ** 4 / 4.0
    Q[2, 2] = Q[3, 3] = q * T ** 2
    Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = q * T ** 3 / 2.0
    Q[4, 4] = sigma_w ** 2 * T ** 2
    return Q


class CTEKF:
    """One CT-EKF track state."""

    def __init__(self, x0: np.ndarray, P0: np.ndarray, sigma_a: float, sigma_w: float,
                 omega_max: float = 2.0, eps: float = 1e-4):
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.sigma_a = sigma_a
        self.sigma_w = sigma_w
        self.omega_max = omega_max
        self.eps = eps

    def predict(self, T: float):
        if T <= 0:
            return
        F = ct_jacobian(self.x, T, self.eps)
        self.x = ct_transition(self.x, T, self.eps)
        self.x[4] = float(np.clip(self.x[4], -self.omega_max, self.omega_max))
        self.P = F @ self.P @ F.T + ct_process_noise(T, self.sigma_a, self.sigma_w)
        self.P = 0.5 * (self.P + self.P.T)

    def innovation(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S

    def mahalanobis2(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> float:
        """Eq. (14): d = (y - y_hat)^T S^-1 (y - y_hat)."""
        y, S = self.innovation(z, H, R)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> bool:
        z = np.asarray(z, float)
        if not np.all(np.isfinite(z)):
            return False
        y, S = self.innovation(z, H, R)
        try:
            K = np.linalg.solve(S, H @ self.P).T          # P H^T S^-1 (S symmetric)
        except np.linalg.LinAlgError:
            return False
        self.x = self.x + K @ y
        self.x[4] = float(np.clip(self.x[4], -self.omega_max, self.omega_max))
        IKH = np.eye(5) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T       # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return True


def batch_mahalanobis2(xs: np.ndarray, Ps: np.ndarray, zs: np.ndarray, H: np.ndarray,
                       R: np.ndarray) -> np.ndarray:
    """Eq. (14) for all (track j, measurement i) pairs. Returns (n_tracks, n_meas)."""
    if len(xs) == 0 or len(zs) == 0:
        return np.zeros((len(xs), len(zs)))
    zh = xs @ H.T                                         # (T, m)
    S = np.einsum('ab,tbc,dc->tad', H, Ps, H) + R         # (T, m, m)
    Sinv = np.linalg.inv(S)
    d = zs[None, :, :] - zh[:, None, :]                   # (T, Z, m)
    return np.einsum('tzi,tij,tzj->tz', d, Sinv, d)


def initial_state(z: np.ndarray, dim_z: int, init_sigma_p: float, init_sigma_v: float,
                  init_sigma_w: float, R: Optional[np.ndarray] = None):
    x = np.zeros(5)
    P = np.diag([init_sigma_p ** 2, init_sigma_p ** 2, init_sigma_v ** 2, init_sigma_v ** 2,
                 init_sigma_w ** 2])
    x[:dim_z] = z[:dim_z]
    if R is not None:
        P[:dim_z, :dim_z] = R
    return x, P
