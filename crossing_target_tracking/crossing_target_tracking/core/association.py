"""Gating and Global Nearest Neighbour association (paper Sec. 3.3.2, Eqs. 14-15).

cost(j, i) = d_M^{i,j} = (y_i - y_hat_j)^T S_j^{-1} (y_i - y_hat_j)    (Eq. 14)
A pair is admissible only if d_M <= tau_G                              (Eq. 15)
Inadmissible pairs get an "infinite" cost and the global assignment is
solved with the Hungarian / Munkres algorithm
(scipy.optimize.linear_sum_assignment). Assignments that land on an
inadmissible pair are discarded (-> unmatched track and measurement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9


@dataclass
class AssociationResult:
    matches: List[Tuple[int, int]]     # (track index, measurement index)
    unmatched_tracks: List[int]
    unmatched_measurements: List[int]
    cost: np.ndarray


def gnn_associate(cost: np.ndarray, gate: float) -> AssociationResult:
    n_t, n_m = cost.shape
    if n_t == 0 or n_m == 0:
        return AssociationResult([], list(range(n_t)), list(range(n_m)), cost)
    c = np.where(np.isfinite(cost) & (cost <= gate), cost, BIG)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols) if c[r, k] < BIG]
    mt = {r for r, _ in matches}
    mm = {k for _, k in matches}
    return AssociationResult(matches, [j for j in range(n_t) if j not in mt],
                             [i for i in range(n_m) if i not in mm], cost)
