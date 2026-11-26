"""Position-velocity clustering of low-level tracks (paper Sec. 3.4, Eqs. 16-17).

    delta_d(p, C) = min_{q in C} || pos_p - pos_q ||          (Eq. 16)
    delta_v(p, C) = max_{q in C} || vel_p - vel_q ||          (Eq. 17)

Procedure (paper, steps 1-4):
  1. randomly select a track as the seed of cluster C_i;
  2. for every unclustered track p compute delta_d(p, C_i), delta_v(p, C_i);
     p joins C_i if both are below their thresholds;
  3. if unclustered tracks remain, randomly select one as the next seed and
     repeat 2;
  4. stop when every track belongs to a cluster.

Interpretation (documented in PAPER_REPRODUCTION.md): step 2 is iterated
until no track joins (growth='fixpoint'); the paper's text admits a single
sweep, available as growth='single_pass'. With fixpoint growth, a chain of
blobs of one long vehicle is collected even if the seed is at one end.
Seeds use a NumPy Generator with a fixed seed -> reproducible.

`use_velocity=False` is the *position-only* ablation (not the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import ClusteringConfig


@dataclass
class Cluster:
    id: int
    members: List[int]            # indices into the input arrays
    member_ids: List[int]         # low-level track ids
    position: np.ndarray          # mean (2,)
    velocity: np.ndarray          # mean (2,)
    omega: float                  # mean turn rate

    @property
    def n_members(self) -> int:
        return len(self.members)


def cluster_tracks(pos: np.ndarray, vel: np.ndarray, omega: np.ndarray, ids: List[int],
                   cfg: ClusteringConfig, rng: np.random.Generator) -> List[Cluster]:
    n = len(pos)
    if n == 0:
        return []
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    V = np.linalg.norm(vel[:, None, :] - vel[None, :, :], axis=2)
    unclustered = np.ones(n, bool)
    clusters: List[Cluster] = []
    while unclustered.any():
        cand = np.flatnonzero(unclustered)
        seed = int(rng.choice(cand))                                   # steps 1 / 3
        members = [seed]
        unclustered[seed] = False
        changed = True
        while changed:
            changed = False
            for p in np.flatnonzero(unclustered):                       # step 2
                dd = D[p, members].min()                                # Eq. (16)
                ok = dd < cfg.tau_d
                if ok and cfg.use_velocity:
                    ok = V[p, members].max() < cfg.tau_v                # Eq. (17)
                if ok:
                    members.append(int(p))
                    unclustered[p] = False
                    changed = True
            if cfg.growth == 'single_pass':
                break
        m = np.array(members)
        clusters.append(Cluster(len(clusters), members, [ids[k] for k in members],
                                pos[m].mean(axis=0), vel[m].mean(axis=0), float(omega[m].mean())))
    return clusters
