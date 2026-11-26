"""M/N confirmation and elimination (Sec. 3.3.3), dropped-frame handling."""
import numpy as np

from crossing_target_tracking.core.config import EKFConfig, TrackManagerConfig
from crossing_target_tracking.core.ct_ekf import H_POS
from crossing_target_tracking.core.tracker import CONFIRMED, PREDICTED, TENTATIVE, MultiTargetTracker


def make(mc=3, nc=5, me=2, ne=4, confirm=True):
    return MultiTargetTracker(EKFConfig(), TrackManagerConfig(confirm_m=mc, confirm_n=nc, delete_m=me, delete_n=ne,
                                                              use_confirmation=confirm),
                              H_POS, np.eye(2) * 0.04)


def z(t, present=True):
    return np.array([[t * 1.0, 0.0]]) if present else np.zeros((0, 2))


def test_confirmation_after_mc_hits_within_nc():
    tr = make()
    pattern = [1, 0, 1, 1]           # hits 3 in 4 instants -> confirmed at the 4th
    for k, p in enumerate(pattern):
        tr.step(k * 0.1, z(k * 0.1, p))
        st = tr.tracks[0].status
        assert st == (CONFIRMED if k == 3 else TENTATIVE)


def test_tentative_deleted_when_mc_unreachable():
    tr = make(mc=3, nc=5)
    tr.step(0.0, z(0))
    tr.step(0.1, z(0.1, False))
    tr.step(0.2, z(0.2, False))
    assert len(tr.tracks) == 1       # 1 hit + 2 remaining instants can still give 3
    tr.step(0.3, z(0.3, False))
    assert len(tr.tracks) == 0


def test_elimination_me_ne_and_coasting():
    tr = make(mc=2, nc=3, me=2, ne=4)
    for k in range(5):
        tr.step(k * 0.1, z(k * 0.1))
    tid = tr.tracks[0].id
    tr.step(0.5, z(0.5, False))
    assert tr.tracks[0].status == PREDICTED     # temporary miss keeps the track
    tr.step(0.6, z(0.6, False))
    assert tr.tracks[0].id == tid               # last 4: [1,1,0,0] -> 2 >= Me
    tr.step(0.7, z(0.7, False))                 # last 4: [1,0,0,0] -> 1 < Me -> deleted
    assert not tr.tracks


def test_no_confirmation_mode_high_level():
    tr = make(confirm=False, me=1, ne=3)
    tr.step(0.0, z(0))
    assert tr.tracks[0].status == CONFIRMED


def test_dropped_frames_count_as_missed_instants():
    tr = make(mc=2, nc=3, me=2, ne=4)
    for k in range(4):
        tr.step(k * 0.1, z(k * 0.1))
    tr.step(0.7, z(0.7, False), missed_instants=3)  # 3 dropped scans + this empty one
    assert not tr.tracks


def test_ids_persistent_and_unique():
    tr = make(mc=1, nc=1)
    for k in range(5):
        tr.step(k * 0.1, np.array([[k * 1.0, 0.0], [k * 1.0, 20.0]]))
    assert sorted(t.id for t in tr.tracks) == [1, 2]
    assert all(len(t.history) == 5 for t in tr.tracks)
