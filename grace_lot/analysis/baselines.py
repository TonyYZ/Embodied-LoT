"""Theory-light kinematic baselines, computed from the same preprocessed data as the LoT.

  kin_fields      -euclid(field sequences): the exact representation the LoT is fitted to
  kin_joints_dtw  -DTW over canonical joint positions (every landmark weighted equally)
  kin_velocity    -DTW over canonical joint velocities (pose dynamics)

Each returns a similarity matrix (higher = more similar) with a NaN diagonal.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from grace_lot.grace.trajectories import Trajectory, resample
from grace_lot.lot_adapter.distance import dtw, euclid

DTW_FRAMES = 48     # joint sequences are resampled to this length before DTW


def _matrix(items: Sequence, dist: Callable) -> np.ndarray:
    n = len(items)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = -dist(items[i], items[j])
    return m


def kinematic_matrices(trajs: Sequence[Trajectory]) -> Dict[str, np.ndarray]:
    joints = [resample(t.joints_full, DTW_FRAMES) for t in trajs]
    velocity = [np.diff(j, axis=0) * DTW_FRAMES for j in joints]
    return {
        "kin_fields": _matrix([t.fields for t in trajs], euclid),
        "kin_joints_dtw": _matrix(joints, dtw),
        "kin_velocity": _matrix(velocity, dtw),
    }
