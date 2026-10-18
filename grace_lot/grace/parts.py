"""Skeleton body-part channels: the perceptual side of the multi-output interface.

Anatomy lives here only, never inside the LoT. Each part is rasterised into
its own field in the same canonical window as the occupancy representation,
giving a (T, P, H, W) array. The LoT's anonymous outputs are matched to these
channels by a permutation-invariant, temporally stable assignment
(inference/likelihood.py).

Every edge of the estimator's skeleton belongs to exactly one part:
  head       face landmarks (0-10)
  torso      shoulder-shoulder, hip-hip, shoulder-hip edges, quadrilateral filled
  left_arm   edges touching the left elbow/wrist/hand (13, 15, 17, 19, 21)
  right_arm  edges touching the right elbow/wrist/hand (14, 16, 18, 20, 22)
  left_leg   edges touching the left knee/ankle/foot (25, 27, 29, 31)
  right_leg  edges touching the right knee/ankle/foot (26, 28, 30, 32)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from grace_lot.grace.trajectories import TORSO, Trajectory, rasterise

PART_NAMES = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
_HEAD = set(range(11))
_TORSO_EDGES = {(11, 12), (11, 23), (12, 24), (23, 24)}
_JOINTS = {"left_arm": {13, 15, 17, 19, 21}, "right_arm": {14, 16, 18, 20, 22},
           "left_leg": {25, 27, 29, 31}, "right_leg": {26, 28, 30, 32}}


def part_of_edge(a: int, b: int) -> str:
    if a in _HEAD and b in _HEAD:
        return "head"
    if (min(a, b), max(a, b)) in _TORSO_EDGES:
        return "torso"
    for part, js in _JOINTS.items():
        if a in js or b in js:
            return part
    raise ValueError(f"edge {(a, b)} belongs to no part")


def part_connections(connections) -> Dict[str, List[Tuple[int, int]]]:
    out = {p: [] for p in PART_NAMES}
    for a, b in connections:
        out[part_of_edge(int(a), int(b))].append((int(a), int(b)))
    return out


def part_fields(traj: Trajectory, pre: dict) -> np.ndarray:
    """(T, P, H, W) per-part coverage fields from the trajectory's time-normalised joints."""
    conns = part_connections(traj.connections)
    torso_idx = [11, 12, 23, 24]                      # left/right shoulder, left/right hip (TORSO order)
    assert len(TORSO) == 4
    chans = []
    for p in PART_NAMES:
        chans.append(rasterise(traj.joints, conns[p], pre["field_size"], pre["window_x"], pre["window_y"],
                               pre["thickness"], pre["raster_scale"], torso_idx if p == "torso" else None))
    return np.stack(chans, axis=1)


def occupancy_background(cfg: dict) -> np.ndarray:
    """(H, W) base rate of body-part occupancy per cell: what a part window usually looks like.

    The mean of ``part_fields`` over every training video of every manner (all frames and parts), so it
    is shared by all programs and carries no manner-specific information. Used as the prediction for cells
    a program makes no claim about (docs/probabilistic_semantics.md). Cached in the processed directory.
    """
    import hashlib
    import json
    from pathlib import Path
    from grace_lot.grace.preprocess import load_trajectory
    from grace_lot.inference.run import videos_for
    pre = cfg["preprocess"]
    key = hashlib.sha1(json.dumps([pre, cfg.get("split", {})], sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = Path(cfg["data"]["processed_dir"]) / f"background_occupancy_{key}.npy"
    if path.exists():
        return np.load(path)
    bym, _ = videos_for(cfg, "train")
    total, count = np.zeros((pre["field_size"], pre["field_size"])), 0
    for m in sorted(bym):
        for v in bym[m]:
            f = part_fields(load_trajectory(cfg, v), pre)                # (T, P, H, W)
            total += f.sum(axis=(0, 1))
            count += f.shape[0] * f.shape[1]
    out = total / max(count, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out
