"""Generic relational motion features computed from normalised joint trajectories.

This is a perceptual front-end only: it does not touch the LoT. Every
feature family is defined by one rule applied uniformly to a fixed, generic
set of body points. No feature is designed for a particular manner.

Body points (from the pose estimator's landmarks):
    head, neck (shoulder midpoint), pelvis (hip midpoint),
    L/R elbow, L/R wrist, L/R knee, L/R ankle

Feature families (all in body-height units or radians; invariant to
position in the frame, translation and body size by construction):
  torso      trunk tilt from vertical, trunk length (bending/compression), shoulder yaw (from 3-D)
  bilateral  for each L/R point pair: horizontal and vertical offset L-R, in the torso frame
  distance   distance between every pair of body points
  crossing   for each L/R limb-segment pair: signed overlap (positive = the segments intersect)
  com        pelvis height relative to its median, its velocity, body vertical extent
  limb_rel   each limb point's position in the torso frame (x across the trunk, y along it)
  dynamics   frame-to-frame speed of each limb point in the torso frame
Derived per-sequence summaries (not time series):
  periodicity  dominant period and autocorrelation strength of every feature
  alternation  correlation between the left and right members of each pair (-1 = anti-phase)
  qualitative  per feature, the sign of change over time in {-1, 0, +1} with a dead zone
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List

import numpy as np

from grace_lot.grace.trajectories import Trajectory, resample

NAMES = ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
MP = {"nose": 0, "left_ear": 7, "right_ear": 8, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13,
      "right_elbow": 14, "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
      "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28}
POINTS = ["head", "neck", "pelvis", "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_knee", "R_knee", "L_ankle", "R_ankle"]
PAIRS = [("L_elbow", "R_elbow"), ("L_wrist", "R_wrist"), ("L_knee", "R_knee"), ("L_ankle", "R_ankle")]
SEGMENTS = {"upper_arm": ("elbow", "shoulder"), "forearm": ("wrist", "elbow"),
            "thigh": ("knee", "hip"), "shank": ("ankle", "knee")}
FAMILIES = ["torso", "bilateral", "distance", "crossing", "com", "limb_rel", "dynamics"]


@dataclass
class Relational:
    names: List[str]            # feature names, each prefixed with its family, e.g. "bilateral:ankle_dx"
    values: np.ndarray          # (T, K)
    summary: Dict[str, float]   # periodicity / alternation descriptors
    qualitative: np.ndarray     # (T-1, K) in {-1, 0, +1}

    def family(self, fam: str) -> np.ndarray:
        idx = [i for i, n in enumerate(self.names) if n.startswith(fam + ":")]
        return self.values[:, idx]


def _points(joints: np.ndarray) -> Dict[str, np.ndarray]:
    """Generic body points, (T, 2) each, from (T, 33, 2) canonical landmarks (y points down)."""
    j = {n: joints[:, MP[n]] for n in NAMES}
    head = (j["nose"] + j["left_ear"] + j["right_ear"]) / 3
    neck = (j["left_shoulder"] + j["right_shoulder"]) / 2
    pelvis = (j["left_hip"] + j["right_hip"]) / 2
    out = {"head": head, "neck": neck, "pelvis": pelvis}
    for side, s in (("L", "left"), ("R", "right")):
        for part in ("elbow", "wrist", "knee", "ankle", "shoulder", "hip"):
            out[f"{side}_{part}"] = j[f"{s}_{part}"]
    return out


def _torso_frame(p: Dict[str, np.ndarray]):
    """Unit vectors per frame: up (pelvis -> neck) and across (perpendicular), in image coordinates."""
    up = p["neck"] - p["pelvis"]
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-9)
    across = np.stack([-up[:, 1], up[:, 0]], axis=1)
    return up, across


def _seg_overlap(a0, a1, b0, b1) -> np.ndarray:
    """Signed overlap of 2-D segments per frame: >0 where they intersect, else -(minimum distance)."""
    def cross(o, a, b):
        return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])
    d1, d2 = cross(b0, b1, a0), cross(b0, b1, a1)
    d3, d4 = cross(a0, a1, b0), cross(a0, a1, b1)
    intersect = (d1 * d2 < 0) & (d3 * d4 < 0)

    def pt_seg(p, s0, s1):
        v = s1 - s0
        t = np.clip(((p - s0) * v).sum(1) / ((v * v).sum(1) + 1e-9), 0, 1)
        return np.linalg.norm(p - (s0 + t[:, None] * v), axis=1)
    dmin = np.min([pt_seg(a0, b0, b1), pt_seg(a1, b0, b1), pt_seg(b0, a0, a1), pt_seg(b1, a0, a1)], axis=0)
    return np.where(intersect, 0.05, -dmin)


def relational_features(joints: np.ndarray, yaw: np.ndarray) -> Relational:
    """(T, 33, 2) canonical joints and (T,) yaw -> relational time series."""
    p = _points(joints)
    up, across = _torso_frame(p)
    to_torso = lambda v: np.stack([(v * across).sum(1), -(v * up).sum(1)], axis=1)   # x across, y along (down +)
    F: Dict[str, np.ndarray] = {}

    trunk = p["neck"] - p["pelvis"]
    F["torso:tilt"] = np.arctan2(trunk[:, 0], -trunk[:, 1])          # 0 = upright, + = leaning towards +x
    F["torso:length"] = np.linalg.norm(trunk, axis=1)
    F["torso:yaw"] = yaw - np.median(yaw)

    for a, b in PAIRS:
        rel = to_torso(p[a] - p[b])
        name = a.split("_")[1]
        F[f"bilateral:{name}_dx"] = rel[:, 0]
        F[f"bilateral:{name}_dy"] = rel[:, 1]

    for a, b in combinations(POINTS, 2):
        F[f"distance:{a}-{b}"] = np.linalg.norm(p[a] - p[b], axis=1)

    for seg, (end, root) in SEGMENTS.items():
        F[f"crossing:{seg}"] = _seg_overlap(p[f"L_{end}"], p[f"L_{root}"], p[f"R_{end}"], p[f"R_{root}"])
    F["crossing:leg_order"] = np.sign(to_torso(p["L_ankle"] - p["R_ankle"])[:, 0])

    py = p["pelvis"][:, 1]
    F["com:pelvis_height"] = -(py - np.median(py))                   # + = higher than usual
    F["com:pelvis_vvel"] = -np.gradient(py)
    allpts = np.stack([v for v in p.values()], axis=1)
    F["com:vertical_extent"] = allpts[:, :, 1].max(1) - allpts[:, :, 1].min(1)

    for name in ("elbow", "wrist", "knee", "ankle"):
        for side in ("L", "R"):
            rel = to_torso(p[f"{side}_{name}"] - p["pelvis"])
            F[f"limb_rel:{side}_{name}_x"] = rel[:, 0]
            F[f"limb_rel:{side}_{name}_y"] = rel[:, 1]
            F[f"dynamics:{side}_{name}_speed"] = np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=1)]

    names = list(F)
    values = np.stack([F[n] for n in names], axis=1)
    return Relational(names, values, _summaries(names, values), _qualitative(values))


def _summaries(names: List[str], values: np.ndarray) -> Dict[str, float]:
    """Dominant period (frames) and its autocorrelation per feature; L/R alternation per pair."""
    out = {}
    T = len(values)
    for i, n in enumerate(names):
        x = values[:, i] - values[:, i].mean()
        if x.std() < 1e-6:
            out[f"period:{n}"], out[f"periodicity:{n}"] = 0.0, 0.0
            continue
        ac = np.correlate(x, x, "full")[T - 1:] / (x @ x)
        lag = 2 + int(np.argmax(ac[2:T // 2])) if T > 6 else 0
        out[f"period:{n}"], out[f"periodicity:{n}"] = float(lag), float(ac[lag])
    for name in ("elbow", "wrist", "knee", "ankle"):
        for axis in ("x", "y"):
            L, R = (values[:, names.index(f"limb_rel:{s}_{name}_{axis}")] for s in ("L", "R"))
            ok = L.std() > 1e-6 and R.std() > 1e-6
            out[f"alternation:{name}_{axis}"] = float(np.corrcoef(L, R)[0, 1]) if ok else 0.0
    return out


def _qualitative(values: np.ndarray, dead_zone: float = 0.1) -> np.ndarray:
    """Sign of change per feature; changes smaller than dead_zone x the feature's own range count as 0."""
    d = np.diff(values, axis=0)
    rng = values.max(0) - values.min(0) + 1e-9
    return np.where(np.abs(d) < dead_zone * rng, 0, np.sign(d)).astype(int)


def from_trajectory(traj: Trajectory, frames: int = 32) -> Relational:
    """Relational features from a stored Trajectory, with the full-rate joints resampled to ``frames``."""
    joints = resample(traj.joints_full, frames)
    yaw = resample(traj.yaw, frames) if len(traj.yaw) != frames else traj.yaw
    return relational_features(joints, yaw)
