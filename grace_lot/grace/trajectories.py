"""Turn a raw pose sequence into normalised body-motion representations.

Pipeline (each step is a plain function, so it can be replaced):

    valid_segment   frames where the whole body is in view
    canonicalise    remove position, translation and body size
    resample        normalise duration to T frames
    rasterise       draw the estimator's own skeleton into T x H x W fields

Nuisance normalisation, which is the only place choices are made:
* scale     the median over the segment of the per-frame landmark bounding-box height
* x         taken relative to the per-frame torso centre (the mean of the
            shoulders and hips); this removes the walk across the screen
* y         taken relative to the *median* torso height of the video; this
            removes camera height but keeps bounces, crouches and drops
No joint and no movement dimension is weighted by hand. Every landmark and
every skeleton edge from the estimator enters the rasterisation equally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from grace_lot.grace.pose import PoseSequence

TORSO = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


@dataclass
class Trajectory:
    """The normalised motion of one video. Arrays with T rows are resampled to ``frames``."""
    joints: np.ndarray       # (T, J, 2) canonical landmark positions, in body heights
    centre: np.ndarray       # (T, 2) torso-centre displacement, in body heights
    yaw: np.ndarray          # (T,) shoulder-line angle in the world x-z plane (radians, unwrapped)
    fields: np.ndarray       # (T, H, W) rasterised skeleton coverage in [0, 1]
    joints_full: np.ndarray  # (F', J, 2) canonical landmarks for the valid segment at the original rate
    segment: Tuple[int, int] # [start, stop) of the valid frames in the original video
    scale_px: float          # body height in pixels
    connections: np.ndarray  # (E, 2)

    def save(self, path) -> None:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path) -> "Trajectory":
        d = np.load(path)
        return cls(**{k: (tuple(d[k]) if k == "segment" else float(d[k]) if k == "scale_px" else d[k])
                      for k in cls.__dataclass_fields__})


def valid_segment(pose: PoseSequence, min_visibility: float, edge_margin: float,
                  max_gap: int = 3) -> Tuple[int, int]:
    """The longest run of frames with the whole body in view. Gaps of up to ``max_gap`` frames are bridged."""
    w, h = pose.size
    xy = pose.image / np.array([w, h])
    inside = np.all((xy > edge_margin) & (xy < 1 - edge_margin), axis=(1, 2))
    ok = inside & (pose.visibility.mean(axis=1) >= min_visibility)
    best, start, gap = (0, 0), None, 0
    for i, good in enumerate(list(ok) + [False] * (max_gap + 1)):
        if good:
            start = i if start is None else start
            gap, last = 0, i
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if last + 1 - start > best[1] - best[0]:
                    best = (start, last + 1)
                start, gap = None, 0
    if best[1] - best[0] < 2:
        raise ValueError("no usable segment: the body is never fully in view")
    return best


def _fill_gaps(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate rows that are all NaN (missed detections)."""
    a = a.copy()
    flat = a.reshape(len(a), -1)
    bad = np.isnan(flat).any(axis=1)
    if bad.any():
        idx = np.arange(len(a))
        for k in range(flat.shape[1]):
            flat[bad, k] = np.interp(idx[bad], idx[~bad], flat[~bad, k])
    return flat.reshape(a.shape)


def canonicalise(pose: PoseSequence, segment: Tuple[int, int]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return canonical joints (F', J, 2), centre trajectory (F', 2), yaw (F',) and the body height in pixels."""
    s0, s1 = segment
    img = _fill_gaps(pose.image[s0:s1])
    world = _fill_gaps(pose.world[s0:s1])
    torso = [pose.joint_names.index(n) for n in TORSO]

    scale = float(np.median(img[:, :, 1].max(axis=1) - img[:, :, 1].min(axis=1)))
    centre = img[:, torso].mean(axis=1)                         # (F', 2)
    ref = np.array([0.0, np.median(centre[:, 1])])
    joints = (img - np.stack([centre[:, 0], np.full(len(img), ref[1])], axis=1)[:, None]) / scale
    centre_traj = (centre - np.array([centre[0, 0], ref[1]])) / scale

    ls, rs = world[:, torso[0]], world[:, torso[1]]
    yaw = np.unwrap(np.arctan2(ls[:, 2] - rs[:, 2], ls[:, 0] - rs[:, 0]))
    return joints, centre_traj, yaw, scale


def resample(a: np.ndarray, frames: int) -> np.ndarray:
    """Linearly resample the first axis to ``frames`` evenly spaced points."""
    src = np.linspace(0, 1, len(a))
    dst = np.linspace(0, 1, frames)
    flat = a.reshape(len(a), -1)
    out = np.stack([np.interp(dst, src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape((frames,) + a.shape[1:])


def rasterise(joints: np.ndarray, connections: Sequence[Tuple[int, int]], size: int,
              window_x: Sequence[float], window_y: Sequence[float], thickness: float,
              raster_scale: int, torso: Optional[Sequence[int]] = None) -> np.ndarray:
    """Draw each frame's skeleton and average it into a ``size x size`` grid of cell coverage in [0, 1].

    Every skeleton edge is drawn with the same thickness (in body heights).
    The torso quadrilateral, if given, is filled, so the trunk has volume.
    """
    import cv2
    R = size * raster_scale
    (x0, x1), (y0, y1) = window_x, window_y
    px = lambda p: ((p[..., 0] - x0) / (x1 - x0) * R, (p[..., 1] - y0) / (y1 - y0) * R)
    width = max(1, int(round(thickness / (y1 - y0) * R)))
    out = np.zeros((len(joints), size, size))
    for t, frame in enumerate(joints):
        canvas = np.zeros((R, R), np.float32)
        xs, ys = px(frame)
        pts = np.stack([xs, ys], 1)
        if torso is not None:
            quad = pts[[torso[0], torso[1], torso[3], torso[2]]]
            cv2.fillPoly(canvas, [np.round(quad * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        for a, b in connections:
            pa, pb = np.round(pts[a] * 16).astype(int), np.round(pts[b] * 16).astype(int)
            cv2.line(canvas, tuple(pa), tuple(pb), 1.0, width, cv2.LINE_AA, shift=4)
        out[t] = cv2.resize(np.clip(canvas, 0, 1), (size, size), interpolation=cv2.INTER_AREA)
    return out


def trajectory_from_pose(pose: PoseSequence, cfg: Dict) -> Trajectory:
    seg = valid_segment(pose, cfg["min_visibility"], cfg["edge_margin"])
    joints, centre, yaw, scale = canonicalise(pose, seg)
    T = cfg["frames"]
    jT = resample(joints, T)
    torso = [pose.joint_names.index(n) for n in TORSO]
    fields = rasterise(jT, pose.connections, cfg["field_size"], cfg["window_x"], cfg["window_y"],
                       cfg["thickness"], cfg["raster_scale"], torso)
    return Trajectory(jT, resample(centre, T), resample(yaw, T), fields, joints, seg, scale,
                      np.array(pose.connections))


def preview(traj: Trajectory, path, title: str = "", lot_fields: Optional[np.ndarray] = None) -> None:
    """Save a PNG: one canonical stick figure per frame above its field (and an optional LoT field row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = 3 if lot_fields is not None else 2
    T = len(traj.fields)
    fig, axes = plt.subplots(rows, T, figsize=(T * 0.9, rows * 1.0 + 0.4))
    for t in range(T):
        ax = axes[0, t]
        for a, b in traj.connections:
            ax.plot(traj.joints[t, [a, b], 0], traj.joints[t, [a, b], 1], "k-", lw=0.8)
        ax.set_xlim(-0.65, 0.65); ax.set_ylim(0.7, -0.6); ax.set_aspect("equal"); ax.axis("off")
        axes[1, t].imshow(traj.fields[t], cmap="gray", vmin=0, vmax=1); axes[1, t].axis("off")
        if lot_fields is not None:
            axes[2, t].imshow(lot_fields[t], cmap="magma", vmin=0, vmax=1); axes[2, t].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
