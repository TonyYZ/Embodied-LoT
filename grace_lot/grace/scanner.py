"""A minimal soft scanner: three overlapping detectors per axis, applied to sets of salient points.

Horizontal: left / center / right.  Vertical: upper / center / lower.
Each detector is a continuous receptive field over normalised position
u in [0, 1] (window coordinates), built from sigmoids with boundaries at
1/3 and 2/3 and softness s:

    w_low(u)    = sig((1/3 - u) / s)
    w_center(u) = sig((u - 1/3) / s) * sig((2/3 - u) / s)
    w_high(u)   = sig((u - 2/3) / s)

A channel's activation for detector i is the maximum response over its
salient points:  a_i = max_{p in S} w_i(p).  An empty channel gives 0.

The bands are detectors, not templates: one salient point in a region is
enough. A thin skeleton line and a broad LoT output therefore activate the
same detectors when they occupy the same regions.

Salient points:
* observed body parts: skeleton points, all with equal salience. These are
  the part's joints plus ``edge_samples`` evenly spaced points along each
  of its skeleton edges.
* LoT outputs: cell centres whose rendered value is at least
  ``lot_salience``. ``render_output`` paints undetermined (yin) bands at 0.5
  times their gate, so the threshold separates drawn material from
  "unknown".

Both sides go through the same detector functions; only the extraction of
salient points depends on the modality.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from grace_lot.grace.parts import PART_NAMES, part_connections
from grace_lot.grace.trajectories import Trajectory

DETECTORS = ["left", "h_center", "right", "upper", "v_center", "lower"]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def receptive_fields(u: np.ndarray, s: float) -> np.ndarray:
    """(..., ) normalised positions -> (..., 3) responses of the low / center / high detectors."""
    lo = _sig((1 / 3 - u) / s)
    mid = _sig((u - 1 / 3) / s) * _sig((2 / 3 - u) / s)
    hi = _sig((u - 2 / 3) / s)
    return np.stack([lo, mid, hi], axis=-1)


def scan_points(points: np.ndarray, s: float) -> np.ndarray:
    """(N, 2) points as (x, y) in [0, 1] window coordinates, y down -> (6,) activations.

    Order: left, h_center, right (from x), then upper, v_center, lower (from y).
    """
    if len(points) == 0:
        return np.zeros(6)
    rx = receptive_fields(points[:, 0], s).max(axis=0)
    ry = receptive_fields(points[:, 1], s).max(axis=0)       # low y = upper
    return np.concatenate([rx, ry])


def scan_field(fields: np.ndarray, s: float, salience: float) -> np.ndarray:
    """(T, K, H, W) rendered channels -> (T, K, 6): salient cells are those with value >= salience."""
    T, K, H, W = fields.shape
    ys, xs = np.mgrid[0:H, 0:W]
    centres = np.stack([(xs.ravel() + 0.5) / W, (ys.ravel() + 0.5) / H], axis=1)
    resp = np.concatenate([receptive_fields(centres[:, 0], s), receptive_fields(centres[:, 1], s)], axis=1)  # (HW, 6)
    mask = fields.reshape(T, K, H * W) >= salience                                                          # (T, K, HW)
    out = np.where(mask[..., None], resp[None, None], 0.0).max(axis=2)
    return out


def part_points(traj: Trajectory, pre: dict, edge_samples: int) -> Dict[str, np.ndarray]:
    """Per part: (T, N, 2) skeleton points in [0, 1] window coordinates (joints + samples along edges)."""
    (x0, x1), (y0, y1) = pre["window_x"], pre["window_y"]
    j = traj.joints                                                # (T, J, 2) canonical
    norm = np.stack([(j[..., 0] - x0) / (x1 - x0), (j[..., 1] - y0) / (y1 - y0)], axis=-1)
    alphas = np.linspace(0, 1, edge_samples + 2)
    out = {}
    for part, edges in part_connections(traj.connections).items():
        pts = [norm[:, a] * (1 - t) + norm[:, b] * t for a, b in edges for t in alphas]
        out[part] = np.stack(pts, axis=1)
    return out


def scan_parts(traj: Trajectory, pre: dict, s: float, edge_samples: int) -> np.ndarray:
    """(T, P, 6) detector activations of every observed body part."""
    pts = part_points(traj, pre, edge_samples)
    T = len(traj.joints)
    return np.stack([[scan_points(pts[p][t], s) for p in PART_NAMES] for t in range(T)])
