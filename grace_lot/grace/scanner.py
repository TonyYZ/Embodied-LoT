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

import os
from functools import lru_cache
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


@lru_cache(maxsize=32)
def _axis_responses(n: int, s: float) -> np.ndarray:
    """(n, 3) detector responses at the centres of n cells along one axis (read-only; shared)."""
    r = receptive_fields((np.arange(n) + 0.5) / n, s)
    r.flags.writeable = False
    return r


def scan_field(fields: np.ndarray, s: float, salience: float) -> np.ndarray:
    """(T, K, H, W) rendered channels -> (T, K, 6): salient cells are those with value >= salience.

    A cell's x detectors depend only on its column and its y detectors only on its row, so each max
    runs over the columns (rows) that hold any salient cell.
    """
    T, K, H, W = fields.shape
    mask = fields >= salience                                                           # (T, K, H, W)
    rx = np.where(mask.any(axis=2)[..., None], _axis_responses(W, s), 0.0).max(axis=2)  # (T, K, 3)
    ry = np.where(mask.any(axis=3)[..., None], _axis_responses(H, s), 0.0).max(axis=2)
    return np.concatenate([rx, ry], axis=2)


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


FEATURES = {"max6": 6, "orient8": 8}


def field_features(fields: np.ndarray, s: float, salience: float, kind: str = "max6") -> np.ndarray:
    """(..., K, H, W) fields -> (..., K, F) features of their salient cells (value >= salience).

    max6     the current scanner (``scan_field``): 3 detectors per axis, max over the salient cells
    orient8  the same detectors averaged over the salient cells, plus the elongation-weighted orientation of
             their principal axis, (e cos 2t, e sin 2t) with e = 1 - l2 / l1, which tells the two diagonals
             apart. An empty channel is all zeros.
    """
    if kind == "max6":
        lead = fields.shape[:-3]
        out = scan_field(fields.reshape((-1,) + fields.shape[-3:]), s, salience)
        return out.reshape(lead + out.shape[-2:])
    if kind != "orient8":
        raise ValueError(f"unknown features {kind!r}")
    H, W = fields.shape[-2:]
    m = (fields >= salience).astype(np.float64)                                  # (..., K, H, W)
    n = m.sum(axis=(-2, -1))
    safe = np.maximum(n, 1.0)
    x, y = (np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H
    cols, rows = m.sum(axis=-2), m.sum(axis=-1)                                  # (..., K, W), (..., K, H)
    fx = cols @ _axis_responses(W, s) / safe[..., None]                         # means over the salient cells
    fy = rows @ _axis_responses(H, s) / safe[..., None]
    mx, my = cols @ x / safe, rows @ y / safe
    cxx = cols @ (x * x) / safe - mx * mx
    cyy = rows @ (y * y) / safe - my * my
    cxy = np.einsum("...hw,h,w->...", m, y, x) / safe - mx * my
    half = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy ** 2)
    l1, l2 = (cxx + cyy) / 2 + half, (cxx + cyy) / 2 - half
    ok = (n >= 2) & (l1 > 1e-12) & (half > 1e-12)
    e = np.where(ok, 1 - l2 / np.where(ok, l1, 1.0), 0.0)
    c2 = np.where(ok, (cxx - cyy) / np.where(ok, 2 * half, 1.0), 0.0)            # cos 2t and sin 2t of the major axis
    s2 = np.where(ok, cxy / np.where(ok, half, 1.0), 0.0)
    return np.concatenate([fx, fy, (e * c2)[..., None], (e * s2)[..., None]], axis=-1) * (n > 0)[..., None]


def part_boards(traj: Trajectory, pre: dict, edge_samples: int) -> np.ndarray:
    """(T, P, H, W) binary boards: a cell is 1 if any of the part's skeleton points falls in it.

    The observation side of ``interface.observation_cells``: parts are quantised to the LoT's cells first, so
    observed and LoT features both come from ``scan_field``. Points beyond the window go to its border cells.
    """
    pts = part_points(traj, pre, edge_samples)
    n = pre["field_size"]
    T = len(traj.joints)
    out = np.zeros((T, len(PART_NAMES), n, n))
    t = np.arange(T)[:, None]
    for k, name in enumerate(PART_NAMES):
        cell = np.clip((pts[name] * n).astype(int), 0, n - 1)                          # (T, M, 2) as (col, row)
        out[t, k, cell[..., 1], cell[..., 0]] = 1.0
    return out


def scan_part_cells(traj: Trajectory, pre: dict, s: float, salience: float, edge_samples: int,
                    features: str = "max6") -> np.ndarray:
    """(T, P, F) features of the observed parts, read from their cell boards (``field_features``)."""
    return field_features(part_boards(traj, pre, edge_samples), s, salience, features)


def feature_background(cfg: dict) -> dict:
    """What "no claim" predicts: real part features and boards pooled over every training video and manner
    (features as ``interface.features``).

    Returns per-detector mean and variance of the features (``mu``, ``var``) and of their frame-to-frame
    changes (``dmu``, ``dvar``), and the real part boards (``boards``, (M, T, H, W)) that unclaimed cells
    are filled from. Training actors only; cached in the processed directory.
    """
    import hashlib
    import json
    from pathlib import Path
    from grace_lot.grace.preprocess import load_trajectory
    from grace_lot.inference.run import videos_for
    pre, itf = cfg["preprocess"], cfg["interface"]
    key = hashlib.sha1(json.dumps([pre, cfg.get("split", {}), itf], sort_keys=True, default=str).encode()).hexdigest()[:12]
    path = Path(cfg["data"]["processed_dir"]) / f"background_features_{key}.npz"
    if not path.exists():
        bym, _ = videos_for(cfg, "train")
        boards = [part_boards(load_trajectory(cfg, v), pre, itf["edge_samples"]) for m in sorted(bym) for v in bym[m]]
        feats = np.concatenate([field_features(b, itf["scan_softness"], itf["lot_salience"], itf.get("features", "max6"))
                                .transpose(1, 0, 2) for b in boards])
        d = np.diff(feats, axis=1)                                                     # feats: (M, T, 6)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")       # written whole, then moved: parallel workers
        with open(tmp, "wb") as f:                                   # never read half a file
            np.savez_compressed(f, mu=feats.mean(axis=(0, 1)), var=feats.var(axis=(0, 1)), dmu=d.mean(axis=(0, 1)),
                                dvar=d.var(axis=(0, 1)),
                                boards=np.concatenate([b.transpose(1, 0, 2, 3) for b in boards]).astype(bool))
        os.replace(tmp, path)
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) if k != "boards" else z[k] for k in z.files}
