"""Limb-motion descriptors: what a manner's movement is, apart from who performs it and when.

Defined a priori (29 Oct 2026), before looking at any scores, from the standard description of gait: each limb
oscillates once or twice per cycle, with some range, in step or out of step with the others. Computed alike from
part boards (a program's outputs or a video's observations) and from skeleton points (a reference without cells).

A video's 16 frames are two whole cycles (``preprocess.sampling = "cycle2"``), so harmonic k = 2 of the frames is
once per cycle and k = 4 twice per cycle. Per part, three signals over the frames: the centroid's x and y, and
the part's axis as an orientation vector (e cos 2t, e sin 2t; e = elongation, so a blob has no direction).

  amplitude  per part, signal (x, y, axis) and harmonic (once, twice per cycle): how much it moves at that rate
  coupling   per limb pair and signal (x, y), at once per cycle: 2 Re(z1 z2*) / (|z1|^2 + |z2|^2), +1 in step
             with equal swings, -1 in antiphase, near 0 if either barely moves

Amplitudes and couplings do not change when the frames are shifted circularly (phase), the body sits elsewhere
(the constant term is dropped), or the clip is resampled to the same two cycles (tempo). Sizes are in board units:
body heights in the board's window.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from grace_lot.grace.parts import PART_NAMES

HARMONICS = {"once": 2, "twice": 4}
SIGNALS = ("x", "y", "axis")
PAIRS = [("left_leg", "right_leg"), ("left_arm", "right_arm"), ("left_arm", "right_leg"), ("right_arm", "left_leg")]


def _signals(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights w (T, P, M) on points at (x, y) (M,) or (T, P, M) -> (T, P, 4): centroid x, y and axis (c, s)."""
    return _from_moments(*((w * b).sum(-1) for b in (1, x, y, x * x, y * y, x * y)))


def _from_moments(n, sx, sy, sxx, syy, sxy) -> np.ndarray:
    """Weighted sums of 1, x, y, x^2, y^2, xy -> (..., 4): centroid x, y and axis (c, s)."""
    n = np.maximum(n, 1e-9)
    mx, my = sx / n, sy / n
    cxx = sxx / n - mx ** 2
    cyy = syy / n - my ** 2
    cxy = sxy / n - mx * my
    half = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy ** 2)
    l1 = (cxx + cyy) / 2 + half
    ok = (l1 > 1e-12) & (half > 1e-12)
    e = np.where(ok, 2 * half / np.where(ok, l1, 1.0), 0.0)               # 1 - l2 / l1
    c = np.where(ok, (cxx - cyy) / np.where(ok, 2 * half, 1.0), 0.0)
    s = np.where(ok, cxy / np.where(ok, half, 1.0), 0.0)
    return np.stack([mx, my, e * c, e * s], axis=-1)


def board_signals(boards: np.ndarray, salience: float = 0.75) -> np.ndarray:
    """Part boards (T, P, H, W) -> (T, P, 4); a cell counts if its value >= salience (as the scanner's), or, with
    ``salience`` None, as much as its value (a board of expected presence)."""
    T, P, H, W = boards.shape
    w = (boards if salience is None else boards >= salience).reshape(T, P, H * W).astype(float)
    yy, xx = np.meshgrid((np.arange(H) + 0.5) / H, (np.arange(W) + 0.5) / W, indexing="ij")
    x, y = xx.ravel(), yy.ravel()
    m = w @ np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=1)   # (T, P, 6): one product for every sum
    return _from_moments(*np.moveaxis(m, -1, 0))


def point_signals(points: Dict[str, np.ndarray]) -> np.ndarray:
    """Skeleton points per part, (T, M, 2) in window coordinates (``scanner.part_points``) -> (T, P, 4)."""
    one = lambda q: _signals(np.ones(q.shape[:2])[:, None], q[:, None, :, 0], q[:, None, :, 1])[:, 0]
    return np.stack([one(points[p]) for p in PART_NAMES], axis=1)


def descriptors(sig: np.ndarray) -> np.ndarray:
    """(T, P, 4) signals -> the descriptor vector (``names()`` gives its order)."""
    Z = np.fft.rfft(sig, axis=0) / len(sig)                                # (K, P, 4)
    out = []
    for k in HARMONICS.values():
        z = Z[k]
        out += [np.abs(z[:, 0]), np.abs(z[:, 1]), np.sqrt(np.abs(z[:, 2]) ** 2 + np.abs(z[:, 3]) ** 2)]
    z = Z[HARMONICS["once"]]
    i = {p: j for j, p in enumerate(PART_NAMES)}
    for a, b in PAIRS:
        for s in (0, 1):
            z1, z2 = z[i[a], s], z[i[b], s]
            out.append(np.array([2 * np.real(z1 * np.conj(z2)) / max(abs(z1) ** 2 + abs(z2) ** 2, 1e-12)]))
    return np.concatenate(out)


def names() -> List[str]:
    out = [f"{p} {s} {h}" for h in HARMONICS for s in SIGNALS for p in PART_NAMES]
    return out + [f"{a}~{b} {s}" for a, b in PAIRS for s in ("x", "y")]
