"""Which source frames become a video's 16 frames (``preprocess.sampling``).

``segment`` (default, every earlier run): the whole valid segment, linearly resampled to ``frames``.
``cycle1`` / ``cycle2``: one or two motion cycles, centred in the segment, resampled to ``frames``, so every
actor's frames cover the same number of cycles whatever their tempo or segment length. The cycle is found
generically, the same way for every manner (docs/dynamic_reachability_audit.md, section 8):

    the autocorrelation of the detrended full-rate joint signal; the period is the first peak, at a lag of at
    least ``min_lag`` with r >= ``threshold``, that reaches 80% of the highest peak.

When no period is found, or the cycles do not fit in the segment, the video falls back to ``segment``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

SAMPLINGS = ("segment", "cycle1", "cycle2")


def _detrended(joints: np.ndarray) -> np.ndarray:
    x = joints.reshape(len(joints), -1).astype(float)
    A = np.c_[np.ones(len(x)), np.arange(len(x))]
    return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]


def detect_period(joints: np.ndarray, min_lag: int = 6, threshold: float = 0.3) -> Optional[int]:
    """The motion period of full-rate joints (F, J, 2) in source frames, or None."""
    x = _detrended(joints)
    L = len(x)
    if L < 2 * min_lag + 2:
        return None
    r = np.array([float((x[:-l] * x[l:]).sum() / max(np.sqrt((x[:-l] ** 2).sum() * (x[l:] ** 2).sum()), 1e-12))
                  for l in range(1, L // 2 + 1)])
    peaks = [l for l in range(min_lag, L // 2) if r[l - 1] >= r[l - 2] and r[l - 1] >= r[l] and r[l - 1] >= threshold]
    if not peaks:
        return None
    top = max(peaks, key=lambda l: r[l - 1])
    return next(l for l in peaks if r[l - 1] >= 0.8 * r[top - 1])


def sample_window(joints: np.ndarray, sampling: str) -> Tuple[int, int]:
    """The source frames [a, b) that become the video's frames."""
    if sampling not in SAMPLINGS:
        raise ValueError(f"unknown sampling {sampling!r}")
    L = len(joints)
    if sampling == "segment":
        return 0, L
    period = detect_period(joints)
    span = None if period is None else int(sampling[-1]) * period + 1
    if span is None or span > L:
        return 0, L                                          # no period, or too few cycles: the whole segment
    a = (L - span) // 2
    return a, a + span
