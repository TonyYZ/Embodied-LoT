"""Transparent distances between two (T, ...) motion sequences. No learned weights."""

from __future__ import annotations

import numpy as np


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared difference over every frame and cell (both sequences are already time-normalised)."""
    return float(np.mean((a - b) ** 2))


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic time warping. Frame cost is the mean squared difference; the total is normalised by path length."""
    A, B = a.reshape(len(a), -1), b.reshape(len(b), -1)
    cost = ((A[:, None, :] - B[None, :, :]) ** 2).mean(axis=2)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    steps = np.zeros((n + 1, m + 1))
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min((acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1))
            acc[i, j] = cost[i - 1, j - 1] + prev[0]
            steps[i, j] = steps[prev[1], prev[2]] + 1
    return float(acc[n, m] / steps[n, m])


DISTANCES = {"euclid": euclid, "dtw": dtw}
