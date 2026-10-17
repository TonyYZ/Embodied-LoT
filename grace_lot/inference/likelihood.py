"""Theory-neutral likelihood: ``log p(v | r) = -beta * D(human_trajectory(v), lot_trajectory(r))``.

``observations`` is a list of field sequences. One element is a single
exemplar video. Several elements give the manner-level likelihood
``sum_a log p(v_a | r)`` over actors (Stage 5). D gives every cell and frame
equal weight.

With ``temporal_weight`` w > 0, D also compares how things change from frame to
frame: D = mean (x - y)^2 + w * mean (dx - dy)^2, with dx(t) = x(t) - x(t-1). A
program then earns likelihood for moving where and when the video moves, and a
still video costs nothing to match still. w = 0 is the frame-only likelihood.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.distance import euclid
from grace_lot.lot_adapter.program import P


def _pair_costs(y: np.ndarray, obs: np.ndarray, obs_energy: np.ndarray):
    """C[n, k, p] = mean over (t, features) of (y_k - obs_np)^2, and y's own energy per channel.

    y is (T, K, F) and obs is (N, T, P, F); obs_energy is (N, P), the mean square of obs.
    """
    T, F = y.shape[0], y.shape[2]
    ey = (y ** 2).mean(axis=(0, 2))                                       # (K,)
    cross = np.einsum("tkc,ntpc->nkp", y, obs) / (T * F)
    return ey[None, :, None] + obs_energy[:, None, :] - 2 * cross, ey


class TrajectoryLikelihood:
    def __init__(self, observations: Sequence[np.ndarray], renderer: LotRenderer,
                 distance: Callable[[np.ndarray, np.ndarray], float], beta: float,
                 temporal_weight: float = 0.0) -> None:
        self.observations = list(observations)
        self.renderer = renderer
        self.distance = distance
        self.beta = beta
        self.w = float(temporal_weight)
        if self.w and distance is not euclid:
            raise ValueError("temporal_weight needs the euclid distance")
        self._stack = np.stack(self.observations).astype(np.float32) if distance is euclid else None
        self._dstack = np.diff(self._stack, axis=1) if self.w else None

    def __call__(self, p) -> float:
        predicted = self.renderer(p)
        if self._stack is not None:      # euclid summed over observations, in one vectorised step
            per_obs = ((self._stack - predicted) ** 2).reshape(len(self._stack), -1).mean(axis=1)
            if self.w:                   # frame-to-frame changes
                d = ((self._dstack - np.diff(predicted, axis=0)) ** 2).reshape(len(self._stack), -1).mean(axis=1)
                per_obs = per_obs + self.w * d
            return -self.beta * float(per_obs.sum())
        return -self.beta * sum(self.distance(obs, predicted) for obs in self.observations)


class MatchedPartsLikelihood:
    """Multi-output likelihood: anonymous LoT outputs vs skeleton part channels, matched once per video.

    For each observed video (a (T, P, H, W) part-field array) and a program's
    rendering (T, K, H, W):

        C[k, p] = mean over t, cells of (output_k(t) - part_p(t))^2

    Channels are padded with empty fields to M = max(K, P), so that an
    unmatched part is scored against nothing and an extra output against
    nothing. One permutation per video, chosen over the whole sequence by
    optimal assignment (Hungarian), gives

        D = sum of matched C / P,    log p(v | r) = -beta * sum_videos D

    The mapping is never re-chosen frame by frame. Re-choosing it would let
    crossing parts be relabelled as bouncing ones and erase alternation.

    With ``temporal_weight`` w, each cost also compares the channels' frame-to-frame
    changes (``C += w * C_changes``), so outputs are matched to parts by how they move too.
    """

    def __init__(self, observations: Sequence[np.ndarray], renderer: LotRenderer, beta: float,
                 temporal_weight: float = 0.0) -> None:
        self.obs = np.stack(observations).astype(np.float32)          # (N, T, P, H, W)
        N, T, P = self.obs.shape[:3]
        self.P = P
        self._set_features(self.obs.reshape(N, T, P, -1), temporal_weight)
        self.renderer = renderer
        self.beta = beta

    def _set_features(self, flat: np.ndarray, temporal_weight: float) -> None:
        """Observed features (N, T, P, F), their energies, and (with w > 0) their frame-to-frame changes."""
        self._flat = flat
        self._energy = (flat ** 2).mean(axis=(1, 3))                   # (N, P)
        self.w = float(temporal_weight)
        if self.w:
            self._dflat = np.diff(flat, axis=1)
            self._denergy = (self._dflat ** 2).mean(axis=(1, 3))

    def _padded(self, y: np.ndarray) -> np.ndarray:
        """(N, M, M) padded cost matrices for output features y (T, K, F)."""
        K = y.shape[1]
        C, ey = _pair_costs(y, self._flat, self._energy)               # (N, K, P), (K,)
        energy = self._energy
        if self.w:                                                     # frame-to-frame changes
            Cd, eyd = _pair_costs(np.diff(y, axis=0), self._dflat, self._denergy)
            C, ey, energy = C + self.w * Cd, ey + self.w * eyd, energy + self.w * self._denergy
        M = max(K, self.P)
        out = np.zeros((len(self.obs), M, M), dtype=np.float64)
        out[:, :K, :self.P] = C
        out[:, K:, :self.P] = energy[:, None, :]                       # missing output -> part vs empty
        out[:, :K, self.P:] = ey[None, :, None]                        # extra output -> output vs empty
        return out

    def costs(self, p) -> np.ndarray:
        """(N, M, M) padded cost matrices: rows = outputs (+ empty), columns = parts (+ empty)."""
        y = self.renderer(p)
        if y.ndim == 3:
            y = y[:, None]
        T, K = y.shape[:2]
        return self._padded(y.reshape(T, K, -1).astype(np.float32))

    def assignments(self, p):
        """Per video: (row indices, column indices, distance D)."""
        from scipy.optimize import linear_sum_assignment
        res = []
        for C in self.costs(p):
            r, c = linear_sum_assignment(C)
            res.append((r, c, float(C[r, c].sum() / self.P)))
        return res

    def __call__(self, p) -> float:
        return -self.beta * sum(d for _, _, d in self.assignments(p))


class MatchedScanLikelihood(MatchedPartsLikelihood):
    """The matched-parts likelihood, computed on scanner activations instead of pixels.

    Observations are (T, P, 6) detector activations of the body parts
    (``grace.scanner.scan_parts``). Each LoT output channel is rendered and
    scanned the same way (``grace.scanner.scan_field``). Costs are mean squared
    activation differences over frames and the 6 detectors. Matching is
    unchanged: one optimal assignment per video for the whole sequence,
    padded with empty channels (all-zero activations).
    """

    def __init__(self, observations, renderer, beta: float, softness: float, salience: float,
                 temporal_weight: float = 0.0) -> None:
        obs = np.stack(observations).astype(np.float32)                 # (N, T, P, 6)
        self.obs = obs
        N, T, P = obs.shape[:3]
        self.P = P
        self._set_features(obs, temporal_weight)
        self.renderer = renderer
        self.beta = beta
        self.softness, self.salience = softness, salience
        self._scan_cache = {}

    def scan(self, p) -> np.ndarray:
        """(T, K, 6) activations of a program's output channels (memoised)."""
        if p not in self._scan_cache:
            from grace_lot.grace.scanner import scan_field
            y = self.renderer(p)
            y = y if y.ndim == 4 else y[:, None]
            if len(self._scan_cache) > 20000:
                self._scan_cache.clear()
            self._scan_cache[p] = scan_field(y, self.softness, self.salience).astype(np.float32)
        return self._scan_cache[p]

    def costs(self, p) -> np.ndarray:
        return self._padded(self.scan(p))                                # scan: (T, K, 6)
