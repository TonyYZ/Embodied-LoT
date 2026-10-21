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


def cell_probability(v: np.ndarray, background) -> np.ndarray:
    """p(figure) per cell from a rendered value v, reading 0.5 as "no claim".

    The claim strength is |2v - 1|: with that probability the cell is claimed (figure if v > 0.5, empty if
    v < 0.5); otherwise it is unclaimed and follows the background base rate. So v = 1 gives 1, v = 0 gives
    0, v = 0.5 gives the background. With ``background=None``, v itself is p(figure).
    """
    if background is None:
        return v
    b = np.asarray(background, dtype=np.float64)
    return np.where(v >= 0.5, (2 * v - 1) + (2 - 2 * v) * b, 2 * v * b)


class ProbabilisticScanLikelihood:
    """The matched scan likelihood under the probabilistic reading (docs/probabilistic_semantics.md).

    A program and a realization define a distribution over binary boards. A cell value v is p(yang); the
    unknown-motion units (universal* / neutral*) take discrete states with prior weights
    (``adapter.motion_combinations``). For each state combination, ``draws`` binary boards are sampled with
    one uniform per cell held across frames (a cell flips only where its probability changes; fixed seeds,
    so a program's score is fixed). Their detector activations give a mean and variance per frame, channel
    and detector, and likewise for frame-to-frame changes.

    Matching is as in ``MatchedScanLikelihood``: one optimal assignment per video. Each cost uses the
    prediction's variance on top of the baseline noise implied by beta (s0 = P T F / (2 beta)):

        C[k, p] = mean_tf[(mu - o)^2 s0 / (s0 + var) + s0 log(1 + var / s0)]  (+ w * the same for changes)

    so a certain program (every cell 0 or 1, no unknown motion) scores as ``MatchedScanLikelihood``. Per
    video, combinations mix by their prior weights: log sum_c w_c exp(log p(v | c)).

    ``background`` (an (H, W) base rate or a number) makes 0.5 mean "no claim": unclaimed cells are drawn
    from the background, the same for every program, instead of as coin flips (``cell_probability``).
    """

    def __init__(self, observations, frames: int, size: int, beta: float, softness: float, salience: float,
                 temporal_weight: float = 0.0, draws: int = 8, cache_size: int = 5000, background=None) -> None:
        from collections import OrderedDict
        self.O = np.stack(observations).astype(np.float64)              # (N, T, P, 6)
        N, T, P, F = self.O.shape
        self.P, self.beta, self.w = P, beta, float(temporal_weight)
        self.s0 = P * T * F / (2 * beta)
        self.s0d = P * (T - 1) * F / (2 * beta * self.w) if self.w else None
        self.dO = np.diff(self.O, axis=1)
        self.e = (self.O ** 2).mean(axis=(1, 3))
        self.de = (self.dO ** 2).mean(axis=(1, 3))
        self.frames, self.size, self.softness, self.salience, self.draws = frames, size, softness, salience, draws
        self.background = background
        self._cache, self._cache_size = OrderedDict(), cache_size

    def moments(self, p) -> list:
        """Per state combination: its weight and the mean / variance of detector activations and changes."""
        from grace_lot.grace.scanner import scan_field
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        out = []
        for states, weight in motion_combinations(p, self.size):
            v = rollout(p, self.frames, self.size, states)
            v = v if v.ndim == 4 else v[:, None]
            certain = bool(np.all((v == 0) | (v == 1)))
            p_fig = cell_probability(v, self.background)
            feats = []
            for d in range(1 if certain else self.draws):
                u = np.random.Generator(np.random.PCG64(10_000 + d)).random(v.shape[1:])
                feats.append(scan_field((u[None] < p_fig).astype(np.float64), self.softness, self.salience))
            Fs = np.stack(feats)
            dF = np.diff(Fs, axis=1)
            out.append(dict(weight=weight, mu=Fs.mean(0), var=Fs.var(0), dmu=dF.mean(0), dvar=dF.var(0)))
        return out

    @staticmethod
    def _cost(mu, var, O, s0):
        a = s0 / (s0 + var)
        b = s0 * np.log1p(var / s0)
        T, F = mu.shape[0], mu.shape[2]
        own = (a * mu ** 2 + b).mean(axis=(0, 2))                        # (K,): against an empty part
        cross = np.einsum("tkf,tkf,ntpf->nkp", a, mu, O) / (T * F)
        oo = np.einsum("tkf,ntpf->nkp", a, O ** 2) / (T * F)
        return own[None, :, None] - 2 * cross + oo, own

    def per_video(self, m) -> np.ndarray:
        """(N,) log-likelihoods for one state combination."""
        from scipy.optimize import linear_sum_assignment
        C, own = self._cost(m["mu"], m["var"], self.O, self.s0)
        e = self.e
        if self.w:
            Cd, ownd = self._cost(m["dmu"], m["dvar"], self.dO, self.s0d)
            C, own, e = C + self.w * Cd, own + self.w * ownd, e + self.w * self.de
        N, K, P = C.shape
        M = max(K, P)
        out = np.empty(N)
        for n in range(N):
            pad = np.zeros((M, M))
            pad[:K, :P] = C[n]
            pad[K:, :P] = e[n][None, :]                                  # a part with no output
            pad[:K, P:] = own[:, None]                                   # an output with no part
            r, c = linear_sum_assignment(pad)
            out[n] = -self.beta * pad[r, c].sum() / P
        return out

    def per_video_scores(self, p) -> np.ndarray:
        from scipy.special import logsumexp
        ms = self.moments(p)
        L = np.stack([self.per_video(m) for m in ms])
        return logsumexp(L + np.log([m["weight"] for m in ms])[:, None], axis=0)

    def __call__(self, p) -> float:
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        value = float(self.per_video_scores(p).sum())
        self._cache[p] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value


MEMO_SIZE = 10_000          # entries per channel / row memo (a few KB each: ~100 MB per worker), then cleared


def _digest(*arrays) -> bytes:
    """A 16-byte key for memoising by array values (raw bytes as keys cost ~36 KB each: GBs per worker)."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.digest()


class GaussianScanLikelihood:
    """A normalised Gaussian likelihood on detector activations (docs/probabilistic_semantics.md, 10-11).

    Observations are the parts' activations read from their cell boards (``interface.observation_cells``), so
    both sides come from ``scan_field``. A program and a state combination of its unknown-motion units
    define a distribution over boards: a cell is claimed with probability |2v - 1| (figure if v > 0.5), and
    unclaimed cells are filled from a real part board of the training videos (``background["boards"]``;
    ``draws`` boards, fixed seeds, one uniform per cell held across frames). The resulting activations give a
    mean and variance per frame, channel and detector, and likewise for frame-to-frame changes. All channels
    share each draw's fill and uniforms: channels that claim the same get the same moments, so the matching
    cannot pick among Monte Carlo variants of "no claim".

    Each output predicts a part with variance = its own variance + the observation noise s2. The noise per
    detector is the across-video variance of the observations given (one manner's actors), floored at
    ``noise_floor`` times the background variance (and 1e-6); pass ``noise`` to fix it instead. A part with
    no output is predicted by the background (pooled real part features). Per video, one optimal matching over the
    whole sequence minimises the summed negative log density

        NLL[k, p] = 1/2 sum_tf [(o - mu)^2 / (s2 + var) + log 2 pi (s2 + var)]   (+ w * the same for changes)

    and log p(v | r) = -temperature * NLL; temperature defaults to 1 / (T F), the mean density per frame and
    detector. Combinations mix by their prior weights: log sum_c w_c exp(log p(v | c)).
    """

    def __init__(self, observations, frames: int, size: int, softness: float, salience: float, background: dict,
                 temperature: float = None, temporal_weight: float = 0.0, draws: int = 256, noise_floor: float = 0.1,
                 noise: dict = None, cache_size: int = 5000, features: str = "max6", phases: int = 1) -> None:
        from collections import OrderedDict
        self.features = features                                          # interface.features (scanner.field_features)
        O = np.stack(observations).astype(np.float64)                   # (N, T, P, F)
        N, T, P, F = O.shape
        self.N, self.phases = N, int(phases)
        # Latent phase: each video is scored at every circular shift of its frames by 0 .. phases - 1 and the
        # densities are averaged (a uniform prior over the phase). Valid when a video's frames are whole cycles,
        # as with preprocess.sampling = "cycle2" (two cycles of 16 frames: phases = 8 covers one cycle). The
        # shifted copies are scored as videos of their own, (video, phase) in video-major order.
        self.O = np.concatenate([np.roll(O, -s, axis=1)[:, None] for s in range(self.phases)], axis=1
                                ).reshape(N * self.phases, T, P, F) if self.phases > 1 else O
        self.dO = np.diff(self.O, axis=1)
        self.P, self.w = P, float(temporal_weight)
        self.temperature = 1.0 / (T * F) if temperature is None else float(temperature)
        self.frames, self.size, self.softness, self.salience, self.draws = frames, size, softness, salience, draws
        self.bg = background
        if noise is None:                                               # from the videos as given, not shifted
            dO = np.diff(O, axis=1)
            noise = dict(s2=O.var(axis=0).mean(axis=(0, 1)), s2d=dO.var(axis=0).mean(axis=(0, 1)))
        self.s2 = np.maximum(noise["s2"], np.maximum(noise_floor * background["var"], 1e-6))     # 1e-6: a numeric guard
        self.s2d = np.maximum(noise["s2d"], np.maximum(noise_floor * background["dvar"], 1e-6))
        self._cache, self._cache_size = OrderedDict(), cache_size

    def moments(self, p) -> list:
        """Per state combination: its weight and the mean / variance of detector activations and changes."""
        from grace_lot.grace.scanner import field_features
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        out = []
        for states, weight in motion_combinations(p, self.size):
            v = rollout(p, self.frames, self.size, states)
            v = v if v.ndim == 4 else v[:, None]
            per = [self._channel_moments(v[:, k]) for k in range(v.shape[1])]
            out.append(dict(weight=weight, **{key: np.stack([c[key] for c in per], axis=1)
                                              for key in ("mu", "var", "dmu", "dvar")}))
        return out

    def _channel_moments(self, c: np.ndarray) -> dict:
        """Mean / variance of one channel's features (T, H, W) over the draws, and of their changes.

        Channels are independent given the shared draws, so each is computed on its own and memoised by its
        values: a proposal usually leaves most channels as they were, and these are not recomputed."""
        from grace_lot.grace.scanner import field_features
        key = _digest(c)
        cache = self.__dict__.setdefault("_channel_cache", {})
        hit = cache.get(key)
        if hit is not None:
            return hit
        T, H, W = c.shape
        if np.all((c == 0) | (c == 1)):
            Fs = field_features(c[:, None], self.softness, self.salience, self.features)[None, :, 0]   # (1, T, F)
        else:
            fills, uniforms = self._draws(T, H, W)                                             # (T, D, H, W), (D, H, W)
            claim, fig = np.abs(2 * c - 1), c > 0.5
            boards = np.where(uniforms[None] < claim[:, None], fig[:, None], fills)          # (T, D, H, W)
            Fs = field_features(boards, self.softness, self.salience, self.features).transpose(1, 0, 2)   # (D, T, F)
        dF = np.diff(Fs, axis=1)
        hit = dict(mu=Fs.mean(0), var=Fs.var(0), dmu=dF.mean(0), dvar=dF.var(0))
        if len(cache) > MEMO_SIZE:
            cache.clear()
        cache[key] = hit
        return hit

    def _draws(self, T: int, H: int, W: int):
        """The draws' fills and uniforms: they depend only on the draw index (fixed seeds), so they are made once.

        Draw d's generator gives the real board it fills from, then one uniform per cell, exactly as drawing
        them per program did (the same numbers in the same order)."""
        key = (T, H, W)
        cached = getattr(self, "_draw_cache", None)
        if cached is None or cached[0] != key:
            real = self.bg["boards"]
            idx, uni = np.empty(self.draws, dtype=int), np.empty((self.draws, H, W))
            for d in range(self.draws):
                rng = np.random.Generator(np.random.PCG64(20_000 + d))
                idx[d] = rng.integers(len(real))
                uni[d] = rng.random((H, W))
            self._draw_cache = (key, real[idx].transpose(1, 0, 2, 3), uni)                    # fills: (T, D, H, W)
        return self._draw_cache[1], self._draw_cache[2]

    @staticmethod
    def _nll(O, mu, var, s2):
        """(N, K, P) summed negative log densities of observations O (N, T, P, F) under channels (T, K, F)."""
        v = (var + s2).transpose(1, 0, 2)[None, :, None]                 # (1, K, 1, T, F)
        m = mu.transpose(1, 0, 2)[None, :, None]
        o = O.transpose(0, 2, 1, 3)[:, None]                             # (N, 1, P, T, F)
        return 0.5 * (((o - m) ** 2) / v + np.log(2 * np.pi * v)).sum(axis=(3, 4))

    def costs(self, m) -> np.ndarray:
        """(N, K + B, P) NLL of each output (then B = max(0, P - K) background rows) against each part."""
        N, T, P, F = self.O.shape
        K = m["mu"].shape[1]
        b = max(0, P - K)
        rows = [self._row(*(m[key][:, k] for key in ("mu", "var", "dmu", "dvar"))) for k in range(K)]
        if b:
            rows += [self._row(*(np.broadcast_to(self.bg[key], (len(m[key]), 1, F))[:, 0]
                                 for key in ("mu", "var", "dmu", "dvar")))] * b
        return np.stack(rows, axis=1)

    def _row(self, mu, var, dmu, dvar) -> np.ndarray:
        """(N, P) cost of one output channel (moments over T, F) against each part, memoised by the moments.

        Computed from contiguous copies, so a row depends only on the values (numpy's summation order follows
        memory layout): within about 1e-16 relative of computing all channels together, and the canonical value.
        """
        key = _digest(mu, var, dmu, dvar) if self.w else _digest(mu, var)
        cache = self.__dict__.setdefault("_row_cache", {})
        row = cache.get(key)
        if row is None:
            mu, var, dmu, dvar = (np.ascontiguousarray(a) for a in (mu, var, dmu, dvar))
            row = self._nll(self.O, mu[:, None], var[:, None], self.s2)[:, 0]
            if self.w:
                row = row + self.w * self._nll(self.dO, dmu[:, None], dvar[:, None], self.s2d)[:, 0]
            if len(cache) > MEMO_SIZE:
                cache.clear()
            cache[key] = row
        return row

    def _matched(self, m):
        """Per video: (rows, columns, matched NLL); outputs beyond the parts stay unmatched."""
        from scipy.optimize import linear_sum_assignment
        res = []
        for C in self.costs(m):
            r, c = linear_sum_assignment(C)
            res.append((r, c, float(C[r, c].sum())))
        return res

    def _scores(self, p):
        """(combinations, N) log densities (phase marginalised), and the matchings per (video, phase)."""
        from scipy.special import logsumexp
        ms = self.moments(p)
        matched = [self._matched(m) for m in ms]
        L = -self.temperature * np.array([[nll for _, _, nll in mt] for mt in matched])      # (combinations, N x phases)
        if self.phases > 1:
            L = logsumexp(L.reshape(len(ms), self.N, self.phases), axis=2) - np.log(self.phases)
        return L + np.log([m["weight"] for m in ms])[:, None], matched

    def per_video_scores(self, p) -> np.ndarray:
        from scipy.special import logsumexp
        return logsumexp(self._scores(p)[0], axis=0)

    def assignments(self, p):
        """Per video, under its most probable state combination: (rows, columns, matched NLL x temperature / P).

        Rows >= the program's output count are background rows (parts no output explains).
        """
        L, matched = self._scores(p)
        best = L.argmax(axis=0)
        S = self.phases                                                  # its best phase, under that combination
        pick = lambda b, n: min(matched[b][n * S:(n + 1) * S], key=lambda x: x[2])
        return [(r, c, nll * self.temperature / self.P) for (r, c, nll) in (pick(b, n) for n, b in enumerate(best))]

    def __call__(self, p) -> float:
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        value = float(self.per_video_scores(p).sum())
        self._cache[p] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value
