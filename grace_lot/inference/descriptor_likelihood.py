"""A likelihood on limb-motion descriptors (``grace.descriptors``): a program is scored on how its outputs move,
not on where each cell is in each frame (``inference.likelihood = "descriptors"``).

A program and a state combination of its unknown-motion units define a distribution over part boards exactly as
for ``GaussianScanLikelihood``: a cell is claimed with probability |2v - 1|, unclaimed cells are filled from a
real part board (``draws`` boards, fixed seeds; each output its own, so separate outputs are drawn independently and
share no spurious coupling). Each draw's boards give each output's descriptors: amplitudes of
its centroid x, y and axis at once and twice per cycle, and the once-per-cycle couplings of limb pairs. Their mean
and variance over the draws predict a video's descriptors, with variance = draw variance + observation noise s2
(the across-video variance of the manner's actors, floored at ``noise_floor`` x the background's).

Per video, outputs are matched to parts by their amplitude costs (one optimal matching); parts no output explains
are predicted by the background (real parts pooled over every training video and manner). Couplings follow the
matching: a pair whose two parts are both matched is predicted from those outputs, drawn together; otherwise by
the background. log p(v | r) = -temperature x NLL; combinations mix by their prior weights.

Nothing in the descriptors depends on the phase, the position or the tempo of the two cycles, so there is no
latent phase or warp here: the program is the schema, the actors' timing and placement are not scored at all.

With ``size_range`` r > 1, a program states how far its parts move only up to one size factor per video (as the
global scale of a drawn character in Lake et al. 2015): its outputs' x and y amplitudes (mean c, spread c^2) are
compared at SIZE_STEPS factors c, log-evenly from 1/r to r (a log-uniform prior, fixed a priori), and the video's
likelihood is their average. Axis amplitudes (angles), couplings (size-free) and the background are not scaled.
The noise s2 of the lengths is then measured on each actor's amplitudes divided by the actor's overall size (the mean
of its length amplitudes over the manner's mean), so the factor alone carries how big an actor moves.

With ``features = "full"`` (``inference.descriptors``; default "motion", the above) each part also has amplitudes at
every rhythm from once per clip to twice per cycle (FULL_HARMONICS: a single step across the clip is no longer
free), and its posture: its average x, y (the board follows the body), size (share of the board) and axis. These are
averages over the clip, so timing still does not count.

With ``open_variance`` (``inference.open_variance``, expected boards only) an output is no surer than what it claims: its
descriptors' predicted variance gains u^2 x the background's, u the share of its board that comes from unclaimed cells. An
output claiming nothing is then predicted as widely as a part no output explains; one claiming everything, as before.

With ``baseline = "blobs"`` (``inference.baseline``, expected boards only) nothing here knows what a body is. An output is
described by its own claims alone, no fill (where it draws nothing, its centre and axis hold their nearest drawn value); what
no output claims is predicted by a generic background, the descriptors of random blobs: 1..8 x 1..8 cells, anywhere, swung
0..3 cells at a random rhythm, no data. An output's prediction blends towards that background by u, 1 minus its figure's
mass-weighted claim strength, so claiming nothing is exactly the background: a program earns only evidence over it.

With ``warmup`` w > 0 (``inference.warmup``) a program runs w ticks before the clip starts, so what is scored is how it
keeps behaving, not its start-up (a one-tick change at the start would otherwise read as a swing at every rhythm).

With ``open_cells = "expected"`` (``inference.open_cells``; default "fill", the draws above) there are no draws: an
output is read as its expected board, each cell claim x (its claims' share of figure) + (1 - claim) x how often a real
part covers that cell, averaged over every real part board and frame. Cells the program makes no claim about then
carry a body's average presence but no motion, so all motion comes from what the program draws (graded where its
claims are partial); the descriptors weigh each cell by its expected presence, and the prediction has no spread of
its own (only the actors' noise s2). Decided 30 Oct 2026, before scoring: the fills had let a program borrow real
footage's motion, and tune it through the cell-by-cell mix, without drawing any.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from grace_lot.grace import descriptors as D
from grace_lot.grace.parts import PART_NAMES
from grace_lot.inference.likelihood import MEMO_SIZE, _digest

A = len(D.HARMONICS) * len(D.SIGNALS)                   # amplitudes per part
PAIRS = [(PART_NAMES.index(a), PART_NAMES.index(b)) for a, b in D.PAIRS]
LENGTHS = np.array([s != "axis" for _ in D.HARMONICS for s in D.SIGNALS])   # which amplitudes are lengths
SIZE_STEPS = 17
FULL_HARMONICS = (1, 2, 3, 4)       # "full": every rhythm from once per clip (two cycles) to twice per cycle
POSTURE = ("x", "y", "size", "axis c", "axis s")     # "full": a part's averages over the clip


def _amplitudes_and_once(sig: np.ndarray, mass: np.ndarray = None):
    """(T, ..., 4) signals -> amplitudes (..., A) and once-per-cycle complex x, y (..., 2). With ``mass`` (T, ...),
    the share of the board a part covers, the full set: amplitudes at every FULL_HARMONICS rhythm, then POSTURE."""
    Z = np.fft.rfft(sig, axis=0) / len(sig)
    amps = []
    for k in (D.HARMONICS.values() if mass is None else FULL_HARMONICS):
        z = Z[k]
        amps += [np.abs(z[..., 0]), np.abs(z[..., 1]), np.sqrt(np.abs(z[..., 2]) ** 2 + np.abs(z[..., 3]) ** 2)]
    if mass is not None:
        amps += [Z[0][..., 0].real, Z[0][..., 1].real, mass.mean(0), Z[0][..., 2].real, Z[0][..., 3].real]
    return np.stack(amps, axis=-1), Z[D.HARMONICS["once"]][..., :2]


def _mass(boards: np.ndarray, salience: float) -> np.ndarray:
    """(T, ..., H, W) boards -> (T, ...): the share of the board that counts as the part."""
    return (boards >= salience).mean(axis=(-2, -1))


def _coupling(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    return 2 * np.real(z1 * np.conj(z2)) / np.maximum(np.abs(z1) ** 2 + np.abs(z2) ** 2, 1e-12)


def video_descriptors(boards: np.ndarray, salience: float, full: bool = False):
    """A video's part boards (T, P, H, W) -> amplitudes (P, A) and couplings (pairs, 2)."""
    amps, z = _amplitudes_and_once(D.board_signals(boards, salience), _mass(boards, salience) if full else None)
    return amps, np.array([_coupling(z[a], z[b]) for a, b in PAIRS])


def _own_descriptors(w: np.ndarray):
    """A claim-weighted board (T, H, W) -> full amplitudes (A,) and once-per-cycle x, y (2,); frames that draw nothing hold the
    nearest drawn centre and axis."""
    T = len(w)
    n = w.sum((1, 2))
    sig = D.board_signals(w[:, None], None)[:, 0]
    drawn = np.nonzero(n > 1e-9)[0]
    if len(drawn) < T:
        sig = sig[drawn[np.abs(np.arange(T)[:, None] - drawn[None]).argmin(1)]]
    a, z = _amplitudes_and_once(sig[:, None], w.mean((1, 2))[:, None])
    return a[0], z[0]


def _blob_background(T: int, N: int, n: int = 4000) -> dict:
    """Descriptors of random blobs, from no data: mean and variance of amplitudes, and of couplings of pairs of blobs."""
    rng = np.random.default_rng(0)
    A, Z = [], []
    while len(A) < n:
        h, w = rng.integers(1, 9, 2)
        r, c = rng.uniform(0, N - h), rng.uniform(0, N - w)
        amp, axis, k, ph = rng.integers(0, 4), rng.integers(0, 2), rng.integers(1, 5), rng.uniform(0, 2 * np.pi)
        b = np.zeros((T, N, N))
        for t in range(T):
            o = int(round(amp * np.sin(2 * np.pi * k * t / T + ph)))
            rr, cc = int(round(r)) + (o if axis == 0 else 0), int(round(c)) + (o if axis == 1 else 0)
            b[t, max(rr, 0):max(rr + h, 0), max(cc, 0):max(cc + w, 0)] = 1.0
        if (b.sum((1, 2)) > 0).any():
            a, z = _own_descriptors(b)
            A.append(a)
            Z.append(z)
    A, Z = np.stack(A), np.stack(Z)
    x = _coupling(Z[:n // 2], Z[n // 2:n // 2 * 2])
    return dict(amu=A.mean(0), avar=A.var(0), cmu=np.broadcast_to(x.mean(0), (len(PAIRS), 2)).copy(),
                cvar=np.broadcast_to(x.var(0), (len(PAIRS), 2)).copy())


class DescriptorLikelihood:
    """See the module docstring. ``observations``: each video's part boards (T, P, H, W)."""

    def __init__(self, observations, frames: int, size: int, salience: float, background: dict,
                 temperature: float = 1.0, draws: int = 256, noise_floor: float = 0.1, noise: dict = None,
                 cache_size: int = 1000, size_range: float = None, features: str = "motion",
                 open_cells: str = "fill", warmup: int = 0, open_variance: bool = False,
                 baseline: str = "parts") -> None:
        if features not in ("motion", "full"):
            raise ValueError(f"unknown descriptors {features!r}")
        if open_cells not in ("fill", "expected"):
            raise ValueError(f"unknown open_cells {open_cells!r}")
        self.expected = open_cells == "expected"
        self.warmup = int(warmup)
        if open_variance and not self.expected:
            raise ValueError("open_variance needs open_cells = 'expected'")
        self.open_variance = open_variance
        if baseline not in ("parts", "blobs"):
            raise ValueError(f"unknown baseline {baseline!r}")
        if baseline == "blobs" and (not self.expected or not features == "full"):
            raise ValueError("baseline 'blobs' needs open_cells = 'expected' and the full descriptors")
        self.blobs = baseline == "blobs"
        self.frames, self.size, self.salience, self.draws = frames, size, salience, draws
        self.full = features == "full"
        self.lengths = LENGTHS if not self.full else np.array(
            [s != "axis" for _ in FULL_HARMONICS for s in D.SIGNALS] + [False] * len(POSTURE))
        r = size_range or 1.0
        self.scales = np.exp(np.linspace(-np.log(r), np.log(r), SIZE_STEPS)) if r > 1 else np.ones(1)
        self.temperature = 1.0 if temperature is None else float(temperature)
        obs = [video_descriptors(b, salience, self.full) for b in observations]
        self.amps = np.stack([a for a, _ in obs])                          # (N, P, A)
        self.coup = np.stack([c for _, c in obs])                          # (N, pairs, 2)
        self.N, self.P, self.phases = len(obs), self.amps.shape[1], 1
        real = background["boards"]                                        # (videos x P, T, H, W), video-major
        bg = [video_descriptors(real[i:i + self.P].transpose(1, 0, 2, 3).astype(float), salience, self.full)
              for i in range(0, len(real) - self.P + 1, self.P)]
        ba, bc = np.stack([a for a, _ in bg]), np.stack([c for _, c in bg])
        self.bg = dict(amu=ba.mean(axis=(0, 1)), avar=ba.var(axis=(0, 1)),   # pooled over parts, like the scanner's
                       cmu=bc.mean(axis=0), cvar=bc.var(axis=0))
        self._real = real
        self._presence = (real >= salience).mean(axis=(0, 1))             # (H, W): a real part's average presence
        if noise is None:
            sized = self.amps
            if len(self.scales) > 1:                                       # each actor at the manner's size
                g = self.amps[..., self.lengths].mean(axis=(1, 2)) / self.amps[..., self.lengths].mean()
                sized = np.where(self.lengths, self.amps / g[:, None, None], self.amps)
            noise = dict(s2=sized.var(axis=0), s2d=self.coup.var(axis=0))
        self.s2 = np.maximum(noise["s2"], np.maximum(noise_floor * self.bg["avar"], 1e-6))      # (P, A)
        self.s2d = np.maximum(noise["s2d"], np.maximum(noise_floor * self.bg["cvar"], 1e-6))    # (pairs, 2)
        if self.blobs:                             # (the noise above keeps its floor)
            self.bg = _blob_background(frames, size)
        self._cache, self._cache_size = OrderedDict(), cache_size
        self._channel_cache = {}

    def _draws(self, T: int, H: int, W: int, k: int = 0):
        """Output k's fills (T, D, H, W) and uniforms (D, H, W); output 0's as ``GaussianScanLikelihood._draws`` (the
        same seeds), each later output its own."""
        cache = self.__dict__.setdefault("_draw_cache", {})
        if (T, H, W, k) not in cache:
            idx, uni = np.empty(self.draws, dtype=int), np.empty((self.draws, H, W))
            for d in range(self.draws):
                rng = np.random.Generator(np.random.PCG64(20_000 + k * self.draws + d))
                idx[d] = rng.integers(len(self._real))
                uni[d] = rng.random((H, W))
            cache[T, H, W, k] = (self._real[idx].transpose(1, 0, 2, 3), uni)
        return cache[T, H, W, k]

    def _channel(self, c: np.ndarray, k: int = 0, strength: np.ndarray = None):
        """Output k (T, H, W) -> per draw: amplitudes (D, A) and once-per-cycle x, y (D, 2). Memoised by value.
        ``strength``: how strongly each cell is claimed (expected boards; without it, read off the values)."""
        key = (_digest(c), k, None if strength is None else _digest(strength))
        hit = self._channel_cache.get(key)
        if hit is not None:
            return hit
        T, H, W = c.shape
        if self.blobs:
            hit = self._blob_channel(c, strength)
            if len(self._channel_cache) > MEMO_SIZE:
                self._channel_cache.clear()
            self._channel_cache[key] = hit
            return hit
        if self.expected:                                                  # (T, 1, H, W): one expected board
            from embodied_lot import _claimed_value
            if strength is None:
                claim, share = np.abs(2 * c - 1), (c > 0.5).astype(float)
            else:                                                          # the claims' share of figure
                claim = np.clip(strength, 0.0, 1.0)
                share = _claimed_value(c, claim)
            boards = (claim * share + (1 - claim) * self._presence)[:, None]
            hit = _amplitudes_and_once(D.board_signals(boards, None), boards.mean(axis=(-2, -1)) if self.full else None)
            if len(self._channel_cache) > MEMO_SIZE:
                self._channel_cache.clear()
            self._channel_cache[key] = hit
            return hit
        if np.all((c == 0) | (c == 1)):
            boards = c[:, None]                                            # (T, 1, H, W): exact, one draw
        else:
            fills, uniforms = self._draws(T, H, W, k)
            claim, fig = np.abs(2 * c - 1), c > 0.5
            boards = np.where(uniforms[None] < claim[:, None], fig[:, None], fills)
        boards = boards.astype(float)
        hit = _amplitudes_and_once(D.board_signals(boards, self.salience),
                                   _mass(boards, self.salience) if self.full else None)
        if len(self._channel_cache) > MEMO_SIZE:
            self._channel_cache.clear()
        self._channel_cache[key] = hit
        return hit

    def _blob_channel(self, c: np.ndarray, strength: np.ndarray):
        """``_channel`` under the blob baseline: amplitudes (1, A), once-per-cycle x, y (1, 2) and u."""
        from embodied_lot import _claimed_value
        claim = np.clip(strength, 0.0, 1.0)
        w = claim * _claimed_value(c, claim)
        mass = w.sum()
        if mass <= 1e-9 * len(w):
            return self.bg["amu"][None], np.zeros((1, 2), dtype=complex), 1.0
        a, z = _own_descriptors(w)
        return a[None], z[None], float(1.0 - (w * claim).sum() / mass)

    def unclaimed_share(self, c: np.ndarray, strength: np.ndarray) -> float:
        """Of an output's expected board (T, H, W) over the clip, the share that comes from unclaimed cells."""
        from embodied_lot import _claimed_value
        claim = np.clip(strength, 0.0, 1.0)
        own, rest = (claim * _claimed_value(c, claim)).sum(), ((1 - claim) * self._presence).sum()
        return float(rest / max(own + rest, 1e-12))

    def _matched(self, v: np.ndarray, strengths: np.ndarray = None):
        """Outputs (T, K, H, W) -> per video (rows, columns, NLL); rows >= K are background rows. With size factors,
        the NLL averages the factors' likelihoods and the matching is the most likely factor's."""
        from scipy.optimize import linear_sum_assignment
        from scipy.special import logsumexp
        K = v.shape[1]
        per = [self._channel(np.ascontiguousarray(v[:, k]), k,
                             None if strengths is None else np.ascontiguousarray(strengths[:, k])) for k in range(K)]
        b = max(0, self.P - K)
        mu = np.stack([p[0].mean(0) for p in per] + [self.bg["amu"]] * b)  # (K + b, A)
        var = np.stack([p[0].var(0) for p in per] + [self.bg["avar"]] * b)
        if self.blobs:                                # towards the background by how little is claimed
            u = np.array([p[2] for p in per])[:, None]
            mu[:K] = (1 - u) * mu[:K] + u * self.bg["amu"]
            var[:K] += u ** 2 * self.bg["avar"]
        if self.open_variance:
            u = [self.unclaimed_share(v[:, k], strengths[:, k]) for k in range(K)]
            var[:K] += np.array(u)[:, None] ** 2 * self.bg["avar"]
        own = (np.arange(K + b) < K)[:, None] & self.lengths[None]              # a program's lengths scale
        f = np.where(own[None], self.scales[:, None, None], 1.0)            # (S, K + b, A)
        V = (f ** 2 * var)[:, None, :, None] + self.s2[None, None, None]   # (S, 1, K + b, P, A)
        C = 0.5 * (((self.amps[None, :, None] - (f * mu)[:, None, :, None]) ** 2) / V
                   + np.log(2 * np.pi * V)).sum(-1)                         # (S, N, K + b, P)
        out = []
        for n in range(self.N):
            best = []
            for c in C[:, n]:
                r, col = linear_sum_assignment(c)
                nll = c[r, col].sum()
                of = {int(p): int(k) for k, p in zip(r, col) if k < K}      # part -> its output
                for j, (pa, pb) in enumerate(PAIRS):
                    if pa in of and pb in of:
                        x = _coupling(per[of[pa]][1], per[of[pb]][1])      # (D, 2) over the paired draws
                        m, sv = x.mean(0), x.var(0)
                        if self.blobs:
                            uu = max(per[of[pa]][2], per[of[pb]][2])
                            m, sv = (1 - uu) * m + uu * self.bg["cmu"][j], sv + uu ** 2 * self.bg["cvar"][j]
                    else:
                        m, sv = self.bg["cmu"][j], self.bg["cvar"][j]
                    sv = sv + self.s2d[j]
                    nll += 0.5 * (((self.coup[n, j] - m) ** 2) / sv + np.log(2 * np.pi * sv)).sum()
                best.append((r, col, float(nll)))
            nlls = np.array([x[2] for x in best])
            r, col, _ = best[int(nlls.argmin())]
            out.append((r, col, float(nlls[0] if len(nlls) == 1 else -(logsumexp(-nlls) - np.log(len(nlls))))))
        return out

    def _scores(self, p):
        from grace_lot.lot_adapter.adapter import motion_combinations, rollout
        L, matched = [], []
        for states, weight in motion_combinations(p, self.size):
            v = rollout(p, self.frames + self.warmup, self.size, states, claims=self.expected)
            v, cl = v if self.expected else (v, None)
            v, cl = v[self.warmup:], None if cl is None else cl[self.warmup:]
            v, cl = (v, cl) if v.ndim == 4 else (v[:, None], None if cl is None else cl[:, None])
            mt = self._matched(v, cl)
            matched.append(mt)
            L.append(np.log(weight) - self.temperature * np.array([nll for _, _, nll in mt]))
        return np.array(L), matched

    def per_video_scores(self, p) -> np.ndarray:
        from scipy.special import logsumexp
        L = self._scores(p)[0]
        return L[0] if len(L) == 1 else logsumexp(L, axis=0)

    def assignments(self, p):
        """Per video, under its most probable state combination: (rows, columns, NLL x temperature / P)."""
        L, matched = self._scores(p)
        best = L.argmax(axis=0)
        return [(r, c, nll * self.temperature / self.P) for (r, c, nll) in (matched[b][n] for n, b in enumerate(best))]

    def __call__(self, p) -> float:
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        value = float(self.per_video_scores(p).sum())
        self._cache[p] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value
