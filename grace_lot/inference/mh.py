"""Metropolis-Hastings over programs, with full per-iteration diagnostics.

    accept with probability  min(1, exp([log p(r') + tau log p(v|r') + log q(r|r')]
                                        - [log p(r)  + tau log p(v|r)  + log q(r'|r)]))

``tau = 1`` is the posterior. ``run_tempered`` runs a ladder of replicas at
tau_0 = 1 > tau_1 > ... (the prior is never tempered) and proposes swaps
between neighbouring replicas: parallel tempering. Hot replicas move
freely between modes and pass good programs down to the cold one. Only the
tau = 1 replica is recorded as posterior samples.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from grace_lot.inference.grammar import Grammar
from grace_lot.inference.proposals import ProposalMixture
from grace_lot.lot_adapter.program import P, Program


@dataclass
class Chain:
    log_post: List[float] = field(default_factory=list)    # every iteration
    log_lik: List[float] = field(default_factory=list)
    log_prior: List[float] = field(default_factory=list)
    moves: Counter = field(default_factory=Counter)        # proposed, per move type
    accepted: Counter = field(default_factory=Counter)     # accepted, per move type
    samples: List[P] = field(default_factory=list)         # after burn-in, thinned
    best: Optional[P] = None
    best_log_post: float = -math.inf
    swaps: List[float] = field(default_factory=list)       # parallel tempering: swap acceptance per neighbouring pair
    taus: List[float] = field(default_factory=list)        # parallel tempering: the ladder used (after any adaptation)

    @property
    def acceptance_rate(self) -> float:
        return sum(self.accepted.values()) / max(sum(self.moves.values()), 1)


def initial_state(grammar: Grammar, log_prior: Callable[[P], float], rng: random.Random) -> P:
    while True:
        p = Program(grammar.sample(rng))
        if log_prior(p) > -math.inf:
            return p


@dataclass
class _State:
    program: P
    lp: float
    ll: float


def _mh_step(s: _State, tau: float, log_prior, log_lik, proposal: ProposalMixture,
             rng: random.Random, chain: Optional[Chain]) -> None:
    name, prop = proposal(s.program, rng)
    if chain is not None:
        chain.moves[name] += 1
    if prop is None:
        return
    new, log_fwd, log_bwd = prop
    new_lp = log_prior(new)
    if new_lp == -math.inf:
        return
    new_ll = log_lik(new)
    log_alpha = (new_lp + tau * new_ll + log_bwd) - (s.lp + tau * s.ll + log_fwd)
    if log_alpha >= 0 or rng.random() < math.exp(log_alpha):
        s.program, s.lp, s.ll = new, new_lp, new_ll
        if chain is not None:
            chain.accepted[name] += 1


def _record(chain: Chain, s: _State, it: int, burn_in: int, thin: int) -> None:
    chain.log_post.append(s.lp + s.ll); chain.log_lik.append(s.ll); chain.log_prior.append(s.lp)
    if s.lp + s.ll > chain.best_log_post:
        chain.best, chain.best_log_post = s.program, s.lp + s.ll
    if it >= burn_in and (it - burn_in) % thin == 0:
        chain.samples.append(s.program)


def run_chain(log_prior: Callable[[P], float], log_lik: Callable[[P], float],
              proposal: ProposalMixture, iterations: int, burn_in: int, thin: int,
              rng: random.Random, init: Optional[P] = None) -> Chain:
    """Plain MH at tau = 1."""
    chain = Chain()
    p = init or initial_state(proposal.grammar, log_prior, rng)
    p = p if isinstance(p, Program) else Program(p)
    s = _State(p, log_prior(p), log_lik(p))
    for it in range(iterations):
        _mh_step(s, 1.0, log_prior, log_lik, proposal, rng, chain)
        _record(chain, s, it, burn_in, thin)
    return chain


def temperature_ladder(n: int, tau_min: float) -> List[float]:
    """n inverse temperatures, geometric from 1 down to tau_min."""
    return [1.0] if n <= 1 else list(np.geomspace(1.0, tau_min, n))


def calibrated_ladder(inf: dict, log_prior, log_lik, grammar, rng: random.Random) -> List[float]:
    """A ladder scaled to the likelihood itself (``inference.ladder = "calibrated"``).

    ``ladder_samples`` programs drawn from the prior are scored; the hottest replica's inverse temperature
    shrinks the spread (standard deviation) of their log-likelihoods to ``hot_spread`` nats, so it moves
    almost as the prior does whatever the likelihood's scale (the Gaussian likelihood ignores ``beta``).
    """
    lls = [log_lik(initial_state(grammar, log_prior, rng)) for _ in range(inf.get("ladder_samples", 32))]
    spread = float(np.std(lls))
    tau_min = min(1.0, inf.get("hot_spread", 1.0) / spread) if spread > 0 else 1.0
    return temperature_ladder(inf.get("replicas", 1), tau_min)


def ladder_for(inf: dict, n_observations: int) -> List[float]:
    """The ladder for a run. The hottest replica sees a total likelihood weight of
    ``hot_beta`` (beta x n_observations x tau_min = hot_beta), so the ladder adapts to beta
    and to how many videos are pooled."""
    tau_min = min(1.0, inf["hot_beta"] / (inf["beta"] * n_observations))
    return temperature_ladder(inf.get("replicas", 1), tau_min)


def run_tempered(log_prior: Callable[[P], float], log_lik: Callable[[P], float],
                 proposal: ProposalMixture, iterations: int, burn_in: int, thin: int,
                 rng: random.Random, taus: List[float], swap_every: int = 1,
                 progress: Optional[Callable[[int, Chain], None]] = None, progress_every: int = 0,
                 adapt_until: int = 0) -> Chain:
    """Parallel tempering. Returns the tau = 1 chain; swap statistics are in ``chain.swaps``.

    ``progress(iteration, chain so far)`` is called every ``progress_every`` iterations.

    With ``adapt_until`` > 0 the ladder's spacing adapts during the first ``adapt_until`` iterations (keep it within
    the burn-in) so that every neighbouring pair swaps at the same rate (Vousden, Farr & Mandel 2016). Its ends stay
    fixed (tau = 1 and the given hottest tau): with temperatures T = 1 / tau, each gap log(T[k+1] - T[k]) takes a
    Robbins-Monro step by how much its pair's running swap rate exceeds the mean, and the gaps are rescaled to keep
    the extent. A fixed target rate would not do: replicas hot enough to sample the prior swap at rate 1 however
    far apart they are. Afterwards the ladder is fixed and the sampler is ordinary tempering: ``chain.swaps`` counts
    only the swaps after adaptation, and ``chain.taus`` is the ladder used.
    """
    taus = list(taus)
    gaps = np.log(np.diff(1.0 / np.asarray(taus, float))) if adapt_until and len(taus) > 2 else None
    extent = 1.0 / taus[-1] - 1.0
    rates = np.full(len(taus) - 1, 0.5)                      # running swap rate per pair, while adapting
    chain = Chain()
    states = []
    for _ in taus:
        p = initial_state(proposal.grammar, log_prior, rng)
        states.append(_State(p, log_prior(p), log_lik(p)))
    tried, done = np.zeros(len(taus) - 1), np.zeros(len(taus) - 1)
    for it in range(iterations):
        for k, (s, tau) in enumerate(zip(states, taus)):
            _mh_step(s, tau, log_prior, log_lik, proposal, rng, chain if k == 0 else None)
        if len(taus) > 1 and it % swap_every == 0:
            k = rng.randrange(len(taus) - 1)          # one random neighbouring pair per sweep
            a, b = states[k], states[k + 1]
            log_alpha = (taus[k] - taus[k + 1]) * (b.ll - a.ll)
            tried[k] += 1
            if gaps is not None and it < adapt_until:        # no extra random draws: same stream as without
                step = (tried[k] + 1) ** -0.6
                rates[k] += step * (math.exp(min(log_alpha, 0.0)) - rates[k])
                gaps[k] += step * (rates[k] - rates.mean())
                gaps += math.log(extent) - math.log(np.exp(gaps).sum())      # keep both ends of the ladder
                taus[1:] = (1.0 / (1.0 + np.cumsum(np.exp(gaps)))).tolist()
            if log_alpha >= 0 or rng.random() < math.exp(log_alpha):
                states[k], states[k + 1] = b, a
                done[k] += 1
        if gaps is not None and it + 1 == adapt_until:
            tried[:] = done[:] = 0                          # report the swap rates of the fixed ladder
        _record(chain, states[0], it, burn_in, thin)
        if progress and progress_every and (it + 1) % progress_every == 0:
            progress(it + 1, chain)
    chain.swaps = (done / np.maximum(tried, 1)).tolist()
    chain.taus = list(taus)
    return chain


def r_hat(traces: List[List[float]]) -> float:
    """Split Gelman-Rubin potential scale reduction of a scalar trace (here, the log-posterior).

    Each chain is cut into halves treated as separate chains, so a chain still drifting is flagged too.
    """
    x = np.array([t for t in traces], float)
    half = x.shape[1] // 2
    x = np.concatenate([x[:, :half], x[:, x.shape[1] - half:]]) if half else x
    m, n = x.shape
    if m < 2 or n < 2:
        return float("nan")
    W = x.var(axis=1, ddof=1).mean()
    B = n * x.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else float("nan")


def posterior(chains: List[Chain]) -> Dict[P, float]:
    """Pooled sample frequencies: program -> posterior weight."""
    counts = Counter(s for c in chains for s in c.samples)
    total = sum(counts.values())
    return {p: n / total for p, n in counts.most_common()}
