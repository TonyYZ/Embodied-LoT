"""Metropolis-Hastings over programs, with full per-iteration diagnostics.

    accept with probability  min(1, exp([log p(r') + tau log p(v|r') + log q(r|r')]
                                        - [log p(r)  + tau log p(v|r)  + log q(r'|r)]))

``tau = 1`` is the posterior. ``run_tempered`` runs a ladder of replicas at
tau_0 = 1 > tau_1 > ... (the prior is never tempered) and proposes swaps
between neighbouring replicas: parallel tempering. Hot replicas move
freely between modes and pass good programs down to the cold one. Only the
tau = 1 replica is recorded as posterior samples. Optionally, neighbouring
replicas also exchange subtrees (crossover), so pieces a hot replica found can
reach the cold one without being rebuilt node by node. ``run_population`` runs
several ladders as one search, with selection among them during the burn-in.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from grace_lot.inference.grammar import Grammar
from grace_lot.inference.proposals import ProposalMixture, _free_sites, _get
from grace_lot.lot_adapter.program import P, Program, replace


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
    crossovers: List[int] = field(default_factory=lambda: [0, 0])   # subtree exchanges between replicas: tried, accepted
    replaced: int = 0                                      # population selection: times continued from a better ladder

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


def _crossover(a: _State, b: _State, tau_a: float, tau_b: float, log_prior, log_lik, rng: random.Random) -> bool:
    """Exchange one subtree between two replicas' programs; accept with the joint tempered MH rule.

    Each side picks uniformly among its subtrees that no dashed edge touches (not the whole program; a whole tree
    of a forest is allowed). Picking the same two paths again undoes the exchange, so the Hastings ratio is
    |S_a| |S_b| / (|S_a'| |S_b'|), the numbers of such sites before and after.
    """
    sites = lambda p: [s for s in _free_sites(p) if s]
    sa, sb = sites(a.program), sites(b.program)
    if not sa or not sb:
        return False
    pa, pb = rng.choice(sa), rng.choice(sb)
    ta, tb = _get(a.program.tree, pa), _get(b.program.tree, pb)
    if ta == tb:
        return False
    na = Program(replace(a.program.tree, pa, tb), a.program.dashed)
    nb = Program(replace(b.program.tree, pb, ta), b.program.dashed)
    lpa, lpb = log_prior(na), log_prior(nb)
    if lpa == -math.inf or lpb == -math.inf:
        return False
    lla, llb = log_lik(na), log_lik(nb)
    log_alpha = ((lpa + tau_a * lla + lpb + tau_b * llb) - (a.lp + tau_a * a.ll + b.lp + tau_b * b.ll)
                 + math.log(len(sa) * len(sb)) - math.log(len(sites(na)) * len(sites(nb))))
    if log_alpha >= 0 or rng.random() < math.exp(log_alpha):
        a.program, a.lp, a.ll = na, lpa, lla
        b.program, b.lp, b.ll = nb, lpb, llb
        return True
    return False


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
                 adapt_until: int = 0, crossover: float = 0.0,
                 init: Optional[Callable[[random.Random], P]] = None) -> Chain:
    """Parallel tempering. Returns the tau = 1 chain; swap statistics are in ``chain.swaps``.

    ``progress(iteration, chain so far)`` is called every ``progress_every`` iterations.

    With ``adapt_until`` > 0 the ladder's spacing adapts during the first ``adapt_until`` iterations (keep it within
    the burn-in) so that every neighbouring pair swaps at the same rate (Vousden, Farr & Mandel 2016). Its ends stay
    fixed (tau = 1 and the given hottest tau): with temperatures T = 1 / tau, each gap log(T[k+1] - T[k]) takes a
    Robbins-Monro step by how much its pair's running swap rate exceeds the mean, and the gaps are rescaled to keep
    the extent. A fixed target rate would not do: replicas hot enough to sample the prior swap at rate 1 however
    far apart they are. Afterwards the ladder is fixed and the sampler is ordinary tempering: ``chain.swaps`` counts
    only the swaps after adaptation, and ``chain.taus`` is the ladder used.

    With ``crossover`` > 0, each sweep also tries, with that probability, to exchange a subtree between a random
    neighbouring pair of replicas (``_crossover``). ``init(rng)`` gives each replica's starting program (default:
    a draw from the generator).
    """
    ladder = _Ladder(log_prior, log_lik, proposal, rng, taus, swap_every, adapt_until, crossover, init)
    for it in range(iterations):
        ladder.sweep(it, burn_in, thin)
        if progress and progress_every and (it + 1) % progress_every == 0:
            progress(it + 1, ladder.chain)
    return ladder.finish()


class _Ladder:
    """One parallel-tempering ladder, advanced one sweep at a time (``run_tempered``, ``run_population``)."""

    def __init__(self, log_prior, log_lik, proposal, rng, taus, swap_every, adapt_until, crossover, init):
        self.log_prior, self.log_lik, self.proposal, self.rng = log_prior, log_lik, proposal, rng
        self.swap_every, self.adapt_until, self.crossover = swap_every, adapt_until, crossover
        self.taus = list(taus)
        self.gaps = np.log(np.diff(1.0 / np.asarray(taus, float))) if adapt_until and len(taus) > 2 else None
        self.extent = 1.0 / self.taus[-1] - 1.0
        self.rates = np.full(len(taus) - 1, 0.5)            # running swap rate per pair, while adapting
        self.chain = Chain()
        self.states = []
        for _ in taus:
            p = init(rng) if init else initial_state(proposal.grammar, log_prior, rng)
            self.states.append(_State(p, log_prior(p), log_lik(p)))
        self.tried, self.done = np.zeros(len(taus) - 1), np.zeros(len(taus) - 1)

    @property
    def log_post(self) -> float:
        """The tau = 1 replica's current log posterior."""
        return self.states[0].lp + self.states[0].ll

    def sweep(self, it: int, burn_in: int, thin: int) -> None:
        states, taus, rng, chain = self.states, self.taus, self.rng, self.chain
        for k, (s, tau) in enumerate(zip(states, taus)):
            _mh_step(s, tau, self.log_prior, self.log_lik, self.proposal, rng, chain if k == 0 else None)
        if len(taus) > 1 and it % self.swap_every == 0:
            k = rng.randrange(len(taus) - 1)          # one random neighbouring pair per sweep
            a, b = states[k], states[k + 1]
            log_alpha = (taus[k] - taus[k + 1]) * (b.ll - a.ll)
            self.tried[k] += 1
            if self.gaps is not None and it < self.adapt_until:     # no extra random draws: same stream as without
                gaps, rates = self.gaps, self.rates
                step = (self.tried[k] + 1) ** -0.6
                rates[k] += step * (math.exp(min(log_alpha, 0.0)) - rates[k])
                gaps[k] += step * (rates[k] - rates.mean())
                gaps += math.log(self.extent) - math.log(np.exp(gaps).sum())      # keep both ends of the ladder
                taus[1:] = (1.0 / (1.0 + np.cumsum(np.exp(gaps)))).tolist()
            if log_alpha >= 0 or rng.random() < math.exp(log_alpha):
                states[k], states[k + 1] = b, a
                self.done[k] += 1
        if self.crossover and len(taus) > 1 and rng.random() < self.crossover:     # no extra draws when off
            k = rng.randrange(len(taus) - 1)
            chain.crossovers[0] += 1
            chain.crossovers[1] += _crossover(states[k], states[k + 1], taus[k], taus[k + 1],
                                              self.log_prior, self.log_lik, rng)
        if self.gaps is not None and it + 1 == self.adapt_until:
            self.tried[:] = self.done[:] = 0                # report the swap rates of the fixed ladder
        _record(chain, states[0], it, burn_in, thin)

    def take(self, other: "_Ladder") -> None:
        """Continue from another ladder's replicas (and its ladder spacing); keep this one's records and stream."""
        self.states = [_State(s.program, s.lp, s.ll) for s in other.states]
        self.taus = list(other.taus)
        if other.gaps is not None:
            self.gaps, self.rates = other.gaps.copy(), other.rates.copy()
        self.chain.replaced += 1

    def finish(self) -> Chain:
        self.chain.swaps = (self.done / np.maximum(self.tried, 1)).tolist()
        self.chain.taus = list(self.taus)
        return self.chain


def run_population(n: int, log_prior: Callable[[P], float], log_lik: Callable[[P], float],
                   proposal: ProposalMixture, iterations: int, burn_in: int, thin: int,
                   rng: random.Random, taus: List[float], swap_every: int = 1,
                   progress: Optional[Callable[[int, List[Chain]], None]] = None, progress_every: int = 0,
                   adapt_until: int = 0, crossover: float = 0.0,
                   init: Optional[Callable[[random.Random], P]] = None,
                   select_every: int = 0) -> List[Chain]:
    """``n`` tempered ladders run side by side (one search), with selection during the burn-in.

    Every ``select_every`` iterations until ``burn_in``, the ladder whose tau = 1 replica has the lowest log
    posterior continues from a copy of the highest one's replicas (``Chain.replaced`` counts it), so compute is not
    spent on ladders stuck far below the others. Selection only chooses where ladders stand at the end of the
    burn-in; afterwards each is ordinary tempering, so the samples are valid. The ladders are no longer
    independent, though: agreement within a search says nothing about convergence; compare separate searches.
    Each ladder draws from its own stream, seeded from ``rng``.
    """
    ladders = [_Ladder(log_prior, log_lik, proposal, random.Random(rng.getrandbits(64)), taus, swap_every,
                       adapt_until, crossover, init) for _ in range(n)]
    for it in range(iterations):
        for ladder in ladders:
            ladder.sweep(it, burn_in, thin)
        if select_every and n > 1 and it + 1 < burn_in and (it + 1) % select_every == 0:
            ranked = sorted(ladders, key=lambda l: l.log_post)
            ranked[0].take(ranked[-1])
        if progress and progress_every and (it + 1) % progress_every == 0:
            progress(it + 1, [l.chain for l in ladders])
    return [l.finish() for l in ladders]


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
