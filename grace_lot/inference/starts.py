"""Data-driven starting programs (``inference.start = "parts"``).

A chain that starts from a random program spends its burn-in climbing towards the region where good programs are.
This starts it nearer, one observed part at a time:
  1. draw a fresh pool of forest trees from the grammar, each with exactly one output window and at least
     ``min_size`` nodes (a bare window leaves the moves almost nothing to work on), and score each alone
     against every part (its matched cost, summed over videos, at its best motion realization);
  2. for each part, hill-climb from its best pool tree with the sampler's own moves for ``steps`` steps, scored by
     that tree's prior plus its tempered fit to that part alone;
  3. keep each part whose fitted tree beats the background (the "no output" prediction), most improved first, up to
     ``max_outputs`` trees; identical trees are fine, each has its own window.
Each call draws its own pool and climbs with its own random moves, so chains (and replicas) start apart, as split
R-hat needs.

Starting points only shape the burn-in: the sampler, and so the posterior, is unchanged.
"""

from __future__ import annotations

import math
import random

import numpy as np

from grace_lot.inference.grammar import Grammar
from grace_lot.inference.mh import initial_state
from grace_lot.lot_adapter.program import OUT, OUTPUTS, P, Program, nodes, size


def tree_part_costs(lik, tree: P) -> tuple:
    """(P,) cost of one single-window tree against each part, and (P,) the background's cost for each part."""
    best = None
    for m in lik.moments(Program(tree)):
        C = lik.costs(m).sum(axis=0)                     # (1 + background rows, P), summed over videos
        best = C if best is None else np.minimum(best, C)
    return best[0], (best[-1] if best.shape[0] > 1 else np.full(best.shape[1], np.inf))


def _single_window(t: P) -> bool:
    kinds = [n.kind for _, n in nodes(t)]
    return t.kind != OUTPUTS and kinds.count(OUT) == 1 and "in" not in kinds


def part_fit_start(lik, proposal, log_prior, rng: random.Random, candidates: int = 100, steps: int = 200,
                   min_size: int = 3) -> Program:
    """A forest of single-window trees, each fitted to the observed part it explains best."""
    grammar: Grammar = proposal.grammar
    pool = []
    for _ in range(50 * candidates):                     # draws: most generator trees have no window
        if len(pool) == candidates:
            break
        t = grammar.sample(rng, 1)                       # a forest tree
        if size(t) >= min_size and _single_window(t) and log_prior(Program(t)) > -math.inf:
            pool.append(t)
    if not pool:
        return initial_state(grammar, log_prior, rng)
    scored = [(t,) + tree_part_costs(lik, t) for t in pool]
    background = scored[0][2]
    tau = lik.temperature
    fitted = []
    for p in range(lik.P):
        tree, cost, _ = min(scored, key=lambda s: s[1][p])
        score = log_prior(Program(tree)) - tau * cost[p]
        for _ in range(steps):                           # greedy: a starting point, not a posterior sample
            _, r = proposal(Program(tree), rng)
            if r is None or r[0].dashed or not _single_window(r[0].tree):
                continue
            lp = log_prior(r[0])
            if lp == -math.inf:
                continue
            s = lp - tau * tree_part_costs(lik, r[0].tree)[0][p]
            if s > score:
                tree, score = r[0].tree, s
        gain = tau * background[p] + score               # tempered fit gained over the background, net of the prior
        if gain > 0:
            fitted.append((gain, tree))
    trees = [t for _, t in sorted(fitted, key=lambda f: -f[0])][:max(1, grammar.max_outputs)]
    while trees:
        prog = Program(P(OUTPUTS, (), tuple(trees))) if len(trees) > 1 else Program(trees[0])
        if log_prior(prog) > -math.inf:
            return prog
        trees = trees[:-1]                               # outside the language as a forest: drop the weakest tree
    return initial_state(grammar, log_prior, rng)
