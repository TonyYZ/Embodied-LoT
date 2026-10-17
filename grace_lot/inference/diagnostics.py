"""Checks that the sampler can reach, and correctly weights, every legal program.

1. ``reachability`` (exhaustive, for small grammars): enumerate every program
   with finite prior and every exact result of every proposal move. Then
   check that one breadth-first search from a single leaf visits the whole
   support. Every move has a reverse move with positive probability, so
   this shows the chain is irreducible.

2. ``family_coverage`` (for the configured grammar, which is too large to
   enumerate): sort programs into *families*. A family is the set of
   composition types used, whether dynamic leaves occur, the number of
   dashed edges, and the depth. Compare family frequencies under a
   flat-likelihood MCMC run (whose target is the prior) with an independent
   importance-sampling estimate of the prior built from the generator. Any
   family with clear prior mass that MCMC never visits is a coverage failure.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, deque
from typing import Dict, Iterator, List, Set, Tuple

from grace_lot.inference.grammar import P_DYNAMIC, Grammar
from grace_lot.inference.mh import run_chain
from grace_lot.inference.prior import SizePrior
from grace_lot.inference.proposals import ProposalMixture, _addable, _free_sites
from grace_lot.lot_adapter.program import (LEAF, MIN_CHILDREN, P, Program, candidate_edges, depth, nodes,
                                           replace, size, with_edges)


# ---- exhaustive enumeration -------------------------------------------------

def enumerate_trees(g: Grammar, d: int = 0, budget: int = None) -> List[P]:
    """Every tree the generator can produce at depth ``d`` with at most ``budget`` nodes."""
    budget = g.max_nodes if budget is None else budget
    if budget < 1:
        return []
    if g.multi_output(d) and g.output_leaves:       # one plain tree, or a forest of 2+ trees (no node of its own)
        return enumerate_trees(g, 1, budget) + [P("outputs", (), kids) for k in g.child_counts("outputs")
                                                for kids in _child_lists(g, 1, budget, k)]
    if g.multi_output(d):
        return [P("outputs", (), kids) for k in g.child_counts("outputs")
                for kids in _child_lists(g, 1, budget - 1, k)]
    out = [P(LEAF, (n, o, dy)) for n in g.primitives for o in g.orientations for dy in g.dynamics]
    if g.output_leaves:
        out.append(P("out"))
    if g.leaf_forced(d):
        return out
    for kind in g.compositions:
        params = [()] if kind != "body" else list(itertools.product(*[g.body_grid(i) for i in range(g.n_body_params)]))
        for k in g.child_counts(kind):
            for kids in _child_lists(g, d + 1, budget - 1, k):
                out += [P(kind, pr, kids) for pr in params]
    return out


def _child_lists(g, d, budget, k):
    if k == 0:
        yield ()
        return
    for first in enumerate_trees(g, d, budget - (k - 1)):
        for rest in _child_lists(g, d, budget - size(first), k - 1):
            yield (first,) + rest


def enumerate_support(g: Grammar, prior: SizePrior) -> List[Program]:
    out = []
    for t in enumerate_trees(g):
        paths = [p for p, _ in nodes(t)]
        edges = candidate_edges(t, g.dashed_targets)
        for k in range(g.max_dashed + 1):
            for es in itertools.combinations(edges, k):
                prog = with_edges(t, es)
                if prior(prog) > -math.inf:
                    out.append(prog)
    return out


def neighbours(prog: Program, g: Grammar) -> Iterator[Program]:
    """Every program that one proposal (of any enabled move type) can produce from ``prog``."""
    t = prog.tree
    for path in _free_sites(prog):                                    # regenerate
        old = _get(t, path)
        for new in enumerate_trees(g, len(path), g.max_nodes - size(t) + size(old)):
            yield Program(replace(t, path, new), prog.dashed)
    for path, n in nodes(t):
        if n.kind == LEAF:                                            # leaf attributes
            for attr, domain in enumerate([g.primitives, g.orientations, g.dynamics]):
                for v in domain:
                    if v != n.params[attr]:
                        params = list(n.params); params[attr] = v
                        yield Program(replace(t, path, n._replace(params=tuple(params))), prog.dashed)
        elif n.kind not in ("body", "outputs", "out", "in"):          # composition type
            for k in g.compositions:
                if k not in (n.kind, "body") and len(n.children) >= MIN_CHILDREN[k]:
                    yield Program(replace(t, path, n._replace(kind=k)), prog.dashed)
        elif n.kind == "body":                                        # body parameters
            for i in range(g.n_body_params):
                for v in g.body_grid(i):
                    if v != n.params[i]:
                        params = list(n.params); params[i] = v
                        yield Program(replace(t, path, n._replace(params=tuple(params))), prog.dashed)
        for i, j in itertools.combinations(range(len(n.children)), 2):   # swap (edges follow)
            kids = list(n.children); kids[i], kids[j] = kids[j], kids[i]
            d = len(path)
            mv = lambda e: (e[:d] + ((j if e[d] == i else i),) + e[d + 1:]
                            if e[:d] == path and len(e) > d and e[d] in (i, j) else e)
            yield with_edges(replace(t, path, n._replace(children=tuple(kids))),
                             [(mv(a), mv(b)) for a, b in prog.dashed])
    if len(prog.dashed) < g.max_dashed:                               # add dashed edge
        for e in _addable(prog, g):
            yield with_edges(t, list(prog.dashed) + [e])
    for e in prog.dashed:                                             # remove dashed edge
        yield with_edges(t, [x for x in prog.dashed if x != e])
    wrap_kinds = [k for k in g.compositions if MIN_CHILDREN[k] == 1]
    for path, n in nodes(t):                                          # wrap
        d = len(path)
        shift = lambda e: e[:d] + (0,) + e[d:] if e[:d] == path else e
        for k in wrap_kinds:
            for pr in ([()] if k != "body" else itertools.product(*[g.body_grid(i) for i in range(g.n_body_params)])):
                yield with_edges(replace(t, path, P(k, tuple(pr), (n,))), [(shift(a), shift(b)) for a, b in prog.dashed])
    ends = {e for edge in prog.dashed for e in edge}
    for path, n in nodes(t):                                          # unwrap
        if n.kind in wrap_kinds and len(n.children) == 1 and path not in ends:
            d = len(path)
            shift = lambda e: e[:d] + e[d + 1:] if e[:d] == path else e
            yield with_edges(replace(t, path, n.children[0]), [(shift(a), shift(b)) for a, b in prog.dashed])
    for path, n in nodes(t):                                          # add / remove child
        if n.kind in (LEAF, "out", "in"):
            continue
        d = len(path)
        if len(n.children) < max(g.child_counts(n.kind)):
            for i in range(len(n.children) + 1):
                shift = lambda e: e[:d] + (e[d] + 1,) + e[d + 1:] if e[:d] == path and len(e) > d and e[d] >= i else e
                for sub in enumerate_trees(g, d + 1, g.max_nodes - size(t)):
                    kids = list(n.children); kids.insert(i, sub)
                    yield with_edges(replace(t, path, n._replace(children=tuple(kids))),
                                     [(shift(a), shift(b)) for a, b in prog.dashed])
        if MIN_CHILDREN[n.kind] < len(n.children) <= max(g.child_counts(n.kind)):
            for i in range(len(n.children)):
                if any(e[:d + 1] == path + (i,) for edge in prog.dashed for e in edge):
                    continue
                shift = lambda e: e[:d] + (e[d] - 1,) + e[d + 1:] if e[:d] == path and len(e) > d and e[d] > i else e
                kids = list(n.children); del kids[i]
                yield with_edges(replace(t, path, n._replace(children=tuple(kids))),
                                 [(shift(a), shift(b)) for a, b in prog.dashed])


def _get(t, path):
    for i in path:
        t = t.children[i]
    return t


def reachability(g: Grammar, prior: SizePrior) -> Dict[str, object]:
    support = set(enumerate_support(g, prior))
    leaf0 = P(LEAF, (g.primitives[0], "V", False))
    if g.output_leaves:                       # the smallest program with a window: (parallel leaf out)
        start = Program(P("parallel", (), (leaf0, P("out"))))
    else:
        start = Program(P("outputs", (), (leaf0,)) if g.max_outputs else leaf0)
    seen: Set[Program] = {start}
    queue = deque([start])
    while queue:
        for nb in neighbours(queue.popleft(), g):
            if nb not in seen and prior(nb) > -math.inf:
                seen.add(nb)
                queue.append(nb)
    missing = support - seen
    return {"support": len(support), "reached": len(seen & support), "outside_support": len(seen - support),
            "missing_examples": [str(p) for p in list(missing)[:5]]}


# ---- family coverage for large grammars --------------------------------------

def family(prog: Program) -> Tuple:
    kinds = tuple(sorted({n.kind for _, n in nodes(prog.tree) if n.kind != LEAF}))
    dyn = any(n.kind == LEAF and n.params[2] for _, n in nodes(prog.tree))
    return kinds, dyn, len(prog.dashed), depth(prog.tree)


def importance_family_prior(g: Grammar, prior: SizePrior, n: int, rng: random.Random) -> Counter:
    """Estimate the prior over families with importance sampling from an independent proposal.

    The proposal is: a tree from the generator, k ~ Uniform{0..max_dashed},
    then k distinct addable edges uniformly at random.
    """
    w = Counter()
    for _ in range(n):
        t = g.sample(rng)
        prog = Program(t)
        k = rng.randrange(g.max_dashed + 1)
        options = _addable(prog, g)
        if k > len(options):
            continue
        edges = rng.sample(options, k)
        prog = with_edges(t, edges)
        lp = prior(prog)
        if lp == -math.inf:
            continue
        log_q = g.log_prob(t) - math.log(g.max_dashed + 1) - math.log(math.comb(len(options), k))
        w[family(prog)] += math.exp(lp - log_q)
    total = sum(w.values())
    return Counter({f: v / total for f, v in w.items()})


def family_coverage(g: Grammar, prior: SizePrior, weights: Dict[str, float], iterations: int,
                    is_samples: int, seed: int = 0) -> Dict[str, object]:
    rng = random.Random(seed)
    ref = importance_family_prior(g, prior, is_samples, rng)
    chain = run_chain(prior, lambda p: 0.0, ProposalMixture(g, weights), iterations, iterations // 10, 5, rng)
    emp = Counter(family(p) for p in chain.samples)
    n = len(chain.samples)
    rows = sorted(ref, key=lambda f: -ref[f])
    unvisited = [(f, ref[f]) for f in rows if ref[f] > 5 / n and emp[f] == 0]
    tv = 0.5 * sum(abs(emp[f] / n - ref[f]) for f in set(ref) | set(emp))
    return {"families_with_prior_mass": len(ref), "families_visited": len(emp),
            "unvisited_with_mass": [(str(f), round(p, 5)) for f, p in unvisited],
            "tv_family_marginal": tv,
            "top": [(str(f), round(emp[f] / n, 4), round(ref[f], 4)) for f in rows[:12]]}
