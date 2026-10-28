"""MH proposal moves derived from the grammar.

Each move returns ``(new_program, log_q_forward, log_q_backward)``, or None if
it does not apply to the current program. A None counts as a rejected step,
which keeps detailed balance. The move type is drawn with fixed probabilities
that do not depend on the state. For moves that undo themselves, those
probabilities cancel. ``add_dashed`` and ``remove_dashed`` undo each other,
so their q terms include the move weights.

| move          | effect                                                    | Hastings term                    |
|---------------|-----------------------------------------------------------|----------------------------------|
| regenerate    | replace a subtree not touched by any dashed edge with a   | 1/#free sites x generator        |
|               | fresh sample from the grammar (insert/delete/replace)     | log-probability, both ways       |
| leaf          | change one attribute of one leaf                          | symmetric                        |
| composition   | change a non-body composition type (valid child count)    | symmetric                        |
| body          | change one body parameter to another grid value           | symmetric                        |
| swap          | swap two children (dashed edges follow their nodes)       | symmetric                        |
| add_dashed    | add a dashed edge source -> leaf                          | w_add / #addable vs w_remove / #edges |
| remove_dashed | remove a dashed edge                                      | the reverse of add_dashed        |
| wrap          | put a node under a new single-child composition           | paired with unwrap               |
| unwrap        | replace a single-child composition by its child           | paired with wrap                 |
| add_child     | insert a generated subtree as a new child                 | paired with remove_child         |
| remove_child  | delete a child subtree (no dashed edges inside)           | paired with add_child            |

Dashed edges follow their nodes when paths shift (wrap, unwrap, add/remove
child, swap). Moves that would orphan an edge are not offered, so every move
has an exact reverse.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from grace_lot.inference.grammar import Grammar
from grace_lot.lot_adapter.program import (LEAF, MIN_CHILDREN, OUTPUTS, P, Program, candidate_edges, nodes,
                                           replace, with_edges)

Proposal = Optional[Tuple[Program, float, float]]


def _touches(prog: Program, path: tuple) -> bool:
    """True if some dashed edge has an endpoint inside the subtree at ``path``."""
    return any(e[:len(path)] == path for edge in prog.dashed for e in edge)


def _free_sites(prog: Program) -> List[tuple]:
    return [path for path, _ in nodes(prog.tree) if not _touches(prog, path)]


def regenerate(prog: Program, g: Grammar, rng: random.Random, w: Dict[str, float]) -> Proposal:
    sites = _free_sites(prog)
    if not sites:
        return None
    path = rng.choice(sites)
    old = _get(prog.tree, path)
    new = g.sample(rng, depth=len(path))
    q = Program(replace(prog.tree, path, new), prog.dashed)
    fwd = -math.log(len(sites)) + g.log_prob(new, len(path))
    bwd = -math.log(len(_free_sites(q))) + g.log_prob(old, len(path))
    return q, fwd, bwd


def _get(tree, path):
    for i in path:
        tree = tree.children[i]
    return tree


def _alternative(rng: random.Random, domain, current):
    options = [v for v in domain if v != current]
    return rng.choice(options) if options else None


def _edit(prog: Program, path: tuple, node) -> Program:
    return Program(replace(prog.tree, path, node), prog.dashed)


def change_leaf(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    sites = [(path, n) for path, n in nodes(prog.tree) if n.kind == LEAF]
    if not sites:
        return None
    path, n = rng.choice(sites)
    attr = rng.randrange(3 if g.allow_dynamic else 2)
    value = _alternative(rng, [g.primitives, g.orientations, g.dynamics][attr], n.params[attr])
    if value is None:
        return None
    params = list(n.params)
    params[attr] = value
    return _edit(prog, path, n._replace(params=tuple(params))), 0.0, 0.0


def change_composition(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    sites = [(path, n) for path, n in nodes(prog.tree) if n.kind not in (LEAF, OUTPUTS, "out", "in")]
    if not sites:
        return None
    path, n = rng.choice(sites)
    kinds = [k for k in g.compositions if len(n.children) in g.child_counts(k)]
    kind = _alternative(rng, kinds, n.kind)
    if kind is None:
        return None
    return _edit(prog, path, n._replace(kind=kind)), 0.0, 0.0


def swap_siblings(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    sites = [(path, n) for path, n in nodes(prog.tree) if len(n.children) >= 2]
    if not sites:
        return None
    path, n = rng.choice(sites)
    i, j = rng.sample(range(len(n.children)), 2)
    kids = list(n.children)
    kids[i], kids[j] = kids[j], kids[i]
    d = len(path)

    def move(e: tuple) -> tuple:
        if e[:d] == path and len(e) > d and e[d] in (i, j):
            return e[:d] + (j if e[d] == i else i,) + e[d + 1:]
        return e

    edges = [(move(a), move(b)) for a, b in prog.dashed]
    return with_edges(replace(prog.tree, path, n._replace(children=tuple(kids))), edges), 0.0, 0.0


def _addable(prog: Program, g: Grammar) -> List[tuple]:
    present = set(prog.dashed)
    return [e for e in candidate_edges(prog.tree, g.dashed_targets) if e not in present]


def add_dashed(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    if len(prog.dashed) >= g.max_dashed:
        return None
    options = _addable(prog, g)
    if not options:
        return None
    q = with_edges(prog.tree, list(prog.dashed) + [rng.choice(options)])
    fwd = math.log(w["add_dashed"]) - math.log(len(options))
    bwd = math.log(w["remove_dashed"]) - math.log(len(q.dashed))
    return q, fwd, bwd


def remove_dashed(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    if not prog.dashed:
        return None
    edge = rng.choice(prog.dashed)
    q = with_edges(prog.tree, [e for e in prog.dashed if e != edge])
    fwd = math.log(w["remove_dashed"]) - math.log(len(prog.dashed))
    bwd = math.log(w["add_dashed"]) - math.log(len(_addable(q, g)))
    return q, fwd, bwd


# ---- local structural moves (paired) ------------------------------------------

def _remap(prog: Program, fn) -> list:
    return [(fn(a), fn(b)) for a, b in prog.dashed]


def _wrap_kinds(g: Grammar) -> list:
    return [k for k in g.compositions if MIN_CHILDREN[k] == 1]


def _unwrap_sites(prog: Program) -> List[tuple]:
    """Single-child compositions that no dashed edge starts or ends at."""
    ends = {e for edge in prog.dashed for e in edge}
    return [p for p, n in nodes(prog.tree) if n.kind != LEAF and len(n.children) == 1 and p not in ends]


def wrap(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    kinds = _wrap_kinds(g)
    if not kinds:
        return None
    sites = [p for p, _ in nodes(prog.tree)]
    path = rng.choice(sites)
    kind = rng.choice(kinds)
    old = _get(prog.tree, path)
    d = len(path)
    shift = lambda e: e[:d] + (0,) + e[d:] if e[:d] == path else e
    q = with_edges(replace(prog.tree, path, P(kind, (), (old,))), _remap(prog, shift))
    fwd = math.log(w["wrap"]) - math.log(len(sites)) - math.log(len(kinds))
    bwd = math.log(w["unwrap"]) - math.log(len(_unwrap_sites(q)))
    return q, fwd, bwd


def unwrap(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    sites = _unwrap_sites(prog)
    if not sites:
        return None
    path = rng.choice(sites)
    node = _get(prog.tree, path)
    if node.kind not in _wrap_kinds(g):
        return None                       # wrap could not have created it, so no reverse exists
    d = len(path)
    shift = lambda e: e[:d] + e[d + 1:] if e[:d] == path else e
    q = with_edges(replace(prog.tree, path, node.children[0]), _remap(prog, shift))
    fwd = math.log(w["unwrap"]) - math.log(len(sites))
    bwd = math.log(w["wrap"]) - math.log(sum(1 for _ in nodes(q.tree))) - math.log(len(_wrap_kinds(g)))
    return q, fwd, bwd


def _growable(prog: Program, g: Grammar) -> List[tuple]:
    return [p for p, n in nodes(prog.tree) if n.kind not in (LEAF, "out", "in") and len(n.children) < max(g.child_counts(n.kind))]


def _removable(prog: Program) -> List[tuple]:
    """(parent path, child index) pairs: the parent keeps its minimum, and no dashed edge touches the child subtree."""
    out = []
    for p, n in nodes(prog.tree):
        if n.kind not in (LEAF, "out", "in") and len(n.children) > MIN_CHILDREN[n.kind]:
            out += [(p, i) for i in range(len(n.children)) if not _touches(prog, p + (i,))]
    return out


def add_child(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    parents = _growable(prog, g)
    if not parents:
        return None
    path = rng.choice(parents)
    node = _get(prog.tree, path)
    i = rng.randrange(len(node.children) + 1)
    sub = g.sample(rng, depth=len(path) + 1)
    kids = list(node.children)
    kids.insert(i, sub)
    d = len(path)
    shift = lambda e: e[:d] + (e[d] + 1,) + e[d + 1:] if e[:d] == path and len(e) > d and e[d] >= i else e
    q = with_edges(replace(prog.tree, path, node._replace(children=tuple(kids))), _remap(prog, shift))
    fwd = (math.log(w["add_child"]) - math.log(len(parents)) - math.log(len(node.children) + 1)
           + g.log_prob(sub, len(path) + 1))
    bwd = math.log(w["remove_child"]) - math.log(len(_removable(q)))
    return q, fwd, bwd


def remove_child(prog: Program, g: Grammar, rng: random.Random, w) -> Proposal:
    options = _removable(prog)
    if not options:
        return None
    path, i = rng.choice(options)
    node = _get(prog.tree, path)
    if len(node.children) > max(g.child_counts(node.kind)):
        return None
    sub = node.children[i]
    kids = list(node.children)
    del kids[i]
    d = len(path)
    shift = lambda e: e[:d] + (e[d] - 1,) + e[d + 1:] if e[:d] == path and len(e) > d and e[d] > i else e
    q = with_edges(replace(prog.tree, path, node._replace(children=tuple(kids))), _remap(prog, shift))
    fwd = math.log(w["remove_child"]) - math.log(len(options))
    bwd = (math.log(w["add_child"]) - math.log(len(_growable(q, g))) - math.log(len(kids) + 1)
           + g.log_prob(sub, len(path) + 1))
    return q, fwd, bwd


MOVES = {"regenerate": regenerate, "leaf": change_leaf, "composition": change_composition,
         "swap": swap_siblings,
         "add_dashed": add_dashed, "remove_dashed": remove_dashed,
         "wrap": wrap, "unwrap": unwrap, "add_child": add_child, "remove_child": remove_child}
PAIRED = [("add_dashed", "remove_dashed"), ("wrap", "unwrap"), ("add_child", "remove_child")]


class ProposalMixture:
    """Pick a move type with fixed probabilities, then apply it."""

    def __init__(self, grammar: Grammar, weights: Dict[str, float]) -> None:
        self.grammar = grammar
        self.names = [k for k, v in weights.items() if v > 0]
        total = sum(weights[k] for k in self.names)
        self.weights = {k: weights[k] / total for k in self.names}
        for a, b in PAIRED:
            if (a in self.weights) != (b in self.weights):
                raise ValueError(f"{a} and {b} must both be enabled, or neither.")

    def __call__(self, prog: Program, rng: random.Random) -> Tuple[str, Proposal]:
        name = rng.choices(self.names, [self.weights[k] for k in self.names])[0]
        return name, MOVES[name](prog, self.grammar, rng, self.weights)
