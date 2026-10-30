"""A library of program pieces (DreamCoder-style): named subtrees with leaf holes and their own dashed edges, which
cost one choice to use (``inference.library``: a JSON file). Programs stay ordinary trees, so the engine, the
likelihoods and the analyses are unchanged; the library enters in two places only.

* The prior (``LibraryPrior``). A program's length is its shortest description: each node is written with the grammar
  (as ``SizePrior``'s), except where a piece matches, which is written as one reference (log of the library size, plus
  log 10 for choosing the library over the grammar) and its holes (one grammar leaf each); a piece's own dashed edges
  come with it. The shortest description is found exactly, by dynamic programming over the tree. The language's limits
  apply to the description: nodes (a piece counts as one, plus its holes), depth (a piece as one level) and dashed
  edges (those outside pieces). Nodes are costed without the depth-limit's forced leaves, so a plain program's length
  differs slightly from ``SizePrior``'s.
* The search (``lib_insert`` / ``lib_remove``, paired as add/remove dashed): insert replaces a subtree no dashed edge
  touches by a fresh instance of a random piece (holes drawn as grammar leaves) with its edges; remove replaces an
  instance carrying exactly its own edges by a grammar sample. Both carry exact Hastings terms.

A library file: ``{"pieces": [{"name": ..., "tree": "(serial_h (parallel ? ?) out)", "edges": ".0.1>.0.0 ...",
"source": ...}]}``, holes written ``?``, edge paths relative to the piece's root.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from embodied_lot import STATIC_ONLY
from grace_lot.inference.grammar import P_DYNAMIC, P_OUT, Grammar
from grace_lot.inference.prior import SizePrior, n_edge_slots
from grace_lot.lot_adapter.program import (LEAF, OUT, OUTPUTS, P, Program, _unfmt, edges_valid, is_valid,
                                           motion_units, nodes, parse, replace, size, windows_as_content)

HOLE = "?"
P_LIBRARY = 0.1                       # the code's (and the generator's) share for "a library piece here"


class Piece:
    def __init__(self, name: str, tree: P, edges: tuple, source: str = "") -> None:
        self.name, self.tree, self.edges, self.source = name, tree, tuple(sorted(edges)), source
        self.holes = sum(1 for _, n in nodes(tree) if n.kind == HOLE)
        self.size = size(tree)

    def __repr__(self) -> str:
        return f"Piece({self.name}: {piece_text(self.tree)})"


def piece_text(t: P) -> str:
    if t.kind == HOLE:
        return HOLE
    if t.kind in (LEAF, OUT) or not t.children:
        return str(t)
    return f"({t.kind} " + " ".join(piece_text(c) for c in t.children) + ")"


def parse_piece(tree: str, edges: str = "") -> Tuple[P, tuple]:
    t = parse(tree.replace(HOLE, "hole__/V"))

    def holes(q: P) -> P:
        if q.kind == LEAF and q.params[0] == "hole__":
            return P(HOLE)
        return q._replace(children=tuple(holes(c) for c in q.children))
    return holes(t), tuple(tuple(_unfmt(x) for x in e.split(">")) for e in edges.split())


class Library:
    def __init__(self, pieces: List[Piece]) -> None:
        self.pieces = pieces
        self.ref = math.log(max(1, len(pieces))) - math.log(P_LIBRARY)    # nats to say "piece k here"

    @classmethod
    def load(cls, path) -> "Library":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Piece(d["name"], *parse_piece(d["tree"], d.get("edges", "")), d.get("source", ""))
                    for d in data["pieces"]])


def match(piece: P, t: P) -> Optional[List[P]]:
    """The leaves filling ``piece``'s holes if ``t`` is an instance of it (holes take leaves only), else None."""
    if piece.kind == HOLE:
        return [t] if t.kind == LEAF else None
    if piece.kind != t.kind or piece.params != t.params or len(piece.children) != len(t.children):
        return None
    out = []
    for a, b in zip(piece.children, t.children):
        m = match(a, b)
        if m is None:
            return None
        out += m
    return out


def instantiate(piece: P, fills: List[P]) -> P:
    it = iter(fills)

    def fill(q: P) -> P:
        if q.kind == HOLE:
            return next(it)
        return q._replace(children=tuple(fill(c) for c in q.children))
    return fill(piece)


def inner_edges(prog: Program, path: tuple) -> tuple:
    """The dashed edges with both ends in the subtree at ``path``, relative to it."""
    n = len(path)
    return tuple(sorted((a[n:], b[n:]) for a, b in prog.dashed if a[:n] == path and b[:n] == path))


def touching(prog: Program, path: tuple) -> int:
    n = len(path)
    return sum(1 for a, b in prog.dashed if a[:n] == path or b[:n] == path)


def leaf_logp(g: Grammar, leaf: P) -> float:
    """Log-probability of a hole's leaf: drawn as the grammar draws a leaf (not a window)."""
    name, _, dyn = leaf.params
    if name not in g.primitives or (dyn and (not g.allow_dynamic or name in STATIC_ONLY)):
        return -math.inf
    lp = -math.log(len(g.primitives)) - math.log(2)
    if g.allow_dynamic and name not in STATIC_ONLY:
        lp += math.log(P_DYNAMIC if dyn else 1 - P_DYNAMIC)
    return lp


def sample_leaf(g: Grammar, rng: random.Random) -> P:
    dyn = g.allow_dynamic and rng.random() < P_DYNAMIC
    name = rng.choice(g.primitives)
    return P(LEAF, (name, rng.choice(g.orientations), dyn and name not in STATIC_ONLY))


def instances(prog: Program, lib: Library, exact: bool = True) -> Dict[tuple, List[Tuple[int, List[P]]]]:
    """Per path, the pieces the subtree there is an instance of, with its holes' leaves. ``exact``: the subtree's
    dashed edges are exactly the piece's own and no other edge touches it (a piece the search may remove); else the
    piece's own edges are among the program's (a piece the prior may read)."""
    out = {}
    for path, t in nodes(prog.tree):
        if t.kind in (LEAF, OUT, OUTPUTS):
            continue
        inner = inner_edges(prog, path)
        for k, pc in enumerate(lib.pieces):
            fills = match(pc.tree, t)
            if fills is None:
                continue
            if exact and (inner != pc.edges or touching(prog, path) != len(pc.edges)):
                continue
            if not exact and not set(pc.edges) <= set(inner):
                continue
            out.setdefault(path, []).append((k, fills))
    return out


class LibraryPrior(SizePrior):
    """``SizePrior`` with a library: see the module docstring."""

    def __init__(self, grammar: Grammar, lambda_size: float, library: Library, dashed_cost: float = 1.0) -> None:
        super().__init__(grammar, lambda_size, dashed_cost)
        self.library = library
        self._memo: Dict[Program, tuple] = {}

    def _node_cost(self, t: P) -> float:
        """Nats to write node ``t`` itself with the grammar (its children not included)."""
        g = self.grammar
        if t.kind == OUT:
            return -(math.log(g.p_leaf) + math.log(P_OUT)) if g.output_leaves else math.inf
        if t.kind == LEAF:
            lp = leaf_logp(g, t)
            return -(math.log(g.p_leaf) + (math.log(1 - P_OUT) if g.output_leaves else 0.0) + lp)
        if t.kind not in g.compositions or len(t.children) not in g.child_counts(t.kind) or t.params:
            return math.inf
        return -(math.log(1 - g.p_leaf) - math.log(len(g.compositions)) - math.log(len(g.child_counts(t.kind))))

    def describe(self, prog: Program) -> tuple:
        """(length of the tree part, description nodes, description depth, edges inside pieces, pieces used)."""
        if prog in self._memo:
            return self._memo[prog]
        lib, g = self.library, self.grammar
        found = instances(prog, lib, exact=False)

        def best(path: tuple, t: P) -> tuple:                 # (cost, nodes, depth, inner edges, pieces)
            kids = [best(path + (i,), c) for i, c in enumerate(t.children)]
            own = (self._node_cost(t) + sum(k[0] for k in kids), 1 + sum(k[1] for k in kids),
                   1 + max((k[2] for k in kids), default=0), sum(k[3] for k in kids), sum((k[4] for k in kids), []))
            options = [own]
            for k, fills in found.get(path, []):
                pc = lib.pieces[k]
                cost = lib.ref + sum(-leaf_logp(g, f) - math.log(g.p_leaf) for f in fills)
                options.append((cost, 1 + len(fills), 1 + (1 if fills else 0), len(pc.edges), [(path, pc.name)]))
            return min(options, key=lambda o: o[0])
        t = prog.tree
        if t.kind == OUTPUTS and len(t.children) not in g.child_counts(OUTPUTS):
            res = (math.inf, 0, 0, 0, [])                   # a forest has 2..max_outputs trees, as the grammar's
        elif t.kind == OUTPUTS:
            kids = [best((i,), c) for i, c in enumerate(t.children)]
            res = (math.log(max(1, g.max_outputs)) + sum(k[0] for k in kids), sum(k[1] for k in kids),
                   max(k[2] for k in kids), sum(k[3] for k in kids), sum((k[4] for k in kids), []))
        else:
            b = best((), t)
            res = ((math.log(g.max_outputs) if g.output_leaves else 0.0) + b[0], *b[1:])
        if len(self._memo) > 20000:
            self._memo.clear()
        self._memo[prog] = res
        return res

    def length(self, prog) -> float:
        prog = prog if isinstance(prog, Program) else Program(prog)
        tree_len, _, _, inner, _ = self.describe(prog)
        k = len(prog.dashed) - inner                        # edges written outside pieces
        edge_bits = (math.log(self.grammar.max_dashed + 1)
                     + math.log(math.comb(n_edge_slots(prog.tree, self.grammar.dashed_targets), k))) if k else 0.0
        return tree_len + edge_bits

    def __call__(self, prog) -> float:
        prog = prog if isinstance(prog, Program) else Program(prog)
        g, t = self.grammar, prog.tree
        tree_len, n_desc, d_desc, inner, _ = self.describe(prog)
        if (not math.isfinite(tree_len) or n_desc > g.max_nodes or d_desc > g.max_depth or not is_valid(t)
                or len(prog.dashed) - inner > g.max_dashed or not edges_valid(prog, g.dashed_targets)):
            return -math.inf
        if g.output_leaves and not 1 <= sum(n.kind == OUT for _, n in nodes(t)) <= g.max_outputs:
            return -math.inf
        if not windows_as_content(t):
            return -math.inf
        if g.max_motion_units is not None and motion_units(prog) > g.max_motion_units:
            return -math.inf
        return -self.lambda_size * self.length(prog)


# ---- search moves -------------------------------------------------------------------------------------------------

def _free(prog: Program) -> List[tuple]:
    return [p for p, n in nodes(prog.tree) if n.kind != OUTPUTS and touching(prog, p) == 0]


def _insert_logp(g: Grammar, lib: Library, subtree: P, edges: tuple) -> float:
    """Log-probability that an insert draws exactly ``subtree`` with ``edges`` (summed over the pieces that give it)."""
    terms = []
    for pc in lib.pieces:
        fills = match(pc.tree, subtree)
        if fills is not None and pc.edges == edges:
            terms.append(-math.log(len(lib.pieces)) + sum(leaf_logp(g, f) for f in fills))
    if not terms:
        return -math.inf
    m = max(terms)
    return m + math.log(sum(math.exp(x - m) for x in terms))


def _shift(edges: tuple, path: tuple) -> tuple:
    return tuple((path + a, path + b) for a, b in edges)


def lib_insert(prog: Program, g: Grammar, rng: random.Random, w):
    lib = getattr(g, "library", None)
    sites = _free(prog)
    if lib is None or not lib.pieces or not sites:
        return None
    path = rng.choice(sites)
    old = _get(prog.tree, path)
    pc = rng.choice(lib.pieces)
    new = instantiate(pc.tree, [sample_leaf(g, rng) for _ in range(pc.holes)])
    q = Program(replace(prog.tree, path, new), tuple(sorted(set(prog.dashed) | set(_shift(pc.edges, path)))))
    removable = instances(q, lib)
    if path not in removable:
        return None
    fwd = math.log(w["lib_insert"]) - math.log(len(sites)) + _insert_logp(g, lib, new, pc.edges)
    bwd = math.log(w["lib_remove"]) - math.log(len(removable)) + g.log_prob(old, len(path))
    return q, fwd, bwd


def lib_remove(prog: Program, g: Grammar, rng: random.Random, w):
    lib = getattr(g, "library", None)
    if lib is None:
        return None
    removable = instances(prog, lib)
    if not removable:
        return None
    path = rng.choice(sorted(removable))
    old = _get(prog.tree, path)
    own = inner_edges(prog, path)
    new = g.sample(rng, depth=len(path))
    q = Program(replace(prog.tree, path, new), tuple(e for e in prog.dashed if e not in set(_shift(own, path))))
    fwd = math.log(w["lib_remove"]) - math.log(len(removable)) + g.log_prob(new, len(path))
    bwd = math.log(w["lib_insert"]) - math.log(len(_free(q))) + _insert_logp(g, lib, old, own)
    return q, fwd, bwd


def _get(tree: P, path: tuple) -> P:
    for i in path:
        tree = tree.children[i]
    return tree
