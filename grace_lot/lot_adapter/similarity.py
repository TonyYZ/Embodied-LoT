"""Program-similarity measures, kept explicitly separate.

Every measure is ``(Program, Program) -> [0, 1]``. Tree measures use the
Zhang-Shasha ordered tree edit distance (unit costs). To that they add one
unit per dashed edge present in one program but not the other; an edge is
labelled by its endpoint labels and their relation (source above, below or
beside the target). The total is normalised as
``1 - d / (|T1| + |T2| + |E1| + |E2|)``. A *view* first transforms each program: it
relabels nodes, reorders children, or deletes leaves. The measures and
ablations differ only in their view.

Main measures:
  flat            weighted Jaccard of the leaf-label multisets (no structure)
  tree_unordered  hierarchy + leaf labels; composition type hidden; sibling order ignored
  tree_ordered    hierarchy + leaf labels; composition type hidden; sibling order kept
  full            hierarchy + leaf labels + composition type (with body parameters) + order
(the three tree measures also include dashed edges, unless ``dashed=False``)

"Order ignored" sorts siblings into a canonical order before the ordered
distance is taken. This is a standard, cheap approximation of unordered
tree edit distance (the exact version is NP-hard).

Posterior similarity:
  S(A, B) = E_{r_A ~ p(r|A), r_B ~ p(r|B)} [S_program(r_A, r_B)]
approximated with paired posterior samples.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from grace_lot.lot_adapter.program import LEAF, P, Program, as_program, get

Tree = Tuple[str, tuple]          # (label, children)


# ---- views -----------------------------------------------------------------

@dataclass(frozen=True)
class View:
    """How a program is seen by a similarity measure."""
    leaf_pattern: bool = True
    leaf_orientation: bool = True
    leaf_dynamic: bool = True
    composition: str = "full"          # "full" (with body params), "kind", "collapsed" (serials merged), "hidden"
    ordered: bool = True
    hierarchy: bool = True             # False: every leaf becomes a child of one root
    drop_leaves: Optional[str] = None  # None, "dynamic" or "static": delete that family of leaves
    dashed: bool = True                # include dashed edges

    def node_label(self, p: P) -> str:
        return self.leaf_label(p) if p.kind == LEAF else self.comp_label(p)

    def edges(self, prog: Program) -> Counter:
        """Multiset of dashed-edge labels: (source label, target label, relation)."""
        if not self.dashed:
            return Counter()
        out = Counter()
        for a, b in prog.dashed:
            src, dst = get(prog.tree, a), get(prog.tree, b)
            if not (self.keep(src) and self.keep(dst)):
                continue
            if not self.hierarchy:
                rel = "any"
            elif b[:len(a)] == a:
                rel = "down"
            elif a[:len(b)] == b:
                rel = "up"
            else:
                rel = "side"
            out[(self.node_label(src), self.node_label(dst), rel)] += 1
        return out

    def leaf_label(self, p: P) -> str:
        name, ori, dyn = p.params
        parts = [name if self.leaf_pattern else "obj"]
        if self.leaf_orientation:
            parts.append(ori)
        if self.leaf_dynamic:
            parts.append("dyn" if dyn else "static")
        return "/".join(parts)

    def comp_label(self, p: P) -> str:
        if self.composition == "hidden":
            return "comp"
        kind = "serial" if self.composition == "collapsed" and p.kind.startswith("serial") else p.kind
        if self.composition == "full" and p.kind == "body":
            kind += str(p.params)
        return kind

    def keep(self, p: P) -> bool:
        return not (p.kind == LEAF and self.drop_leaves == ("dynamic" if p.params[2] else "static"))

    def tree(self, p: P) -> Tree:
        if not self.hierarchy:
            return ("root", tuple(sorted((self.leaf_label(q), ()) for q in _leaves(p) if self.keep(q))))
        return self._tree(p)

    def _tree(self, p: P) -> Tree:
        if p.kind == LEAF:
            return (self.leaf_label(p), ())
        kids = [self._tree(c) for c in p.children if self.keep(c)]
        if not self.ordered:
            kids.sort(key=repr)
        return (self.comp_label(p), tuple(kids))


def _leaves(p: P) -> List[P]:
    return [p] if p.kind == LEAF else [q for c in p.children for q in _leaves(c)]


# ---- Zhang-Shasha tree edit distance --------------------------------------

def _postorder(t: Tree):
    labels, lmd = [], []

    def visit(node: Tree) -> int:
        first = None
        for child in node[1]:
            leftmost = visit(child)
            first = leftmost if first is None else first
        labels.append(node[0])
        lmd.append(len(labels) - 1 if first is None else first)
        return lmd[-1]

    visit(t)
    return labels, lmd


@lru_cache(maxsize=200_000)
def tree_edit_distance(a: Tree, b: Tree) -> int:
    la, ia = _postorder(a)
    lb, ib = _postorder(b)
    keyroots = lambda lmd: [i for i in range(len(lmd)) if all(lmd[j] != lmd[i] for j in range(i + 1, len(lmd)))]
    n, m = len(la), len(lb)
    td = [[0] * m for _ in range(n)]
    for i in keyroots(ia):
        for j in keyroots(ib):
            i0, j0 = ia[i], ib[j]
            fd = [[0] * (j - j0 + 2) for _ in range(i - i0 + 2)]
            for x in range(1, i - i0 + 2):
                fd[x][0] = x
            for y in range(1, j - j0 + 2):
                fd[0][y] = y
            for x in range(1, i - i0 + 2):
                for y in range(1, j - j0 + 2):
                    xi, yj = i0 + x - 1, j0 + y - 1
                    if ia[xi] == i0 and ib[yj] == j0:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[x - 1][y - 1] + (la[xi] != lb[yj]))
                        td[xi][yj] = fd[x][y]
                    else:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1,
                                       fd[ia[xi] - i0][ib[yj] - j0] + td[xi][yj])
    return td[n - 1][m - 1]


def _count(t: Tree) -> int:
    return 1 + sum(_count(c) for c in t[1])


# ---- measures --------------------------------------------------------------

Measure = Callable[[P, P], float]


def tree_measure(view: View) -> Measure:
    def measure(a, b) -> float:
        a, b = as_program(a), as_program(b)
        ta, tb = view.tree(a.tree), view.tree(b.tree)
        ea, eb = view.edges(a), view.edges(b)
        d = tree_edit_distance(ta, tb) + sum(((ea - eb) + (eb - ea)).values())
        return 1.0 - d / (_count(ta) + _count(tb) + sum(ea.values()) + sum(eb.values()))
    return measure


def flat_measure(view: View) -> Measure:
    def measure(a, b) -> float:
        a, b = as_program(a).tree, as_program(b).tree
        ca = Counter(view.leaf_label(q) for q in _leaves(a) if view.keep(q))
        cb = Counter(view.leaf_label(q) for q in _leaves(b) if view.keep(q))
        union = sum((ca | cb).values())
        return sum((ca & cb).values()) / union if union else 1.0
    return measure


MEASURES: Dict[str, Measure] = {
    "flat":           flat_measure(View()),
    "tree_unordered": tree_measure(View(composition="hidden", ordered=False)),
    "tree_ordered":   tree_measure(View(composition="hidden", ordered=True)),
    "full":           tree_measure(View()),
}

# Ablations of the full measure: remove one representational distinction at a time.
ABLATIONS: Dict[str, Measure] = {
    "full-no_dashed":       tree_measure(View(dashed=False)),
    "full-no_order":        tree_measure(View(ordered=False)),
    "full-no_hierarchy":    tree_measure(View(hierarchy=False)),
    "full-no_body_params":  tree_measure(View(composition="kind")),
    "full-collapse_serial": tree_measure(View(composition="collapsed")),
    "full-no_orientation":  tree_measure(View(leaf_orientation=False)),
    "full-no_dynamic_flag": tree_measure(View(leaf_dynamic=False)),
    "full-drop_dynamic":    tree_measure(View(drop_leaves="dynamic")),
    "full-drop_static":     tree_measure(View(drop_leaves="static")),
    "flat-patterns_only":   flat_measure(View(leaf_orientation=False, leaf_dynamic=False)),
}


# ---- posterior expectation -------------------------------------------------

def sample_pairs(post_a: Dict[P, float], post_b: Dict[P, float], n: int,
                 rng: random.Random) -> List[Tuple[P, P]]:
    pa, wa = zip(*post_a.items())
    pb, wb = zip(*post_b.items())
    return list(zip(rng.choices(pa, wa, k=n), rng.choices(pb, wb, k=n)))


def expected_similarity(measure: Measure, pairs: Sequence[Tuple[P, P]]) -> float:
    return sum(measure(a, b) for a, b in pairs) / len(pairs)
