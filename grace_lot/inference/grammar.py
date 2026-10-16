"""The search grammar: which programs exist, and a generator with exact log-probabilities.

    R    -> Leaf                                    with prob p_leaf (1 at the depth limit)
          | K(R_1 .. R_k)                           K uniform over the allowed compositions,
                                                    k uniform over {MIN_CHILDREN[K] .. max_children}
    Leaf -> obj(pattern, ori, dynamic)              each attribute uniform; dynamic only if allowed
    body -> body(h, w, speed)(R_1 .. R_k)           each parameter uniform over its grid
            body(y, x, h, w, speed)(R_1 .. R_k)     when body_positions is non-empty (earlier runs:
                                                    bodies placed by hand rather than by the tree)

The generator is the proposal for subtree regeneration. Its log-probability
gives the exact Hastings correction. It is *not* the prior (see prior.py).
Grammar ablations (no hierarchy, a reduced primitive set, collapsed
compositions) are just different settings of this class.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from grace_lot.lot_adapter.program import LEAF, MIN_CHILDREN, OUTPUTS, P

P_DYNAMIC = 0.25


@dataclass
class Grammar:
    primitives: Sequence[str]
    compositions: Sequence[str]
    allow_dynamic: bool
    body_positions: Sequence[float]
    body_extents: Sequence[float]
    body_speeds: Sequence[float]
    max_depth: int
    max_nodes: int
    p_leaf: float
    max_children: int
    max_dashed: int = 0
    max_outputs: int = 0          # > 0: the root is an ``outputs`` container with 1..max_outputs anonymous channels
    dashed_targets: str = "leaves"   # "leaves" or "any": which nodes dashed edges may target

    @classmethod
    def from_config(cls, lot: dict) -> "Grammar":
        positions = [] if lot.get("tree_placed_bodies", True) else lot["body_positions"]
        return cls(lot["primitives"], lot["compositions"], lot["allow_dynamic"],
                   positions, lot["body_extents"], lot["body_speeds"],
                   lot["max_depth"], lot["max_nodes"], lot["p_leaf"], lot["max_children"],
                   lot.get("max_dashed", 0), lot.get("max_outputs", 0), lot.get("dashed_targets", "leaves"))

    # -- attribute domains (also used by the symmetric moves) ---------

    @property
    def orientations(self):
        return ("V", "H")

    @property
    def dynamics(self):
        return (False, True) if self.allow_dynamic else (False,)

    @property
    def n_body_params(self) -> int:
        """3 (h, w, speed) when bodies are placed by the tree; 5 (y, x, h, w, speed) otherwise."""
        return 5 if self.body_positions else 3

    def body_grid(self, i: int) -> Sequence[float]:
        """Allowed values of body parameter i."""
        grids = [self.body_extents, self.body_extents, self.body_speeds]
        if self.body_positions:
            grids = [self.body_positions, self.body_positions] + grids
        return grids[i]

    def child_counts(self, kind: str) -> range:
        if kind == OUTPUTS:
            return range(1, max(1, self.max_outputs) + 1)
        return range(MIN_CHILDREN[kind], max(MIN_CHILDREN[kind], self.max_children) + 1)

    def multi_output(self, depth: int) -> bool:
        return self.max_outputs > 0 and depth == 0

    def leaf_forced(self, depth: int) -> bool:
        return depth >= self.max_depth - 1 or not self.compositions

    # -- generator -----------------------------------------------------

    def sample(self, rng: random.Random, depth: int = 0) -> P:
        if self.multi_output(depth):
            k = rng.choice(self.child_counts(OUTPUTS))
            return P(OUTPUTS, (), tuple(self.sample(rng, 1) for _ in range(k)))
        if self.leaf_forced(depth) or rng.random() < self.p_leaf:
            dyn = self.allow_dynamic and rng.random() < P_DYNAMIC
            return P(LEAF, (rng.choice(self.primitives), rng.choice(self.orientations), dyn))
        kind = rng.choice(self.compositions)
        k = rng.choice(self.child_counts(kind))
        params = tuple(rng.choice(self.body_grid(i)) for i in range(self.n_body_params)) if kind == "body" else ()
        return P(kind, params, tuple(self.sample(rng, depth + 1) for _ in range(k)))

    def log_prob(self, p: P, depth: int = 0) -> float:
        """Log-probability that ``sample(depth)`` returns ``p`` (-inf if it cannot)."""
        if self.multi_output(depth) or p.kind == OUTPUTS:
            if not (self.multi_output(depth) and p.kind == OUTPUTS and len(p.children) in self.child_counts(OUTPUTS)):
                return -math.inf
            return -math.log(len(self.child_counts(OUTPUTS))) + sum(self.log_prob(c, 1) for c in p.children)
        forced = self.leaf_forced(depth)
        if p.kind == LEAF:
            name, ori, dyn = p.params
            if name not in self.primitives or (dyn and not self.allow_dynamic):
                return -math.inf
            lp = 0.0 if forced else math.log(self.p_leaf)
            lp -= math.log(len(self.primitives)) + math.log(2)
            if self.allow_dynamic:
                lp += math.log(P_DYNAMIC if dyn else 1 - P_DYNAMIC)
            return lp
        if forced or p.kind not in self.compositions or len(p.children) not in self.child_counts(p.kind):
            return -math.inf
        lp = math.log(1 - self.p_leaf) - math.log(len(self.compositions)) - math.log(len(self.child_counts(p.kind)))
        if p.kind == "body":
            if len(p.params) != self.n_body_params:
                return -math.inf
            for i, v in enumerate(p.params):
                if v not in self.body_grid(i):
                    return -math.inf
                lp -= math.log(len(self.body_grid(i)))
        return lp + sum(self.log_prob(c, depth + 1) for c in p.children)
