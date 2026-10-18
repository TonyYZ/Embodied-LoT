"""The search grammar: which programs exist, and a generator with exact log-probabilities.

    R    -> Leaf                                    with prob p_leaf (1 at the depth limit)
          | K(R_1 .. R_k)                           K uniform over the allowed compositions,
                                                    k uniform over {MIN_CHILDREN[K] .. max_children}
    Leaf -> obj(pattern, ori, dynamic)              each attribute uniform; dynamic only if allowed
    body -> body(R_1 .. R_k)                        no parameters: the tree sizes and places it
            body(h, w, speed)(R_1 .. R_k)           when body_extents is non-empty (earlier runs)
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
from typing import Optional, Sequence

from grace_lot.lot_adapter.program import GROUPS, LEAF, MIN_CHILDREN, OUT, OUTPUTS, P

P_DYNAMIC = 0.25
P_OUT = 0.15                  # with output_leaves: the chance that a generated leaf is an output window


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
    output_leaves: bool = False      # outputs are ``out`` leaves (1..max_outputs, anywhere in the forest's trees)
    content_windows: bool = False    # with output_leaves: windows are never templates (enforced by the prior)
    grouping: bool = False           # grouping_nodes: field / body are unary units; template copies
    max_motion_units: Optional[int] = None   # probabilistic_motion: at most this many unknown-motion units

    @classmethod
    def from_config(cls, lot: dict) -> "Grammar":
        sized = lot.get("tree_sized_bodies", True)
        positions = [] if sized or lot.get("tree_placed_bodies", True) else lot["body_positions"]
        return cls(lot["primitives"], lot["compositions"], lot["allow_dynamic"],
                   positions, [] if sized else lot["body_extents"], [] if sized else lot["body_speeds"],
                   lot["max_depth"], lot["max_nodes"], lot["p_leaf"], lot["max_children"],
                   lot.get("max_dashed", 0), lot.get("max_outputs", 0), lot.get("dashed_targets", "leaves"),
                   bool(lot.get("max_outputs", 0)) and lot.get("output_leaves", True),
                   bool(lot.get("max_outputs", 0)) and lot.get("output_leaves", True) and lot.get("content_windows", True),
                   bool(lot.get("grouping_nodes", True)),
                   lot.get("max_motion_units") if lot.get("probabilistic_motion", True) else None)

    # -- attribute domains (also used by the symmetric moves) ---------

    @property
    def orientations(self):
        return ("V", "H")

    @property
    def dynamics(self):
        return (False, True) if self.allow_dynamic else (False,)

    @property
    def n_body_params(self) -> int:
        """0 when bodies are sized by the tree; 3 (h, w, speed) when only placed by it; 5 (y, x, h, w, speed)."""
        if not self.body_extents:
            return 0
        return 5 if self.body_positions else 3

    def body_grid(self, i: int) -> Sequence[float]:
        """Allowed values of body parameter i."""
        grids = [self.body_extents, self.body_extents, self.body_speeds]
        if self.body_positions:
            grids = [self.body_positions, self.body_positions] + grids
        return grids[i]

    def child_counts(self, kind: str) -> range:
        if kind == OUTPUTS:                         # with output_leaves a forest has 2+ trees; one tree is plain
            return range(2 if self.output_leaves else 1, max(1, self.max_outputs) + 1)
        if self.grouping and kind in GROUPS:          # grouping is unary: one unit over one child
            return range(1, 2)
        return range(MIN_CHILDREN[kind], max(MIN_CHILDREN[kind], self.max_children) + 1)

    def multi_output(self, depth: int) -> bool:
        """The root is a forest of 1..max_outputs trees (each its own space; with output_leaves, channels are windows)."""
        return self.max_outputs > 0 and depth == 0

    def leaf_forced(self, depth: int) -> bool:
        return depth >= self.max_depth - 1 or not self.compositions

    # -- generator -----------------------------------------------------

    def sample(self, rng: random.Random, depth: int = 0) -> P:
        if self.multi_output(depth) and self.output_leaves:   # K trees, K uniform; one tree is just the tree
            k = rng.randrange(1, self.max_outputs + 1)
            if k == 1:
                return self.sample(rng, 1)
            return P(OUTPUTS, (), tuple(self.sample(rng, 1) for _ in range(k)))
        if self.multi_output(depth):
            k = rng.choice(self.child_counts(OUTPUTS))
            return P(OUTPUTS, (), tuple(self.sample(rng, 1) for _ in range(k)))
        if self.leaf_forced(depth) or rng.random() < self.p_leaf:
            if self.output_leaves and rng.random() < P_OUT:
                return P(OUT)
            dyn = self.allow_dynamic and rng.random() < P_DYNAMIC
            return P(LEAF, (rng.choice(self.primitives), rng.choice(self.orientations), dyn))
        kind = rng.choice(self.compositions)
        k = rng.choice(self.child_counts(kind))
        params = tuple(rng.choice(self.body_grid(i)) for i in range(self.n_body_params)) if kind == "body" else ()
        return P(kind, params, tuple(self.sample(rng, depth + 1) for _ in range(k)))

    def log_prob(self, p: P, depth: int = 0) -> float:
        """Log-probability that ``sample(depth)`` returns ``p`` (-inf if it cannot)."""
        if self.multi_output(depth) and self.output_leaves:
            if p.kind != OUTPUTS:                    # a single tree, generated as a forest's tree
                return -math.log(self.max_outputs) + self.log_prob(p, 1)
            if len(p.children) not in self.child_counts(OUTPUTS):
                return -math.inf
            return -math.log(self.max_outputs) + sum(self.log_prob(c, 1) for c in p.children)
        if self.multi_output(depth) or p.kind == OUTPUTS:
            if not (self.multi_output(depth) and p.kind == OUTPUTS and len(p.children) in self.child_counts(OUTPUTS)):
                return -math.inf
            return -math.log(len(self.child_counts(OUTPUTS))) + sum(self.log_prob(c, 1) for c in p.children)
        forced = self.leaf_forced(depth)
        if p.kind == OUT:
            return (0.0 if forced else math.log(self.p_leaf)) + math.log(P_OUT) if self.output_leaves else -math.inf
        if p.kind == LEAF:
            name, ori, dyn = p.params
            if name not in self.primitives or (dyn and not self.allow_dynamic):
                return -math.inf
            lp = 0.0 if forced else math.log(self.p_leaf)
            if self.output_leaves:
                lp += math.log(1 - P_OUT)
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
