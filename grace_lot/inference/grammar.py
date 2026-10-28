"""The search grammar: which programs exist, and a generator with exact log-probabilities.

    R    -> Leaf                                    with prob p_leaf (1 at the depth limit)
          | K(R_1 .. R_k)                           K uniform over the allowed compositions,
                                                    k uniform over its child counts (``child_counts``)
    Leaf -> obj(pattern, ori, dynamic)              each attribute uniform; dynamic only if allowed
          | out                                     with ``max_outputs``: an output window (prob P_OUT)

With ``max_outputs`` > 0 the root is a forest of 1..max_outputs trees (each its own space; one tree is just the
tree) with 1..max_outputs ``out`` windows anywhere in them, and a window is only ever content (``content_windows``).
A field or body holds 1..max_children children, drawn together.

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
P_OUT = 0.15                  # the chance that a generated leaf is an output window (with max_outputs)


@dataclass
class Grammar:
    primitives: Sequence[str]
    compositions: Sequence[str]
    allow_dynamic: bool
    max_depth: int
    max_nodes: int
    p_leaf: float
    max_children: int
    max_dashed: int = 0
    max_outputs: int = 0          # > 0: 1..max_outputs ``out`` windows, in a forest of 1..max_outputs trees
    dashed_targets: str = "leaves"   # "leaves" or "any": which nodes dashed edges may target
    max_motion_units: Optional[int] = None   # at most this many unknown-motion units (``motion_units``)

    @classmethod
    def from_config(cls, lot: dict) -> "Grammar":
        return cls(lot["primitives"], lot["compositions"], lot["allow_dynamic"], lot["max_depth"], lot["max_nodes"],
                   lot["p_leaf"], lot["max_children"], lot.get("max_dashed", 0), lot.get("max_outputs", 0),
                   lot.get("dashed_targets", "leaves"), lot.get("max_motion_units"))

    @property
    def output_leaves(self) -> bool:
        """Outputs are ``out`` windows (the program has some)."""
        return self.max_outputs > 0

    # -- attribute domains (also used by the symmetric moves) ---------

    @property
    def orientations(self):
        return ("V", "H")

    @property
    def dynamics(self):
        return (False, True) if self.allow_dynamic else (False,)

    def child_counts(self, kind: str) -> range:
        if kind == OUTPUTS:                         # a forest has 2+ trees; one tree is plain
            return range(2, max(1, self.max_outputs) + 1)
        if kind in GROUPS:                          # a unit over one child or over several
            return range(1, self.max_children + 1)
        return range(MIN_CHILDREN[kind], max(MIN_CHILDREN[kind], self.max_children) + 1)

    def multi_output(self, depth: int) -> bool:
        """The root is a forest of 1..max_outputs trees."""
        return self.max_outputs > 0 and depth == 0

    def leaf_forced(self, depth: int) -> bool:
        return depth >= self.max_depth - 1 or not self.compositions

    # -- generator -----------------------------------------------------

    def sample(self, rng: random.Random, depth: int = 0) -> P:
        if self.multi_output(depth):                # K trees, K uniform; one tree is just the tree
            k = rng.randrange(1, self.max_outputs + 1)
            if k == 1:
                return self.sample(rng, 1)
            return P(OUTPUTS, (), tuple(self.sample(rng, 1) for _ in range(k)))
        if self.leaf_forced(depth) or rng.random() < self.p_leaf:
            if self.output_leaves and rng.random() < P_OUT:
                return P(OUT)
            dyn = self.allow_dynamic and rng.random() < P_DYNAMIC
            return P(LEAF, (rng.choice(self.primitives), rng.choice(self.orientations), dyn))
        kind = rng.choice(self.compositions)
        k = rng.choice(self.child_counts(kind))
        return P(kind, (), tuple(self.sample(rng, depth + 1) for _ in range(k)))

    def log_prob(self, p: P, depth: int = 0) -> float:
        """Log-probability that ``sample(depth)`` returns ``p`` (-inf if it cannot)."""
        if self.multi_output(depth):
            if p.kind != OUTPUTS:                    # a single tree, generated as a forest's tree
                return -math.log(self.max_outputs) + self.log_prob(p, 1)
            if len(p.children) not in self.child_counts(OUTPUTS):
                return -math.inf
            return -math.log(self.max_outputs) + sum(self.log_prob(c, 1) for c in p.children)
        if p.kind == OUTPUTS:
            return -math.inf
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
        if (forced or p.kind not in self.compositions or len(p.children) not in self.child_counts(p.kind)
                or p.params):
            return -math.inf
        lp = math.log(1 - self.p_leaf) - math.log(len(self.compositions)) - math.log(len(self.child_counts(p.kind)))
        return lp + sum(self.log_prob(c, depth + 1) for c in p.children)
