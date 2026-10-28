"""Simplicity priors: ``log p(r) = -lambda * L(r) + C``, restricted to the grammar's support.

Two choices of program length L, set by ``prior`` in the config:

``description_length`` (default)
    L(r) is the number of nats needed to write r down with the grammar:
    ``-log P_gen(tree)`` (every choice costs the log of its number of
    options), plus the cost of the dashed edges. That edge cost is choosing
    how many (uniform over 0..max_dashed) and then which ones (uniform over
    all subsets of that size). At ``lambda = 1`` this is a proper prior that
    favours small programs; ``lambda > 1`` favours them more strongly.

``node_count``
    L(r) = nodes + ``dashed_cost`` per edge.
    Warning: the number of programs grows by a factor of 16-243 with each
    added node, so a small lambda makes this prior favour the *largest*
    programs (see diagnostics.family_coverage). Kept for comparison only.

Programs outside the grammar, or beyond its depth, size and dashed-edge
limits, get -inf. So do programs with a window leaf (``out``/``in``) as a template's template (windows are only
ever content), with more than ``max_outputs`` windows or none (when there are outputs), or with more than
``max_motion_units`` unknown-motion units.
"""

from __future__ import annotations

import math

from grace_lot.inference.grammar import Grammar
from grace_lot.lot_adapter.program import (as_program, candidate_edges, depth, edges_valid, is_valid,
                                           motion_units, nodes, size, windows_as_content)


def n_edge_slots(tree, targets: str = "leaves") -> int:
    """How many dashed edges are possible on ``tree`` (see ``program.candidate_edges``)."""
    return len(candidate_edges(tree, targets))


class SizePrior:
    def __init__(self, grammar: Grammar, lambda_size: float, dashed_cost: float = 1.0,
                 kind: str = "description_length") -> None:
        if kind not in ("description_length", "node_count"):
            raise ValueError(f"unknown prior {kind!r}")
        self.grammar = grammar
        self.lambda_size = lambda_size
        self.dashed_cost = dashed_cost
        self.kind = kind

    @classmethod
    def from_config(cls, grammar: Grammar, inf: dict) -> "SizePrior":
        return cls(grammar, inf["lambda_size"], inf.get("dashed_cost", 1.0), inf.get("prior", "description_length"))

    def length(self, prog) -> float:
        prog = as_program(prog)
        t, k = prog.tree, len(prog.dashed)
        if self.kind == "node_count":
            return size(t) + self.dashed_cost * k
        edge_bits = (math.log(self.grammar.max_dashed + 1) + math.log(math.comb(n_edge_slots(t, self.grammar.dashed_targets), k))
                     if self.grammar.max_dashed else 0.0)
        return -self.grammar.log_prob(t) + edge_bits

    def __call__(self, prog) -> float:
        prog = as_program(prog)
        g, t = self.grammar, prog.tree
        forest = 1 if g.output_leaves and t.kind == "outputs" else 0   # a forest's trees are not under a node
        if (size(t) - forest > g.max_nodes or depth(t) - forest > g.max_depth or not is_valid(t)
                or len(prog.dashed) > g.max_dashed or not edges_valid(prog, g.dashed_targets)
                or g.log_prob(t) == -math.inf):
            return -math.inf
        if g.output_leaves and not 1 <= sum(n.kind == "out" for _, n in nodes(t)) <= g.max_outputs:
            return -math.inf                                # 1..max_outputs output windows
        if not windows_as_content(t):
            return -math.inf                                # windows are content, never templates
        if g.max_motion_units is not None and motion_units(prog) > g.max_motion_units:
            return -math.inf                                # unknown motion: a bounded number of latent units
        return -self.lambda_size * self.length(prog)
