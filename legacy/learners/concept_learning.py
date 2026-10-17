"""Metropolis-Hastings learner over concept graphs.

What it can do
--------------
* It searches over **tree structure**: Serial, Parallel, Plan and Body nodes
  with ObjectNode leaves under one ``ParallelNode(I, ...)`` root.
* Its moves are: change a trigram, flip a trigram's orientation, flip its
  dynamic flag, add a leaf, remove a leaf, and change a structure node's type.
* It is scored on ``Observation``s (static) and ``Trajectory``s (temporal,
  replayed with ``step()``).
* It works with a complexity prior (size and depth) and a Gaussian likelihood.

What it cannot do yet
---------------------
* propose dashed edges, input/output windows, or a second (imagined) root
* learn spatial output fields (it learns scalar responses only)
* reuse fragments across tasks (``dreamcoder_lot`` does this, for flat
  scanner sets only)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from embodied_lot import (
    PATTERN_TO_NAME, TRIGRAM_PATTERNS, BodyNode, ConceptGraph, InputNode, Node,
    ObjectNode, Orientation, ParallelNode, PlanNode, SerialNode, StructureNode,
)
from learning_data import (
    Item, Observation, Trajectory, evaluate_on_dataset, predict, responses,
)

LearningData = List[Item]


# ============================================================
# Prior and likelihood
# ============================================================

class ConceptPrior:
    """Occam prior: ``log P(G) = -size_penalty * size(G) - depth_penalty * depth(G)``."""

    def __init__(self, size_penalty: float = 0.1, depth_penalty: float = 0.05) -> None:
        self.size_penalty = size_penalty
        self.depth_penalty = depth_penalty

    def log_prior(self, graph: ConceptGraph) -> float:
        return -self.size_penalty * graph.size() - self.depth_penalty * graph.depth()

    def __repr__(self) -> str:
        return f"ConceptPrior(size_penalty={self.size_penalty}, depth_penalty={self.depth_penalty})"


class ConceptLikelihood:
    """Gaussian noise: ``log P(r | a) = -(r - a)^2 / (2 noise^2)``, weighted per item."""

    def __init__(self, noise: float = 0.15) -> None:
        if noise <= 0:
            raise ValueError("noise must be positive.")
        self.noise = noise

    def _gauss(self, predicted: float, response: float) -> float:
        return -(response - predicted) ** 2 / (2 * self.noise ** 2)

    def log_likelihood(self, graph: ConceptGraph, data: Sequence[Item]) -> float:
        total = 0.0
        for item in data:
            pairs = zip(predict(graph, item), responses(item))
            total += item.weight * sum(self._gauss(p, r) for p, r in pairs)
        return total

    def __repr__(self) -> str:
        return f"ConceptLikelihood(noise={self.noise})"


# ============================================================
# Proposals
# ============================================================

_ALL_PATTERN_LIST = list(TRIGRAM_PATTERNS.values())
_STRUCTURE_TYPES = [SerialNode, ParallelNode, BodyNode, PlanNode]


def _random_object_node(node_id: str, rng: random.Random) -> ObjectNode:
    pattern = rng.choice(_ALL_PATTERN_LIST)
    orientation = rng.choice(list(Orientation))
    is_dynamic = rng.random() < 0.3
    return ObjectNode(node_id, pattern, orientation, is_dynamic=is_dynamic,
                      name=PATTERN_TO_NAME[tuple(pattern)])


class MutationProposal:
    """Returns a mutated deep copy of the graph.

    Every move is symmetric, so the acceptance test reduces to the posterior
    ratio. If the chosen move cannot be applied, ``change_trigram`` is used
    instead.
    """

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "change_trigram":   0.30,
        "flip_orientation": 0.10,
        "flip_dynamic":     0.10,
        "add_leaf":         0.15,
        "remove_leaf":      0.15,
        "change_structure": 0.20,
    }

    def __init__(self, operation_weights: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None) -> None:
        ops, probs = zip(*(operation_weights or self._DEFAULT_WEIGHTS).items())
        self._ops = list(ops)
        self._probs = [p / sum(probs) for p in probs]
        self._rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"proposed_{self._counter}"

    def propose(self, graph: ConceptGraph) -> ConceptGraph:
        candidate = copy.deepcopy(graph)
        operation = self._rng.choices(self._ops, weights=self._probs, k=1)[0]
        if not getattr(self, f"_op_{operation}")(candidate):
            self._op_change_trigram(candidate)
        return candidate

    def _random_leaf(self, graph: ConceptGraph) -> Optional[ObjectNode]:
        leaves = graph.leaf_nodes()
        return self._rng.choice(leaves) if leaves else None

    def _op_change_trigram(self, graph: ConceptGraph) -> bool:
        node = self._random_leaf(graph)
        if node is None:
            return False
        node.pattern = tuple(self._rng.choice(_ALL_PATTERN_LIST))
        node.name = PATTERN_TO_NAME[node.pattern]
        return True

    def _op_flip_orientation(self, graph: ConceptGraph) -> bool:
        node = self._random_leaf(graph)
        if node is None:
            return False
        node.orientation = (Orientation.HORIZONTAL if node.orientation == Orientation.VERTICAL
                            else Orientation.VERTICAL)
        return True

    def _op_flip_dynamic(self, graph: ConceptGraph) -> bool:
        node = self._random_leaf(graph)
        if node is None:
            return False
        node.is_dynamic = not node.is_dynamic
        return True

    def _op_add_leaf(self, graph: ConceptGraph) -> bool:
        parents = [n for n in graph.nodes if isinstance(n, StructureNode)]
        if not parents:
            return False
        parent = self._rng.choice(parents)
        leaf = _random_object_node(self._next_id(), self._rng)
        graph.add_node(leaf)
        graph.add_solid_edge(parent, leaf)
        return True

    def _op_remove_leaf(self, graph: ConceptGraph) -> bool:
        """Remove a leaf whose parent keeps at least one other child."""
        removable = [
            leaf for leaf in graph.leaf_nodes()
            if any(leaf in p.children and len(p.children) >= 2
                   for p in graph.nodes if isinstance(p, StructureNode))
        ]
        if not removable:
            return False
        graph.remove_node(self._rng.choice(removable))
        return True

    def _op_change_structure(self, graph: ConceptGraph) -> bool:
        """Change one structure node's type. A ParallelNode root is never changed, because it is the field hub."""
        candidates = [n for n in graph.nodes if isinstance(n, StructureNode)
                      and not (n is graph.root and isinstance(n, ParallelNode))]
        if not candidates:
            return False
        old = self._rng.choice(candidates)
        new_cls = self._rng.choice([c for c in _STRUCTURE_TYPES if not isinstance(old, c)])
        graph.replace_node(old, new_cls(old.node_id))
        return True


# ============================================================
# Sampler
# ============================================================

@dataclass
class MCMCTrace:
    scores: List[float] = field(default_factory=list)   # log-posterior after each acceptance
    best_score: float = -np.inf
    best_iteration: int = 0
    n_accepted: int = 0
    n_total: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(self.n_total, 1)


class ConceptLearner:
    """Metropolis-Hastings search for ``argmax log P(data | G) + log P(G)``."""

    def __init__(self, observations: Sequence[Item], n_iterations: int = 1000,
                 prior: Optional[ConceptPrior] = None,
                 likelihood: Optional[ConceptLikelihood] = None,
                 proposal: Optional[MutationProposal] = None,
                 initial_graph: Optional[ConceptGraph] = None,
                 random_seed: Optional[int] = None) -> None:
        self.observations = observations
        self.n_iterations = n_iterations
        self.prior = prior or ConceptPrior()
        self.likelihood = likelihood or ConceptLikelihood()
        self.proposal = proposal or MutationProposal(seed=random_seed)
        self._initial_graph = initial_graph
        self._np_rng = np.random.default_rng(random_seed)

    def log_posterior(self, graph: ConceptGraph) -> float:
        return self.likelihood.log_likelihood(graph, self.observations) + self.prior.log_prior(graph)

    @staticmethod
    def _make_initial_graph() -> ConceptGraph:
        """``ParallelNode(I, neutral)``: the smallest graph that can see a field."""
        graph = ConceptGraph(name="initial")
        hub, inp = ParallelNode("hub"), InputNode("I", modality="retina")
        leaf = ObjectNode("root", TRIGRAM_PATTERNS["neutral"], Orientation.VERTICAL, name="neutral")
        graph.add_nodes([hub, inp, leaf])
        graph.add_solid_edge(hub, inp)
        graph.add_solid_edge(hub, leaf)
        graph.root = hub
        return graph

    def run(self, verbose: bool = False) -> Tuple[ConceptGraph, MCMCTrace]:
        """Run the sampler and return the best graph seen and the trace."""
        current = self._initial_graph if self._initial_graph is not None else self._make_initial_graph()
        current_score = self.log_posterior(current)
        best_graph, best_score = copy.deepcopy(current), current_score
        trace = MCMCTrace()

        for iteration in range(self.n_iterations):
            candidate = self.proposal.propose(current)
            candidate_score = self.log_posterior(candidate)
            log_accept = min(0.0, candidate_score - current_score)
            trace.n_total += 1

            if np.log(self._np_rng.random() + 1e-300) < log_accept:
                current, current_score = candidate, candidate_score
                trace.n_accepted += 1
                trace.scores.append(current_score)
                if current_score > best_score:
                    best_graph, best_score = copy.deepcopy(current), current_score
                    trace.best_score, trace.best_iteration = best_score, iteration

            if verbose and iteration % 100 == 0:
                print(f"  Iteration {iteration:>5d} / {self.n_iterations}  "
                      f"score={current_score:+.3f}  best={best_score:+.3f}  "
                      f"accept_rate={trace.acceptance_rate:.2f}  graph_size={current.size()}")
        if verbose:
            print(f"\n  Run complete.  Best score: {trace.best_score:+.3f}  "
                  f"(iteration {trace.best_iteration})  Acceptance rate: {trace.acceptance_rate:.2f}")
        return best_graph, trace


# ============================================================
# Structural similarity
# ============================================================

def tree_edit_similarity(g1: ConceptGraph, g2: ConceptGraph) -> float:
    """1 - normalised Levenshtein distance between the pre-order node-type sequences of the two roots.

    This is a cheap approximation of tree edit distance.
    """
    seq1 = _flatten_structure(g1.root)
    seq2 = _flatten_structure(g2.root)
    return 1.0 - _levenshtein(seq1, seq2) / max(len(seq1), len(seq2), 1)


def _flatten_structure(node: Optional[Node]) -> List[str]:
    if node is None:
        return []
    return [node.node_type] + [t for c in node.children for t in _flatten_structure(c)]


def _levenshtein(s1: List[str], s2: List[str]) -> int:
    dp = list(range(len(s2) + 1))
    for i in range(1, len(s1) + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, len(s2) + 1):
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + (s1[i - 1] != s2[j - 1]))
    return dp[-1]


__all__ = [
    "ConceptLearner", "ConceptLikelihood", "ConceptPrior", "MCMCTrace", "MutationProposal",
    "Observation", "Trajectory", "evaluate_on_dataset", "tree_edit_similarity",
]
