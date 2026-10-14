"""
DreamCoder-style concept learning for Embodied-LoT networks.

This module implements a small wake/compress learner for ConceptGraph
programs.  It is not a full DreamCoder clone, but it keeps the core loop:

1. Wake: enumerate graph programs from a library and fit each task.
2. Compress: find fragments repeatedly used by good programs and add them
   back to the library as cheaper reusable chunks.
3. Repeat: search again with the expanded library.

The graph language is intentionally conservative:

    ParallelNode(root)
      InputNode("I")
      ObjectNode(scanner_0, ...)
      ObjectNode(scanner_1, ...)

Multiple scanners compose through the existing ParallelNode semantics.  No
extra weighted-average logic is added here.

What it can do
--------------
* It enumerates flat scanner sets: ``ParallelNode(I, scanner, ...)``.
* It invents reusable two-scanner chunks shared across tasks (compression).
* It is scored on the same ``Observation``/``Trajectory`` data as the MCMC
  learner (see ``learning_data``), using weighted MSE plus library cost.

What it cannot do yet
---------------------
* propose Serial, Plan or Body structure, dashed edges, or a second
  (imagined) root (``concept_learning`` searches tree structure)
* learn dynamic scanners
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from embodied_lot import (
    TRIGRAM_PATTERNS,
    ConceptGraph,
    InputNode,
    ObjectNode,
    Orientation,
    ParallelNode,
)
from learning_data import (                                    # noqa: F401  (re-exported)
    Item, Observation, Trajectory, frame_scores, make_trigram_field, targets_and_weights,
    weighted_mse,
)


Pattern = Tuple[float, ...]
ScannerKey = Tuple[str, str]


class Example(Observation):
    """An ``Observation`` whose response is called ``target``."""

    def __init__(self, field, target: float, label: str = "", weight: float = 1.0) -> None:
        super().__init__(field, target, weight, label)

    @property
    def target(self) -> float:
        return self.response


@dataclass
class DreamTask:
    """A small concept-learning task solved by enumerating graph programs."""

    name: str
    examples: List[Item]          # Observations/Examples and/or Trajectories


@dataclass(frozen=True)
class ScannerSpec:
    """A reusable ObjectNode scanner specification."""

    name: str
    pattern: Pattern
    orientation: Orientation = Orientation.VERTICAL

    @property
    def key(self) -> ScannerKey:
        axis = "V" if self.orientation == Orientation.VERTICAL else "H"
        return (self.name, axis)


@dataclass
class LibraryFragment:
    """
    A library expression that expands into one or more scanner specs.

    Base fragments contain one scanner.  Compression adds composite fragments
    containing two or more scanners with a lower learned cost.
    """

    name: str
    scanners: Tuple[ScannerSpec, ...]
    cost: float = 1.0
    invented: bool = False
    uses: int = 0

    @property
    def arity(self) -> int:
        return len(self.scanners)


@dataclass
class CandidateProgram:
    """A concrete graph program and its task score."""

    name: str
    fragments: Tuple[str, ...]
    scanner_keys: Tuple[ScannerKey, ...]
    graph: ConceptGraph
    predictions: List[float]
    mse: float
    complexity: float
    objective: float


@dataclass
class WakeRound:
    """One wake/compress iteration summary."""

    iteration: int
    solutions: Dict[str, CandidateProgram]
    invented_fragments: List[str] = field(default_factory=list)


def default_library(
    *,
    include_neutral: bool = False,
    include_compound_trigrams: bool = True,
) -> List[LibraryFragment]:
    """
    Build the initial scanner library from the current trigram vocabulary.

    By default this includes the named trigram patterns except neutral and
    void.  Both vertical and horizontal orientations are included.
    """

    skip = {"void"}
    if not include_neutral:
        skip.add("neutral")
    if not include_compound_trigrams:
        skip.update({"lower", "upper", "periphery", "universal"})

    fragments: List[LibraryFragment] = []
    seen: set[Tuple[str, str]] = set()
    for name, pattern in TRIGRAM_PATTERNS.items():
        if name in skip:
            continue
        for orientation in (Orientation.VERTICAL, Orientation.HORIZONTAL):
            spec = ScannerSpec(name, tuple(pattern), orientation)
            if spec.key in seen:
                continue
            seen.add(spec.key)
            axis = "v" if orientation == Orientation.VERTICAL else "h"
            fragments.append(
                LibraryFragment(f"{name}_{axis}", (spec,), cost=1.0)
            )
    return fragments


def build_scanner_graph(
    specs: Sequence[ScannerSpec],
    *,
    name: str = "candidate",
) -> ConceptGraph:
    """Construct a ConceptGraph from scanner specs."""

    graph = ConceptGraph(name)
    root = ParallelNode("root")
    input_node = InputNode("I", modality="retina")
    graph.add_nodes([root, input_node])
    graph.add_solid_edge(root, input_node)

    used_ids: set[str] = {"root", "I"}
    for index, spec in enumerate(specs):
        base_id = f"{spec.name}_{'v' if spec.orientation == Orientation.VERTICAL else 'h'}"
        node_id = base_id
        suffix = 1
        while node_id in used_ids:
            suffix += 1
            node_id = f"{base_id}_{suffix}"
        used_ids.add(node_id)
        node = ObjectNode(
            node_id,
            spec.pattern,
            spec.orientation,
            name=spec.name,
        )
        graph.add_node(node)
        graph.add_solid_edge(root, node)

    graph.root = root
    return graph


def evaluate_program(graph: ConceptGraph, examples: Sequence[Item]) -> List[float]:
    """Predictions for every frame of every example."""
    return frame_scores(graph, examples)[0]


def program_error(predictions: Sequence[float], examples: Sequence[Item]) -> float:
    """Weighted mean squared error over frames."""
    return weighted_mse(predictions, *targets_and_weights(examples))


class DreamCoderLearner:
    """
    Small DreamCoder-style learner for Embodied-LoT scanner networks.

    Parameters
    ----------
    library:
        Initial library.  If omitted, `default_library()` is used.
    max_fragments:
        Maximum number of library expressions in an enumerated program.
    max_scanners:
        Maximum expanded scanner nodes in an enumerated graph.
    complexity_weight:
        Penalty multiplier for library expression cost.
    compression_min_uses:
        A scanner pair must appear in this many best programs before being
        invented as a composite fragment.
    compression_factor:
        Invented fragment cost is `factor * sum(part_costs)`.
    """

    def __init__(
        self,
        library: Optional[Sequence[LibraryFragment]] = None,
        *,
        max_fragments: int = 2,
        max_scanners: int = 3,
        complexity_weight: float = 0.015,
        compression_min_uses: int = 2,
        compression_factor: float = 0.65,
    ) -> None:
        self.library: List[LibraryFragment] = list(library or default_library())
        self.max_fragments = max_fragments
        self.max_scanners = max_scanners
        self.complexity_weight = complexity_weight
        self.compression_min_uses = compression_min_uses
        self.compression_factor = compression_factor
        self.history: List[WakeRound] = []

    def enumerate_programs(self) -> Iterable[Tuple[Tuple[LibraryFragment, ...], Tuple[ScannerSpec, ...]]]:
        """Yield library-fragment combinations and their expanded scanners."""

        seen: set[Tuple[ScannerKey, ...]] = set()
        ordered_library = sorted(self.library, key=lambda frag: (frag.cost, frag.name))
        for count in range(1, self.max_fragments + 1):
            for fragments in itertools.combinations_with_replacement(ordered_library, count):
                scanners = tuple(
                    scanner
                    for fragment in fragments
                    for scanner in fragment.scanners
                )
                if len(scanners) > self.max_scanners:
                    continue
                keys = tuple(scanner.key for scanner in scanners)
                canonical = tuple(sorted(keys))
                if canonical in seen:
                    continue
                seen.add(canonical)
                yield fragments, scanners

    def fit_task(self, task: DreamTask) -> CandidateProgram:
        """Return the lowest-objective program for one task."""

        best: Optional[CandidateProgram] = None
        for fragments, scanners in self.enumerate_programs():
            graph = build_scanner_graph(scanners, name=task.name)
            predictions, targets, weights = frame_scores(graph, task.examples)
            mse = weighted_mse(predictions, targets, weights)
            complexity = sum(fragment.cost for fragment in fragments)
            objective = mse + self.complexity_weight * complexity
            candidate = CandidateProgram(
                name=" + ".join(fragment.name for fragment in fragments),
                fragments=tuple(fragment.name for fragment in fragments),
                scanner_keys=tuple(scanner.key for scanner in scanners),
                graph=graph,
                predictions=predictions,
                mse=mse,
                complexity=complexity,
                objective=objective,
            )
            if best is None or candidate.objective < best.objective:
                best = candidate
        if best is None:
            raise RuntimeError("No candidate programs were enumerated.")
        return best

    def wake(self, tasks: Sequence[DreamTask]) -> Dict[str, CandidateProgram]:
        """Fit every task with the current library."""

        return {task.name: self.fit_task(task) for task in tasks}

    def compress(self, solutions: Dict[str, CandidateProgram]) -> List[str]:
        """
        Invent cheap composite fragments from scanner pairs reused by solutions.
        Also slightly lowers the cost of fragments that were actually used.
        """

        fragment_by_name = {fragment.name: fragment for fragment in self.library}
        for solution in solutions.values():
            for name in solution.fragments:
                fragment = fragment_by_name.get(name)
                if fragment is None:
                    continue
                fragment.uses += 1
                fragment.cost = max(0.25, fragment.cost * 0.96)

        pair_counts: Dict[Tuple[ScannerKey, ScannerKey], int] = {}
        spec_by_key: Dict[ScannerKey, ScannerSpec] = {}
        for solution in solutions.values():
            unique_keys = sorted(set(solution.scanner_keys))
            for key in unique_keys:
                for fragment in self.library:
                    for spec in fragment.scanners:
                        if spec.key == key:
                            spec_by_key[key] = spec
            for pair in itertools.combinations(unique_keys, 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        existing = {
            tuple(sorted(scanner.key for scanner in fragment.scanners))
            for fragment in self.library
        }
        invented: List[str] = []
        for pair, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
            if count < self.compression_min_uses or pair in existing:
                continue
            specs = tuple(spec_by_key[key] for key in pair if key in spec_by_key)
            if len(specs) != 2:
                continue
            part_cost = self._scanner_cost(specs[0]) + self._scanner_cost(specs[1])
            name = "chunk_" + "__".join(f"{key[0]}_{key[1].lower()}" for key in pair)
            self.library.append(
                LibraryFragment(
                    name=name,
                    scanners=specs,
                    cost=max(0.3, part_cost * self.compression_factor),
                    invented=True,
                    uses=count,
                )
            )
            invented.append(name)
        return invented

    def _scanner_cost(self, spec: ScannerSpec) -> float:
        best = 1.0
        for fragment in self.library:
            if len(fragment.scanners) == 1 and fragment.scanners[0].key == spec.key:
                best = min(best, fragment.cost)
        return best

    def run(
        self,
        tasks: Sequence[DreamTask],
        *,
        iterations: int = 2,
    ) -> Dict[str, CandidateProgram]:
        """Run wake/compress iterations and return final task solutions."""

        solutions: Dict[str, CandidateProgram] = {}
        for iteration in range(iterations):
            solutions = self.wake(tasks)
            invented = self.compress(solutions)
            self.history.append(WakeRound(iteration, solutions, invented))
        solutions = self.wake(tasks)
        return solutions


def describe_program(program: CandidateProgram) -> str:
    """Human-readable one-line program summary."""

    scanners = ", ".join(f"{name}/{axis}" for name, axis in program.scanner_keys)
    return (
        f"{program.name}: scanners=[{scanners}] "
        f"mse={program.mse:.4f} complexity={program.complexity:.3f} "
        f"objective={program.objective:.4f}"
    )


__all__ = [
    "CandidateProgram",
    "DreamCoderLearner",
    "DreamTask",
    "Example",
    "LibraryFragment",
    "Observation",
    "ScannerSpec",
    "Trajectory",
    "WakeRound",
    "build_scanner_graph",
    "default_library",
    "describe_program",
    "evaluate_program",
    "make_trigram_field",
]
