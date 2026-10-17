"""
Demos for the DreamCoder-style Embodied-LoT learner.

Run:

    python dreamcoder_demos.py

The demos print learned graph programs and their predictions.  The learned
programs are ordinary ConceptGraph objects, so they can also be passed to
visualize_lot.visualize().
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]   # legacy: siblings + embodied_lot
from embodied_lot import TRIGRAM_PATTERNS, Orientation
from dreamcoder_lot import (
    DreamCoderLearner,
    DreamTask,
    Example,
    ScannerSpec,
    build_scanner_graph,
    default_library,
    describe_program,
    evaluate_program,
    make_trigram_field,
)


def _example_set(pattern_names: list[str], orientation: Orientation) -> list:
    return [
        (name, make_trigram_field(TRIGRAM_PATTERNS[name], orientation))
        for name in pattern_names
    ]


def demo_1_learn_top_detector() -> None:
    """Learn a single TOP scanner from supervised examples."""

    print("\n=== Demo 1: learn a TOP detector ===")
    teacher = build_scanner_graph(
        [
            # "top" is (0, 0, 1); index 2 occupies the top band.
            ScannerSpec("top", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL)
        ],
        name="teacher_top",
    )

    fields = _example_set(
        ["top", "bottom", "center", "universal", "void", "upper"],
        Orientation.VERTICAL,
    )
    teacher_examples = [Example(field, 0.0, label=name) for name, field in fields]
    targets = evaluate_program(teacher, teacher_examples)
    examples = [
        Example(field, target, label=name)
        for (name, field), target in zip(fields, targets)
    ]

    learner = DreamCoderLearner(max_fragments=2, max_scanners=2)
    solution = learner.run([DreamTask("top_detector", examples)], iterations=1)["top_detector"]

    print(describe_program(solution))
    for example, pred in zip(examples, solution.predictions):
        print(f"  {example.label:10s} target={example.target:.3f} pred={pred:.3f}")


def demo_2_invent_edge_chunk() -> None:
    """
    Learn two related tasks that both need TOP+BOTTOM.

    The initial library excludes compound trigrams such as periphery, so the
    learner must express "edge presence" as a composition of top and bottom
    scanners.  Because the same pair solves multiple tasks, compression adds
    a cheaper invented chunk to the library.
    """

    print("\n=== Demo 2: invent a reusable TOP+BOTTOM chunk ===")
    primitive_library = default_library(include_compound_trigrams=False)
    learner = DreamCoderLearner(
        primitive_library,
        max_fragments=2,
        max_scanners=2,
        compression_min_uses=2,
    )

    teacher = build_scanner_graph(
        [
            ScannerSpec("top", TRIGRAM_PATTERNS["top"], Orientation.VERTICAL),
            ScannerSpec("bottom", TRIGRAM_PATTERNS["bottom"], Orientation.VERTICAL),
        ],
        name="teacher_vertical_edges",
    )

    fields = _example_set(
        ["top", "bottom", "center", "universal", "void", "periphery"],
        Orientation.VERTICAL,
    )
    teacher_examples = [Example(field, 0.0, label=name) for name, field in fields]
    targets = evaluate_program(teacher, teacher_examples)

    task_a = DreamTask(
        "vertical_edge_presence_a",
        [Example(field, target, label=name) for (name, field), target in zip(fields, targets)],
    )
    task_b = DreamTask(
        "vertical_edge_presence_b",
        [Example(field, target, label=f"variant_{name}") for (name, field), target in zip(fields, targets)],
    )

    solutions = learner.run([task_a, task_b], iterations=2)
    for round_info in learner.history:
        invented = ", ".join(round_info.invented_fragments) or "none"
        print(f"  round {round_info.iteration}: invented={invented}")

    for task_name, solution in solutions.items():
        print(describe_program(solution))
        print("  predictions:", ", ".join(f"{p:.3f}" for p in solution.predictions))

    invented = [fragment for fragment in learner.library if fragment.invented]
    print("  invented library fragments:")
    for fragment in invented:
        parts = ", ".join(f"{spec.name}/{spec.key[1]}" for spec in fragment.scanners)
        print(f"    {fragment.name}: [{parts}], cost={fragment.cost:.3f}, uses={fragment.uses}")


if __name__ == "__main__":
    demo_1_learn_top_detector()
    demo_2_invent_edge_chunk()
