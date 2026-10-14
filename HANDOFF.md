# Embodied LoT Handoff

## Purpose

This repository is a computational model of the structure of grounded human
concepts, built from perceptual and attentional operators. It has a core
evaluator, two learning experiments, runnable demos, and a standalone browser
visualizer.

## Current Entry Points

```text
python demos.py [--visualize]
python mcmc_demos.py
python dreamcoder_demos.py
python visualize_demo.py
python -m unittest test_embodied_lot
```

`visualize_demo.py` regenerates `top_replay.html`.

## Runtime Files

| File | Responsibility |
| --- | --- |
| `embodied_lot.py` | Core nodes, graph evaluation, dashed gates, body motion, output rendering |
| `trigrams.py` | Pattern aliases such as `TOP`, `BOTTOM`, `UNIVERSAL`, `VOID`, `V`, and `H` |
| `demos.py` | Main worked examples; public builders include `build_above()` and `build_abc()` |
| `visualize_lot.py` | Serializes a graph into an editable standalone HTML visualizer |
| `visualize_demo.py` | Small replay example for the visualizer |
| `learning_data.py` | `Observation`/`Trajectory` data and scoring shared by both learners |
| `concept_learning.py` | Metropolis-Hastings learner over graph structure |
| `mcmc_demos.py` | Example for the MCMC learner |
| `dreamcoder_lot.py` | Wake/compress learner for reusable scanner fragments |
| `dreamcoder_demos.py` | Examples for the DreamCoder-style learner |
| `visualizer/` | Visualizer page (`page.html`, `viewer.css`, `viewer.js`) |

The capabilities of each learner are listed in its module docstring and in
the README. They are separate approaches; neither replaces the other.

`legacy/` and `images/` are historical experiments and assets. They are not
part of the current runtime path.

## Core Model

### Observed and imagined object nodes

An `ObjectNode` is observed when it is beneath a `ParallelNode` that has an
`InputNode` child. It scans the current `ReceptiveField`.

```text
observed activation = gate * scan(field)
```

An `ObjectNode` without input context is imagined or field-free.

```text
imagined activation = gate
```

### Dashed edges

Dashed edges are positive conditional gates:

```text
A --dashed--> B
B.gate = mean(incoming source activations)
```

Propagation uses dashed-edge topological order. Field-free object targets
immediately mirror `activation = gate`; therefore an imagined chain
`A -> B -> C` can propagate in one `evaluate()` call. Observed targets keep
their gate but are re-scanned against their field.

Dashed edges are not logical NOT and do not subtract from targets.

### Yang and yin readings

Every activation is simultaneously a yang and yin proportion. If `a` is its
yang reading, its yin reading is `1 - a`; equivalently, `1 - yin = yang`.
This complement is available everywhere in the model, including for field
values and dashed-edge source activations. A dashed edge propagates its
source's stored activation; it does not silently choose the complementary
reading. Structures such as a `VOID` plan template explicitly use `1 - a`
when they require the yin reading in their computation.

### Telicity and complement blocking

Use a `VOID` object as the first template child of a field-free `PlanNode` to
explicitly apply that complement in a branch:

```text
PlanNode(VOID, content) = (1 - void.gate) * content.gate
```

For example, an observed completion detector can gate the `VOID` template,
while an enabling detector gates the content/action node.

### Evaluation

```text
evaluate(): reset gates -> scan -> propagate dashed gates -> scan again
step():     scan -> propagate dashed gates -> update BodyNodes
```

`step()` preserves gates and body state. `evaluate()` is stateless.

### Spatial order and body motion

Vertical patterns read bottom to top; horizontal patterns read left to right.
`SerialNode` averages the activations of its allocated child strips.
The two directions in `TRIGRAM_MOTION` describe spatial halves, not y/x
components. Body translation averages each half's direction times magnitude,
then applies it along the dynamic template's orientation: y for vertical,
x for horizontal. Opposing half-motions cancel centre translation.

## Two-Component Pattern

The common pattern is two independent solid-edge roots joined only by dashed
edges:

```text
observed root:  ParallelNode(I, scanners)
imagined root:  ParallelNode(O, imagined structure)

observed scanners --dashed--> imagined object gates
```

`ConceptGraph._scan()` evaluates every solid-edge root. A graph does not need
to force the imagined root below the observed root.

## Visualizer Notes

`visualize_lot.py` writes a self-contained HTML file. It supports replay,
manual node layout, marquee selection, graph editing, export, custom sprites,
and browser-side evaluation for edited structures.

- Manual layout is the default. `auto layout` is opt-in and turns off when a
  node is manually dragged.
- The visualizer uses the same observed-versus-imagined dashed-gate rule as
  the Python core and processes dashed edges topologically.
- Browser edits affect only the embedded graph snapshot; they never mutate the
  Python `ConceptGraph` that generated the file.

## Maintenance Rules

1. Keep observed-vs-imagined behavior in `embodied_lot.py` and
   `visualizer/viewer.js` (the browser-side evaluator) aligned.
2. Add a demo when changing semantic behavior, especially gates, plans, or
   body motion.
3. Keep current demos in `demos.py`; do not import from `legacy/`.
4. Prefer `InputNode` and `OutputNode` in implementation code. `ReturnNode`
   and `ReceptiveFieldNode` remain compatibility aliases.
5. Run the entry points above plus `py_compile` after core or visualizer work.
6. Settled semantics: dashed edges copy the source activation (never invert);
   only VOID templates use the yin reading `1 - a`; `SerialNode` takes the
   mean of its strips.
7. Band and strip geometry lives in one place, `band_region()` in
   `embodied_lot.py`. Use it rather than re-deriving row/column slices.
