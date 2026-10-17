# Embodied Language of Thought

Embodied LoT is a computational model of the structure of grounded human
concepts. A concept is a graph of perceptual and attentional operators:
image-schema scanners, input/output windows, structural nodes, and
conditional dashed gates. Its results are meant to explain generalisation and
learning experiments, and later perhaps gesture and colexification data.

## Quick Start

```python
import numpy as np

from embodied_lot import ConceptGraph, InputNode, ObjectNode, ParallelNode, ReceptiveField
from trigrams import TOP, V

graph = ConceptGraph("top detector")
# connect() adds any new nodes and the solid edges hub -> I, hub -> top
graph.root = graph.connect(ParallelNode("hub"), InputNode("I", modality="retina"),
                           ObjectNode("top", TOP, V))

graph.set_field("I", ReceptiveField(np.ones((9, 9))))
score = graph.evaluate()
graph["top"].activation          # nodes are looked up by id
```

Run the demos with:

```text
python demos.py               # add --visualize to also write demos_gallery.html
python visualize_demo.py      # writes top_replay.html
python -m unittest test_embodied_lot
```

## Core Semantics

### Observed and imagined nodes

An `ObjectNode` is observed when it has an `InputNode` context through a
`ParallelNode`; it scans the input field. A field-free `ObjectNode` is
imagined; it represents its current gate directly.

```text
observed activation = gate * scan(field)
imagined activation = gate
```

### Dashed edges

A dashed edge is a positive conditional gate:

```text
A --dashed--> B
B.gate = A.activation
```

Several incoming dashed edges are averaged. Dashed targets are processed in
topological order. Field-free imagined object nodes immediately mirror their
new gate to activation, so an `A -> B -> C` chain can propagate in one
evaluation.

Dashed edges are not logical NOT or inhibitory edges. Every activation `a` is
simultaneously readable as yang `a` and yin `1 - a`, including field values
and dashed-edge source activations. A dashed edge propagates its stored
activation; it does not silently select the complementary reading. A `VOID`
`PlanNode` template is one structure that explicitly applies `1 - a`.

### Plan nodes and complement blocking

For a field-free plan whose first child is a `VOID` template and whose last
child is content, the yin reading is used explicitly:

```text
PlanNode(VOID, content) = (1 - void.gate) * content.gate
```

This supports patterns such as a telic action: an observed completion detector
gates the `VOID` template and suppresses an imagined action content node.

### Serial composition

`SerialNode` gives each child one strip of its field (bottom-to-top or
left-to-right, like trigram bands) and takes the **mean** of the children's
scores. Partial matches therefore still count; this is deliberate, not an
AND.

### Imagined evaluation

Without a field, every node of an imagined subtree is still evaluated on
every step. Dashed loops inside a body therefore change its content over time
while it stays the same body, and bodies nested under plans or serial nodes
move. (`embodied_lot.EVALUATE_IMAGINED_SUBTREES`; `False` reproduces an
earlier shortcut.)

### Gameboard dynamics (grid world)

By default (`graph.dynamics = "gameboard"`; "legacy" keeps the original step),
`step()` runs a board-game grid world:
- observers (dashed sources, including parents) perceive the imagined scene
  drawn on the previous tick, within their own grid region;
- every dashed effect takes one tick, for leaf and group targets alike;
- a dynamic trigram that is on (strictly above 0.5) sends each body in its
  own region one whole-cell movement order (each time a body walks in, it
  gets another), carried out one cell at a time at a
  rate set by its activation;
- cells nobody draws are unknown (0.5); imagined yin bands are known yin (0);
- motion runs from a trigram's figure side across (top/H* moves left), and
  a dynamic template turns its content to face its motion instead of moving
  anything (nested turns add up, and observers and motion inside turned
  content turn with it);
- programs have no numbers in them: a `body` fills the region its place in
  the tree gives it, and moves at its motion leaf's gate times the
  trigram's motion strength.

`python -m grace_lot.analysis.gameboard_traces` builds an interactive explainer
(`results/gameboard_viz/gameboard.html`).

Bodies can therefore be driven back and forth by observers that watch them.
See HANDOFF.md (item 10).

### Evaluation

`evaluate()` is stateless:

```text
reset gates -> scan -> propagate dashed gates -> scan again
```

`step()` preserves gates and body state across calls:

```text
scan -> propagate dashed gates -> update BodyNodes
```

## Main Files

| File | Responsibility |
| --- | --- |
| `embodied_lot.py` | Core graph, node types, evaluation, body motion, output rendering |
| `trigrams.py` | Readable aliases for trigram patterns and orientations |
| `demos.py` | Worked examples (`demo_1` ... `demo_12`) and builders such as `build_above()` and `build_abc()`; importing it runs nothing |
| `visualize_lot.py` | Writes the standalone interactive visualizer and replay pages |
| `visualizer/` | The visualizer page itself: `page.html`, `viewer.css`, `viewer.js` |
| `visualize_demo.py` | Generates `top_replay.html` from a small two-component graph |

## Learners

The learner is the GRACE program learner in `grace_lot/inference/`: tempered
Metropolis-Hastings with exact Hastings corrections over program trees, scored on
rendered videos. The first two learners (an MCMC learner over observing concept
graphs, and a DreamCoder-style wake/compress learner) are retired to
`legacy/learners/` with their demos and toy data; see the README there.

## Visualization

```python
from visualize_lot import visualize

graph.set_field("I", field)
graph.evaluate()
visualize(graph, "network.html")
```

The generated HTML supports replay, manual layout, graph editing, structure
export, and a browser-side approximation of evaluation. Browser edits are
exploratory; they do not mutate the original Python graph.

To change the page, edit the plain files in `visualizer/`. `visualize_lot.py`
only inserts the title and the graph data. `viewer.js` contains a JavaScript
copy of the evaluator, which has to be kept in step with `embodied_lot.py`.

## GRACE locomotion pipeline

`grace_lot/` tests whether LoT programs inferred from human locomotion videos
(the GRACE database) predict human similarity judgments. The stages are kept
separate and each is configured in `configs/grace_poc.toml`:

| Package | Stage |
| --- | --- |
| `grace_lot/grace/` | video -> MediaPipe pose -> normalised skeleton -> T x H x W field sequence |
| `grace_lot/lot_adapter/` | program trees, `lot_trajectory` rollouts, distances, program similarity |
| `grace_lot/inference/` | grammar, prior, likelihood, MH proposals, sampler |
| `grace_lot/analysis/` | human matrix, kinematic baselines, RSA, ablations, plots |

The design is in `docs/grace_proposal.md` and the first results are in
`docs/grace_results.md`. The raw videos, caches and results are gitignored.
The only change to the core is an opt-in `render_output(..., place_bodies=True)`.

## Historical Material

The `legacy/` directory and `images/` directory contain prior experiments and
reference assets. They are intentionally not imported by the current runtime.
