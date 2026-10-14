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
python mcmc_demos.py
python dreamcoder_demos.py
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
| `learning_data.py` | `Observation`, `Trajectory` and the scoring code both learners share |
| `concept_learning.py` | Metropolis-Hastings learner over graph structure |
| `mcmc_demos.py` | MCMC learner example |
| `dreamcoder_lot.py` | DreamCoder-style wake/compress learner over flat scanner sets |
| `dreamcoder_demos.py` | DreamCoder learner examples |
| `visualize_lot.py` | Writes the standalone interactive visualizer and replay pages |
| `visualizer/` | The visualizer page itself: `page.html`, `viewer.css`, `viewer.js` |
| `visualize_demo.py` | Generates `top_replay.html` from a small two-component graph |

## Learners

The two learners are separate experimental approaches. They share their data
types and scoring (`learning_data.py`), so the same `Observation`s and
`Trajectory`s can be given to either.

| | MCMC (`concept_learning`) | DreamCoder-style (`dreamcoder_lot`) |
| --- | --- | --- |
| Search | Metropolis-Hastings over local mutations | Enumeration of library combinations |
| Hypotheses | Trees of Serial/Parallel/Plan/Body nodes with trigram leaves | Flat `ParallelNode(I, scanner, ...)` sets |
| Moves | change trigram, flip orientation, flip dynamic, add/remove leaf, change structure type | - |
| Reuse across tasks | none | compresses repeated scanner pairs into cheaper library chunks |
| Score | Gaussian log-likelihood + size/depth prior | weighted MSE + library cost |
| Data | Observations and Trajectories | Observations and Trajectories |
| Not yet | dashed edges, I/O windows, imagined roots, output fields | everything structural, dynamic scanners |

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

## Historical Material

The `legacy/` directory and `images/` directory contain prior experiments and
reference assets. They are intentionally not imported by the current runtime.
