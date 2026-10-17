# Embodied-LoT x GRACE: the computation and its speed (2026-10-17)

## The task
Bayesian program induction. For each of 26 locomotion manners in the GRACE video set (25 actors;
split 15 train / 5 validation / 5 sealed test), infer a posterior over **programs** in a
grounded "language of thought" that explains that manner's training videos.

- **Data:** each video becomes 16 frames. Two interfaces exist:
  - occupancy: a 12x12 field per frame;
  - body parts: 6 parts (head, torso, arms, legs), each read by 6 soft spatial detectors per frame.
- **Hypotheses:** discrete program trees.
  - Leaves are trigrams (8 kinds x 2 orientations, static or dynamic) and `out` / `in` window leaves.
  - Compositions are parallel, serial, plan (template + content) and body.
  - Dashed gate edges link nodes; a program may be a forest of up to 6 trees.
  - Limits: up to 24 nodes, depth 5, 5 dashed edges.
  - There are no continuous parameters.
- **Rendering a program = simulating it.** A custom "gameboard" engine runs the program for 16 ticks on a
  12x12 board. Each tick it does the following:
  - observers re-evaluate their subtrees against last tick's board;
  - gates propagate;
  - motion leaves issue orders that move bodies and "mass" cells (several motion kinds);
  - templates rotate content frames;
  - every tree's space is redrawn;
  - each output window is read out.
  This is branching Python over a small object graph, not array maths.
- **Likelihood:** -beta x the summed distance between the rendered output channels and each training
  video. The distance includes a frame-to-frame-change term, and for body parts it uses an optimal
  output-to-part assignment (Hungarian algorithm) per video.
- **Prior:** description length of the tree (grammar log-probability plus dashed-edge cost), weighted
  by lambda.

## The sampler
A custom Metropolis-Hastings sampler over trees, with **parallel tempering**. It is pure Python plus
numpy; no PPL is involved. NUTS/HMC don't apply: the space is discrete and there are no gradients.

- **Moves** (each with an exact Hastings correction, tested against enumerated distributions):
  regenerate a subtree, change a leaf, change a composition, swap siblings, add/remove a dashed edge,
  wrap/unwrap a node, add/remove a child.
- **Per manner:** 4 chains x 12 temperature replicas x 40,000 iterations = **1.92 M MH steps**. Each
  step is one program rollout (16 ticks) plus a likelihood against 15 videos, with an LRU cache of
  rendered programs.
- **Selection:** a hyperparameter grid (lambda x beta x change-term weight) is scored by recognising
  the validation actors' videos, keeping only settings that converge (median R-hat <= 1.1).
- **Parallelism:** one chain per process (multiprocessing), 24 workers. Hardware: 24 CPU cores / 32
  threads, 32 GB RAM, plus an RTX 4060 laptop GPU (8 GB) that is currently unused.

## Measured speed
These are wall-clock times without a profiler. An earlier figure of ~62 ms per step for v6 was
inflated about 3x by cProfile's per-call overhead.

| Language version | Time per MH step | What it means |
| --- | --- | --- |
| v2 (legacy semantics) | ~0.4 ms | a full 3x3 selection took ~8 h |
| v3 (gameboard engine, occupancy) | ~1.7 ms | a full 3x3 selection (936 chains) took ~10 h on 24 workers |
| v6 (current), before optimisation | 27.1 ms | measured on a jumping chain, 600 iterations x 12 replicas |
| v6 (current), after the three optimisations below | **11.2 ms** | the same chain: 2.4x faster end to end |

## Optimisations done (2026-10-17), each verified bit-for-bit
The checks:
- the exact rollouts of 240 representative v6 programs, 150 v3 programs and the 13 explainer
  examples;
- the 600-program legacy fuzz;
- the 43 unit tests.

Mean v6 rollout time per step:

| Change | v6 rollout | v3 rollout |
| --- | --- | --- |
| baseline | 23.4 ms | 4.5 ms |
| 1. cache rollout-invariant graph metadata (roots, parents, tree index, roles, template set, input context, grid index arrays), keyed on a structure version | 17.4 ms (-26%) | 4.3 ms |
| 2. incremental recomputation: memoised node views (recomputed only when structure, body placement or template gates change); per-tree dirty redraws (redrawn only when that tree's gates, bodies, mass, inputs or flags change), which removes the second draw per tick | 10.2 ms (-41%) | 4.1 ms |
| 3. scalar arithmetic for tiny reductions: `_mean` for short lists, `sum()/size` for band means, sub-fields as views of already-checked parents, scalar clip/trunc | 6.5 ms (-36%) | 3.4 ms |

Total: **3.6x faster rollouts** for v6 and 1.3x for v3.

A numpy subtlety found on the way: `np.mean` of 3-7 values may add in vectorised lanes, and the
order depends on memory layout, so a plain Python sum can differ by one rounding unit. `_mean` uses a
Python sum only for 1-2 values and `np.add.reduce` (what `np.mean` does internally) otherwise.

### The likelihood path (profiled after the rollout work)
On a real jumping chain (300 iterations x 12 replicas, 15 training videos), rollouts are ~94% of the
wall time; the whole matching likelihood after the rollout is ~1 ms per step before and ~0.3 ms after.
Per program (240-program corpus, rollouts pre-rendered):

| Stage | Before | After |
| --- | --- | --- |
| detector scan (`scan_field`) | 0.223 ms | 0.051 ms |
| cost matrices, frames | 0.048 ms | unchanged |
| cost matrices, frame-to-frame changes | 0.046 ms | unchanged |
| Hungarian, 15 videos | 0.018 ms | unchanged |
| whole likelihood | 0.36 ms | 0.18 ms |

The one change: a cell's x detectors depend only on its column and its y detectors only on its row,
so `scan_field` takes each max over the columns (rows) holding any salient cell, with the detector
responses cached per grid size. The values, matchings and log-likelihoods are bit-identical (3000
random-field cases plus the 240 programs' log-likelihoods and assignments).

Not changed, and why:
- The cost einsum could be reordered (matmul, pre-transposed observations), but that changes the
  floating-point summation order, and it is ~1% of a step.
- Observed energies and changes are already computed once per likelihood, not per video or call.
- Reusing per-output scans or cost rows across proposals saves at most the 0.18 ms left. Only ~3% of
  likelihood calls hit the render cache, so nearly every step is a fresh rollout. The reuse that would
  pay is in the rollout: a proposal that changes one tree could keep the rollouts of trees not linked
  to it by dashed edges (their spaces are separate). The kept trees' node ids must not shift, since
  they seed the random pushes. This is not yet implemented.

## Constraints on speed-ups
- The hypothesis space is discrete and heterogeneous (trees of different shapes), so JAX or GPU
  vectorisation needs a redesigned engine: compile programs to fixed tensor "instruction lists" and
  run thousands of programs in lockstep, with masking for branches. It would also need a sampler that
  advances many chains in lockstep.
- The semantics must be preserved exactly. A fuzz test checks bit-for-bit equivalence of rollouts.

## Candidate speed-ups (not yet done)
1. Fewer temperature replicas (an adaptive ladder), early stopping per manner once R-hat converges,
   and fewer iterations with better proposals.
2. Compile the hot loops (numba/Cython) after restructuring the data as arrays.
3. In the long run, a lockstep batched engine on the GPU.
