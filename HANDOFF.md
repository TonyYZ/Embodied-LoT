# Embodied LoT Handoff

## Purpose

This repository is a computational model of the structure of grounded human
concepts, built from perceptual and attentional operators. It has a core
evaluator, two learning experiments, runnable demos, and a standalone browser
visualizer.

## Current Entry Points

```text
python demos.py [--visualize]
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
| `legacy/learners/` | The first two learners (MCMC over observing graphs, DreamCoder-style), retired 2026-10-17; the GRACE learner is `grace_lot/inference/` |
| `visualizer/` | Visualizer page (`page.html`, `viewer.css`, `viewer.js`) |

The learner is the GRACE program learner (`grace_lot/inference/`). The two
earlier learners are retired to `legacy/learners/` (see the README there); they
still run, but predate the current language.

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
8. Imagined (field-free) evaluation reaches every node on every step
   (`EVALUATE_IMAGINED_SUBTREES = True`). Dashed loops and dynamic trigrams
   inside bodies, plans and serial nodes are therefore live. A body can hold an
   internal cycle and change state while remaining the same body. A field-free
   `SerialNode` averages its children. `False` reproduces the earlier
   shortcut; GRACE configs select it with `lot.imagined_evaluation`
   ("full" | "legacy"), and runs made before the change pin "legacy".
9. Motion is graded: velocity = trigram direction x activation x step, so a
   weaker activation of a motion node moves the body more slowly.
10. `step()` has two dynamics (`ConceptGraph.dynamics`, default from
    `embodied_lot.DEFAULT_DYNAMICS`, now "gameboard"; GRACE configs use
    `lot.dynamics`, and the four existing configs pin "legacy"):
    - **"legacy"**: bodies drift along their first dynamic trigram, and
      imagined dashed targets update within the same step.
    - **"gameboard"**: a board-game grid world (`ConceptGraph.grid` cells).
      - Any node that is a dashed source but not a dashed target (or a leaf
        with an input window) is an **observer**.
        - It perceives the previous tick's imagery in its own grid region
          (strip or body window) and is not drawn.
        - A parent observer perceives the pattern its children form, read the
          way its type composes them: serial strips, parallel pooling, plan
          template, body window. Its whole subtree is its perceptual pattern,
          and only the top-most observer perceives directly.
        - Everything else is imagined and drawn. Consequence: an internal
          imagined oscillator must use nodes that are both sources and targets;
          `plan(VOID t, x)` with only `plan -> t` makes the plan an observer.
      - Every dashed effect takes one tick, and dashed targets start unknown
        (gate 0.5; 0 when `UNKNOWN_BACKGROUND` is off).
        Orders are issued before propagation in each tick, so a gate set at
        tick t acts from tick t+1 whether its target is a leaf or a group.
      - When an imagined dynamic trigram's effective activation rises strictly
        above `ONSET_THRESHOLD` (so seeing only unknown, 0.5, is not "on"), it
        issues one **order** to every body whose window overlaps the trigram's
        own region (its strip, or its body's window) at that tick. Where the
        body sits in the tree does not matter. While the trigram stays on, every
        body that enters its region gets one order, including one that left and
        came back. A body that stays inside gets no more. Only presence is
        remembered (`_inside`), not which bodies were ordered. An order is one
        body extent in
        whole cells, in the trigram's direction. Orders add as vectors.
      - A body advances at most one cell per order per tick, at
        `step_size x activation` cells per tick via accumulated credit. Weaker
        activation therefore means the token moves less often. Positions are
        always whole cells.
    The self-observing oscillator in `test_grace_lot.py` is the reference example.
11. Dashed edges may target any node. A structure's gate scales its whole
    composite (`GATE_STRUCTURES`; config `lot.gate_structures`):
    - its own activation;
    - the weight its subtree is drawn with;
    - in gameboard, the effective activation of motion leaves beneath it.
    A plan template's gate keeps its separate role as the template
    probability. The GRACE search grammar chooses targets with
    `lot.dashed_targets` ("leaves" | "any"); existing configs use "leaves",
    which reproduces earlier search spaces.
12. Undrawn imagery is unknown, 0.5 (`UNKNOWN_BACKGROUND`; config
    `lot.unknown_background`, default true; the four existing configs pin
    false).
    - Imagined yang bands pull a cell towards 1, and imagined yin bands pull it
      towards 0 (known yin).
    - A neutral bit draws nothing, and neither does a dynamic (motion) trigram:
      it specifies motion, not a static figure.
    - Yang and yin paint pool separately by noisy-OR (Y, N), and a cell reads
      0.5 + 0.5(Y - N).
    - Observers still score only their yang bands, so their yin bands act as
      masks.
    - With false, undrawn cells are 0 and yin paints half strength into the
      yang noisy-OR. This is bit-for-bit the earlier renderer (fuzz-checked).
13. Saved run configs written before a semantics key existed are read with its
    legacy value (`program.saved_lot`), and new runs record every semantics
    key explicitly (`program.resolved_lot`). Re-analysing old runs is
    therefore unaffected by the new defaults.
14. Body positions are not parameters (`TREE_PLACED_BODIES`; config
    `lot.tree_placed_bodies`, default true; the four existing configs pin
    false).
    - `reset_gates()` centres every body in the region its place in the tree
      gives it: its serial strip, its parent body's window, or the whole
      field.
    - Grammar bodies are `body[h,w,speed]`, so `body_positions` is unused.
      Earlier runs keep `body[y,x,h,w,speed]` with hand-set positions.
    - Extent stays a fraction of the whole board.
    - A vertical serial stacks its children bottom-up (band 0 is the bottom,
      as in trigrams).
15. Motion runs from a dynamic trigram's figure side across to the
    background on both axes (`MOTION_AWAY_FROM_FIGURE`; config
    `lot.motion_away_from_figure`, default true; the four existing configs
    pin false).
    - `top/V*` moves down, `bottom/V*` up, `top/H*` left and `bottom/H*`
      right.
    - Before, horizontal motion was mirrored: horizontal bands count from
      the left and columns grow rightward, so the vertical rule's cancelling
      reversal was missing.
    - Observed motion is scored as |motion|, so perception is unaffected.
16. Dynamic templates orient their content (`ORIENTING_TEMPLATES`; config
    `lot.orienting_templates`, default true; the four existing configs pin
    false, which keeps the earlier half-field selection).
    - Each dynamic leaf in a template gives a direction (its motion), and the
      content's up is turned to face it (`orient_slots`, `_orientation_map`).
    - A serial template gives one slot per strip. Any other template gives one
      slot over its region, facing the sum of its dynamic leaves' directions.
      Static parts keep their usual slots (yang bands).
    - Turning is a rotation done on whole cells. Every slot cell shows one
      cell of the content's upright frame, so the four axis directions are
      exact permutations and diagonals become stair-stepped bands.
    - Observers read the field carried into that frame; frame cells nothing
      maps to read as unknown.
    - As for static templates, with p (the template's gate) the content is
      oriented, and with 1 - p it is used unchanged. Field-free, turning
      changes nothing, so an orienting plan evaluates to its content.
    - Dynamic leaves inside templates issue no gameboard orders.
    - Turns compose: a dynamic template inside turned content turns again,
      so the angles add. This holds for drawing and observing alike.
    - Gameboard geometry follows the turns through `node_views`. Each node gets
      views `(weight, chain, rect)`: one per template slot (p / n) plus the
      unturned 1 - p view, each with its chain of turned frames.
      - Observers perceive through their views (a weighted mean).
      - A motion leaf's space is the union of its views' board cells
        (`view_masks`), and its push is turned back to the board
        (`_view_turn`). A diagonal push is one order per axis.
      - Content under a static template now uses the template's yang slots
        too; before, it took the plan's whole region.
    - Bodies nested in turned content live in their turned frame. Their
      position and extent are fractions of that frame, and tree placement,
      drawing and windows all work in the frame's cells. A frame is its own
      root when observed, so a nested body carves its window from it.
      - Moves run along the frame's axes: a board push is turned into the
        body's frame (`_view_unturn`), and a diagonal is one order per axis.
      - Overlap with a motion leaf's space uses the body's board cells
        (`_moving_bodies`).
      - A body copied into several slots has one state, kept in its first
        view's frame.
    - The visualizer's `derived_fields` is not turned.
17. A program's meaning comes purely from its tree (`TREE_SIZED_BODIES`;
    config `lot.tree_sized_bodies`, default true). Every config with saved
    results pins false, including grace_v3, grace_v3_frozen and
    grace_v3_parts.
    - A body is `body(...)`, with no numbers. It fills the region its place in
      the tree gives it.
    - In gameboard a body moves at its motion leaf's gate (times its groups' gates)
      times the trigram's motion strength, in cells per tick. `upper` and
      `lower` have half the strength of `top` and `bottom`.
    - Earlier `body[...]` programs still parse for pinned runs.
18. The likelihood can compare frame-to-frame changes (`inference.temporal_weight`
    w, default 0, so every saved run is unchanged): D = mean (x - y)^2 + w * mean
    (dx - dy)^2, with dx(t) = x(t) - x(t-1).
    - It applies to the occupancy interface and to both body-part interfaces.
      For the body-part ones, the change term is part of the cost the
      once-per-video output-to-part matching minimises.
    - Held-out recognition uses the same distance, and it now scores the
      body-part interfaces too (`heldout._parts_scores`).
    - `[selection].temporal_weight` adds w as a third grid axis.
    - With w = 0 all three likelihoods match the previous code exactly
      (450 evaluations checked).
19. Motion kinds (`MOTION_KINDS`; config `lot.motion_kinds`, default true; every
    config with results pins false). Motion moves **mass** as well as bodies.
    Mass is the cells drawn by imagined leaves outside bodies: particles with
    accumulated whole-cell offsets, drawn by the tree at their displaced
    positions, leaving unknown cells behind. Bodies stay whole tokens.
    - `top` / `bottom`: bulk. Everything in the space moves together: a body
      by its extent, mass by the space's extent.
    - `upper` / `lower`: gradient. Layers leave one per tick from the leading
      edge, so mass spreads and bodies queue.
    - `center` / `periphery`: the two halves of the space move apart or
      together, each by half its own size.
    - `universal`: a random push along the leaf's axis for each cell or body,
      every tick while on. It is seeded by leaf and tick, so videos are
      deterministic.
    - Everything moves at the gate, one cell or layer per tick when fully
      open.
    - `configs/grace_v5_parts.toml` is lenient to complex structure: 24
      nodes, 5 dashed edges, dashed cost 0.5, and a λ grid of {0.25, 0.5, 1}
      with w in {0, 1}.
20. Programs are forests, and outputs are leaves (`lot.output_leaves`, default true
    when `max_outputs` > 0; the parts configs with results pin false).
    - A program is 1..`max_outputs` trees, written side by side with no node above
      them: `(parallel center/H out) (parallel top/V out) | .1.0>.0.0`, where
      dashed-edge paths start with the tree's index. A single tree is just
      itself. (Internally a forest of 2+ trees is a P("outputs") tuple, which
      is not counted as a node; earlier runs' `(outputs ...)` programs still
      parse.) **Each tree is its own space**
      (`SEPARATE_TREE_SPACES`). Its observers perceive only that space, and its
      motion leaves move only its own bodies and mass. Things share a space
      only by being composed under a common node; trees still link through
      dashed edges.
    - An output is an `out` leaf (`WindowNode`): no pattern, draws nothing,
      scans nothing, no dashed edges. There can be any number of them
      (1..`max_outputs` in all), anywhere.
    - An output is only the region its window occupies: its own tree's space
      inside the window, in the window's frame. Its size and position come from
      its place in the tree, and it moves with a body it sits in.
    - To produce an output (for the likelihood), the window's cells are
      stretched to the full grid. So a window carried by a body outputs the
      same pattern throughout, and an output changes only when something moves
      relative to its window: content leaving it (then unknown), or a body
      moving within it.
    - Channels are matched to body parts by likelihood, as before. The prior
      requires 1..`max_outputs` windows, and MH is tested exact and
      irreducible.
    - With the pin (false), all roots are drawn on one shared board, and each
      root's channel is its whole board (earlier runs).
    - `configs/grace_v6_parts.toml` uses all of this.
21. Input windows are the mirror image of outputs: an `in` leaf (`InWindowNode`).
    - Its region comes from the tree. It writes an external field into its
      tree's space there, resized to the window's cells, on the board's
      scale: 1 figure, 0 known empty, 0.5 unknown. Observers covering it see
      it.
    - It scans nothing and takes no dashed edges. Data goes in through
      `ConceptGraph.set_window_input(k, field)`.
    - The search grammar doesn't generate it; GRACE needs no inputs.
    - The old `InputNode` portal is kept, as legacy, for the original demos,
      learners (now in `legacy/learners/`) and visualizers.
22. `python -m grace_lot.analysis.gameboard_traces` writes `results/gameboard_viz/`: the
    engine traces, and `gameboard.html`, a step-through explainer of the examples.
23. Windows are never templates (`lot.content_windows`, default true with output
    leaves; runs saved without the key read it as false).
    - In a body or plan the last child is the content and every earlier child a
      template (nested left to right). An `out` or `in` leaf may only be the
      content, just as windows neither send nor receive dashed edges.
    - Enforced by the prior (-inf), like the 1..max_outputs window count, so valid
      programs keep their prior and MH stays exact (tested).
    - The diag4 run predates it: 81-98% of its posterior mass used a window as a
      pass-through template, mostly to stack several outputs in one body.
24. Copying, grouping and agency are separate operators (`lot.grouping_nodes`, default
    true; every existing config pins it false; `configs/grace_v7_parts.toml` uses it).
    - `template T… M` copies its material M into its template's slots (and orients it
      when T is dynamic). It groups nothing. It is the earlier `plan`, and `plan` still
      parses as `template`.
    - `field c` makes its one child a rigid unit. Outside motion overlapping any part of
      it moves the whole unit. It has no agency.
    - `body c` is a field with agency. Its own switched-on dynamic leaves propel it
      every tick: gate cells per tick along the leaf's turned direction. Universal
      pushes it at random; centre and periphery give no net direction.
    - A dynamic leaf inside a field or body still moves every other substance
      overlapping its region (other units and loose mass in the same tree's space),
      never its own groups.
    - Nested groups follow their parent: an instance's window is carried along with the
      region it receives (`GroupToken.rect`, anchored where it was placed). Outside pushes
      reach only top-level units, so a nested group moves only with its parent, or relative
      to it by its own agency (each leaf belongs to its innermost group). A leaf never pushes
      anything inside its own units.
    - Template copies of a group are independent instances, each with its own place,
      orders and drive (`GroupToken`). An instance is keyed by its path of template
      slots, which the views and the painter share.
    - `field` and `body` are unary, so there is one encoding per concept. v7 keeps v6's
      limits (`max_depth` 5), so it differs from v6 only in these semantics; only templated
      bodies (`body (template …)`) cost one level more than v6's `(body T C)`.
    - The window rule (`content_windows`) applies to `template` only.
    - Verified: with the flag off, all v6/v3/explainer and legacy-fuzz rollouts are
      bit-identical (1,003 programs), and the 44 earlier tests pass. New
      `GroupingTests` cover instances, rigid fields, loose copies, agency, outward-only
      leaves, the grammar and MH exactness.
    - The gameboard explainer pins the v6 semantics.
25. The probabilistic reading (`docs/probabilistic_semantics.md`): a cell value is p(yang), gates are mixture
    weights, and unknown motion is a latent.
    - `lot.probabilistic_motion`: `universal*` means motion of unknown direction, so a unit (its innermost
      field or body, else the leaf) is pushed one way or the other, ½ each. `neutral*` means unknown
      whether there is motion: still ½, either way ¼. `void*` means no motion.
    - `lot.max_motion_units` caps the units, at 2 in v7.
    - `inference.likelihood = "probabilistic"` (`ProbabilisticScanLikelihood`):
      - enumerates the motion states;
      - samples binary boards from the cell probabilities (one uniform per cell, held across frames; fixed
        seeds);
      - scores detector means and variances with a Gaussian cost on top of the β baseline, mixed by prior
        weight.
    - Certain programs score exactly as with the matched likelihood.
    - v7 uses both, and `neutral` is a primitive. Older configs pin both off; their rollouts are bit-identical.
    - A 0.5 cell is "no claim". It is predicted by a shared background, the training base rate of part
      occupancy (`parts.occupancy_background`, `inference.background = "occupancy"`), so concepts are
      deviations from the average body.
    - Cells vacated by moving mass, or by a moving body or group instance (its home region minus where it is
      now), read as known empty, so motion can't withdraw a claim.

26. The normalised likelihood for v7 (`docs/probabilistic_semantics.md` §10-11; `inference.likelihood = "gaussian"`,
    `GaussianScanLikelihood`).
    - `interface.observation_cells`: observed parts are quantised to the 12 × 12 cells before scanning
      (`scanner.part_boards`, `scan_part_cells`), so observed and LoT features come from the same `scan_field`.
      The gaussian likelihood requires it.
    - Each output predicts a part with a mean and a variance (claims with strength |2v − 1|; unclaimed cells
      filled from real part boards of the training videos, `scanner.feature_background`, cached). All
      outputs share each draw's fill, so the matching can't pick among Monte Carlo variants of "no claim".
    - Observation noise per detector = across-actor variance of the manner's videos, floored at
      `inference.noise_floor` (0.1) × the background variance. A part with no output is predicted by the
      background.
    - log p = −temperature × the matched Gaussian NLL; the temperature defaults to 1/(T·F). β is unused by it.
    - v7 now uses it, with `cell_draws = 256`; `"probabilistic"` (item 25) remains selectable. Older configs
      are unchanged.
    - Not yet handled: `analysis/heldout.py` still scores with the matched likelihood on point features;
      with this likelihood it must take the noise from the training videos, never the held-out ones.
25. v7 observation interface (sections 8 and 15 of docs/dynamic_reachability_audit.md).
    - `preprocess.sampling = "cycle2"` (`grace/temporal.py`): each video's 16 frames
      cover two motion cycles, resampled from the full-rate joints.
      - The period is the first autocorrelation peak reaching 80% of the highest,
        found the same way for every manner.
      - Without a period, or when two cycles don't fit, a video falls back to its
        whole segment (2 bowing videos and 1 swinging among the diagnostic manners).
      - Applied in `load_trajectory`, so every consumer of v7 gets the same frames.
      - "segment" (the default) is every earlier run's behaviour.
    - `interface.features = "orient8"` (`scanner.field_features`): 8 features per part.
      They are the 3 + 3 detectors averaged over the salient cells, plus the
      elongation-weighted principal-axis orientation, which tells / from \.
      - Used on both sides (cell-quantised parts and LoT boards) and in the background.
      - "max6" (the default) is the earlier scanner, bit-identical.
    - The Gaussian likelihood makes its Monte Carlo fills and uniforms once per draw
      index, not once per program (`_draws`). The same numbers give the same scores
      (153 programs bit-identical), 26 → 14 ms per program.
26. Measure trees (`lot.measure_trees`, v7 only; every older config pins it off).
    - The case: a tree whose top node is a dashed source would be all observer. Nothing
      in it is drawn, so it would scan an empty space and read a constant.
    - The rule: such a tree is drawn instead, and its activation is the mean figure
      claim of its own space as drawn last tick: mean of max(0, 2v - 1) over the
      board, 0.5 before the first drawing.
    - The effect: a dashed edge from it sets its target's gate to that amount, an alpha
      defined by the tree. A one-third-figure tree fades its target to 1/3, and a
      source whose figure changes modulates its target over time.
    - Unchanged: trees with an input window keep perceiving, and observers inside a
      drawn tree are as before.
    - `MeasureTreeTests` cover the fade, the time-varying alpha and the observer
      control.
    - The gameboard explainer now pins all of the v6 semantics, not just grouping.
27. Inference and tooling for v7 (Run 4 in `docs/v7_diagnostics.md`).
    - `inference.ladder = "calibrated"`: the hottest replica is set from the likelihood itself (the
      spread of 32 prior programs' log-likelihoods shrinks to `hot_spread` nats). Run 4 showed this is
      too hot: replicas below tau ≈ 0.05 swap 95-99% of the time. Next: space the ladder by swap rates.
    - `mh.r_hat` is split R-hat (each chain cut in half), so chains drifting together are flagged.
    - Runs print a progress line per chain every `progress_every` iterations and save checkpoints;
      `python -m grace_lot.inference.run --status <run dir>` gives split R-hat mid-run.
    - Probabilistic motion: a moved body leaves a full trail of known emptiness (every cell it has
      claimed, until `reset_gates()`), kept as a separate layer that content moving in covers
      (`_covered`). Trails are part of the draw-cache key.
    - Speed, 2.6x per chain: `adapter.rollout` rolls out trees joined by no dashed edge separately and
      caches them (v7 semantics only; bit-identical on 52k rollouts); the Gaussian likelihood memoises
      per-channel cost rows (canonical per row, within ~1e-16 of the batched sum).
    - `python -m grace_lot.analysis.playground` serves the v7 visualizer (`visualizer/page.html`: v7
      node types, program text in and out, runs on the real engine with tick replay and boards) at /,
      and a board-first page at /gameboard. Each run checks itself against `ConceptGraph.step()` and
      whole-program rollouts. `page.html` also opens on its own, with an example graph.
