# GRACE x Embodied-LoT: technical proposal

Question: can a compact compositional program language recover structural
invariants across human locomotion events, and do the inferred programs
predict human similarity judgments beyond low-level kinematic similarity?

## 0. Data facts that shape the design

* 676 videos at 640x480 and 25 fps, with one actor walking left to right
  across a plain wall.
* `00F_*` are the **experimenter's gesture videos**, not locomotion. The
  locomotion actors are 01-13 F/M. Actor 10F withdrew consent, so her
  videos are missing (25 locomotion actors x 26 manners).
* `Norming_Study_3.csv` holds the pairwise similarity ratings: 26 manners,
  all 325 pairs, about 20 ratings each (1-7), one exemplar video per
  manner. **The exemplar for scurrying is `10F_scurrying.mp4`, which is not
  in the public release.** The prototype uses a configurable substitute
  (`configs/grace_poc.toml`) and flags it in every result.

## 1. How the LoT represents and runs programs

* A program is a `ConceptGraph`: a tree of nodes joined by solid edges, plus
  dashed gate edges.
  * Leaves are `ObjectNode(pattern, orientation, is_dynamic)`, with
    `pattern` one of 8 trigrams (+ neutral).
  * Composition nodes are `ParallelNode` (average), `SerialNode` (split into
    strips with H or V orientation, then mean), `PlanNode` (template ->
    content replacement), and `BodyNode` (a plan inside a window with
    position, extent and velocity, moved by its first dynamic trigram).
* **Execution, recognition direction:** `evaluate()` / `step()` scan
  input fields and give scalar activations.
* **Execution, generative direction:** `render_output()` draws the
  imagined subgraph as a 2-D field (noisy-OR; yin bands at 0.5,
  "unknown").
* **Time:** `step()` keeps gates and moves `BodyNode`s by constant
  velocity, clamped at the field border. There is no oscillation or
  periodicity primitive. So the current language can express *where body
  mass is* (layout over bands and strips), *which windows translate in
  which direction*, and *gating*, but not *rhythm*. This is the main
  expressive limit of the proof of concept, and I state it rather than
  quietly extend the language.
* **Intermediate outputs that need no pixels:** per-node activations,
  gates, `body_states()` (position/velocity/extent per step), and coarse
  fields (`render_output`, `derived_fields`). There is no skeleton or joint
  output.
* **One gap:** `render_output()` draws a `BodyNode`'s content over its
  parent's whole region and ignores the body's position. A body's motion
  is therefore invisible in rendered output. I add an opt-in flag
  `render_output(..., place_bodies=True)`. It draws each body in the same
  window that `evaluate()` already carves out. The default is unchanged, so
  no existing semantics change.

## 2. Shared human/LoT representation

The LoT's native currency is a `ReceptiveField`: a coarse 2-D salience map,
optionally with a motion channel. Both sides are therefore reduced to a
**sequence of T canonical body-centred fields**, a tensor `Y` of shape
`T x H x W` (default 16 x 12 x 12):

* `human_trajectory(v)`: video -> MediaPipe Pose (33 landmarks, 2-D +
  3-D world, visibility) -> keep frames where the whole body is visible ->
  canonical coordinates:
  * x is relative to the per-frame torso centre, which removes walking
    translation;
  * y is relative to the video's median torso height, which *keeps*
    bounces and drops;
  * both are scaled by the median body height.

  Then resample to T frames and rasterise the **library's own full
  skeleton** (all 35 connections at one thickness, torso filled) into
  H x W cell-coverage values in [0, 1]. No joint is weighted by hand.
* `lot_trajectory(r)`: compile program `r` into an imagined root
  `ParallelNode(O, r)`, `reset_gates()`, then for t = 0..T-1:
  `render_output(shape=(H, W), place_bodies=True)` followed by `step()`.

Alignment:
* Both sides use the same canonical window, which is fixed in body-height
  units.
* Both use the same T. Time is normalised by linear resampling of the
  valid segment; DTW is available as an alternative.
* Both use the same [0, 1] scale.

The numeric pose tensor (raw, world and canonical landmarks, the centre
trajectory, and the shoulder-line yaw as orientation) is saved alongside.
The kinematic baselines use it.

## 3. Simplest likelihood

`log p(v | r) = -beta * D(Y_v, Y_r)`, where `D` is the mean squared cell
difference after time normalisation (or DTW). Every cell and every frame
has equal weight. `beta` is configurable. Cells are strongly correlated,
so I use `beta` (effectively the number of independent observations / 2
sigma^2) rather than a per-cell sigma, which would make the posterior
absurdly peaked. For several actors of one manner,
`log p(v_1..N | r) = sum_a log p(v_a | r)` (Stage 5).

## 4. Simplest prior

`log p(r) = -lambda * L(r)`, where L counts nodes (leaves and composition
nodes). Body parameters each cost a configurable extra weight. Hard limits
(max depth, max nodes) bound the space, so the prior is proper. `lambda` is
configurable.

## 5. MH proposals (all valid under the grammar)

The grammar used for search is the generative fragment of the LoT:

```
R    -> Leaf | Par(R+) | Ser_{H|V}(R+) | Plan(R, R+) | Body_{pos,ext,speed}(R+)
Leaf -> Obj(trigram, H|V, dynamic?)
```

Body position, extent and speed come from small discrete grids.

| Move | Hastings correction |
| --- | --- |
| regenerate a subtree from the grammar's generator (covers insert, delete, replace) | `q(old|new)/q(new|old)` from the generator's exact log-probability and node counts |
| change a leaf attribute (trigram, orientation, dynamic) | symmetric |
| change a composition type (Par <-> Ser_H <-> Ser_V <-> Plan) | symmetric |
| change one body parameter | symmetric |
| swap two siblings | symmetric |

Dashed edges and observed (scanning) roots are left out of the first
grammar. They have no effect on the rendered output when no input field is
present.

## 6. Bottlenecks

* **Rollouts.** Each program rollout is T render passes in Python (~1-20
  ms). This is handled with:
  * an LRU cache keyed by the canonical program string;
  * small fields (12 x 12);
  * T = 16;
  * size and depth limits;
  * one process per video.
* **Pose estimation.** It runs once and is cached: ~28 fps on CPU, about 6
  min for all videos.
* **MCMC mixing.** Mixing over discrete trees is the real risk. Repeated
  chains plus R-hat-like comparisons of log-posterior and of MAP agreement
  are reported.

## 7. Smallest end-to-end prototype

1. Preprocess the 26 Study-3 exemplars into `data/processed/` and
   `.npz` files, plus stick-figure previews.
2. Run 2 MH chains per exemplar and keep thinned posterior samples.
3. Compute program similarity:
   * flat bag of primitives;
   * unordered tree;
   * ordered tree;
   * full labelled tree.
   Each is the posterior expectation over sample pairs, with MAP as a
   diagnostic.
4. Build the human 26 x 26 matrix from Study 3. Compare by Spearman over
   the 325 pairs, a Mantel permutation test over labels, and a bootstrap
   over participants.
5. Kinematic baselines on the same preprocessed data: field-sequence
   distance and joint-trajectory DTW. Then regress
   `human ~ kinematic + LoT`, with a permutation test for the LoT increment.
6. Ablations are grammar and similarity variants chosen in config (no
   hierarchy, no order, collapsed composition, primitive families removed).
   All are scored the same way.

Nothing in steps 1-3 sees the human matrix. The human data enter only at
step 4.

## Revision 2 (after the first prototype)

| Change | Why |
| --- | --- |
| **Dashed edges are part of the search space** (`Program = tree + dashed edges`, targets are leaves) | In the existing LoT, a dashed loop through a `plan(VOID ..)` template alternates its content between frames under `step()`. That is how the LoT expresses rhythm. The first prototype's search could not propose dashed edges. The LoT's semantics are unchanged. |
| **Description-length prior** `L(r) = -log P_gen(tree) + edge-choice cost`, lambda selected | Coverage diagnostics showed the node-count prior with lambda = 0.5 put about 67% of prior mass at the size limits. The number of programs grows faster with size than exp(-0.5 x nodes) shrinks them. |
| **Parallel tempering**, with an adaptive ladder (the hottest replica has total likelihood weight `hot_beta`) | Mixing: R-hat up to 15 in the first prototype. |
| **Local tree moves** wrap/unwrap and add/remove child, each with an exact reverse; dashed edges follow path shifts | Synthetic recovery: search success went from 18/24 (subtree regeneration only) to 21/24 at 20k iterations, and to 23/24 at 40k. |
| **Manner-level inference over actors**, split 15 train / 5 validation / 5 test (25 actors: 01-13 F/M, 10F withdrew) | Infers a manner program `p(r_m | v_m1..v_mN)`, chooses hyperparameters on held-out actors, and keeps an untouched test set. |
| **Hyperparameters are chosen by validation recognition** (expected distance from a validation video to each manner's posterior renderings), never by human data | This criterion does not depend on beta and directly measures generalisation across actors. |
| **The test split is sealed in code** (`heldout.evaluate` and `analysis.run` refuse test actors without an explicit unseal flag) | The test set is used once, after the configuration is frozen. |
| **Speed:** scalar clamps instead of `np.clip`, a plain-class `Node` (not an ABC), periodic-state detection in rollouts, single-threaded BLAS per worker | About 8x faster sampling; fixed memory exhaustion from 800 MB of OpenBLAS commit per process. All behaviour-preserving (old-vs-new fuzz comparison and brute-force rollout checks). |

Verification, all without human data: exact MH and tempering tests against
enumerated posteriors (every move type), exhaustive reachability on a small
grammar, family coverage on the configured grammar, and synthetic program
recovery (`grace_lot/inference/recovery.py`).
