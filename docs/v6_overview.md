# Embodied-LoT x GRACE: where the model stands (v6, 2026-10-17)

A briefing for someone who last saw the project around v3.

## 1. Purpose (unchanged)
Embodied-LoT is a computational model of grounded conceptual representation. Concepts are programs in
a "language of thought" (LoT) whose operators are perceptual and attentional: bands, strips, templates,
bodies and motion on a small grid. The GRACE pipeline (`grace_lot/`) infers, for each of 26 locomotion
manners (jumping, swinging, bowing, ...), a posterior over programs that explain that manner's videos.

- **Data:** 25 actors, split 15 train / 5 validation / 5 test. The test split is sealed: nothing is
  inspected or scored on it.
- **Clips:** each clip becomes 16 frames.
- **Learner:** the GRACE learner is the only one now. The old MCMC and DreamCoder learners are retired
  to `legacy/learners/`.

## 2. What changed since v3

| Version | Interface | Main change |
| --- | --- | --- |
| v3 | occupancy (one 12x12 field per frame) | gameboard grid-world engine; parametric bodies (position, extent, speed); one output |
| v4/v5 | body parts (scanner activations) | anonymous outputs matched to skeleton parts; motion kinds; bodies without parameters; lenient limits |
| **v6** | body parts | outputs and inputs are window leaves; programs are forests; frame-to-frame change term in the likelihood |

v3 results (frozen, λ=1, β=250, the only converged setting): validation top-1 recognition 0.308,
against 0.785 for a kinematic prototype and 0.615 for 1-NN. The fits were static silhouettes. That is
what v4 to v6 try to fix.

## 3. The language in v6

### Programs are trees, and only trees
"The semantics is purely defined by the tree": programs carry no numeric parameters. Sizes are
relative to the tree. Nothing has an absolute size until it is evaluated on a board.

### Leaves
- **Trigrams:** 8 kinds (top, center, upper, bottom, periphery, lower, universal, void) x 2
  orientations (H/V).
  - Each is static or dynamic. A trailing `*` in printed programs marks dynamic.
  - Each draws yang (figure) and yin bands.
- **`out` window:** an output.
  - It delivers only the region it occupies in its tree, in its own frame.
  - That region is stretched to 12x12 when the output is produced.
  - An output sharing a parallel parent with the rest of its tree delivers that whole region.
  - It may only be content: never a body's or plan's template (`content_windows`).
- **`in` window:** the symmetric input leaf. GRACE needs none.

### Compositions
- **parallel:** children share the same region. Values are averaged; window leaves are excluded from
  the averages.
- **serial_h / serial_v:** strips.
- **plan:** template(s) + content.
- **body:** a movable object that fills its tree region.

### Dashed edges
A dashed edge copies the source's activation into the target's **gate**. Dashed edges are the only
links between trees.

### Forests
A program is one or more trees, written side by side. The notation has no "outputs" mother node.
- Spaces drawn by separate trees stay separate unless the trees share a parallel node.
- The prior requires 1 to 6 `out` leaves.
- Outputs are matched to body parts by likelihood, not declared.

### Roles
- **Observers** perceive and are not drawn. They are dashed sources that are not dashed targets, and
  their subtrees.
- **Imagined** nodes are drawn.
- An observer perceives its own tree's space.

### Values
- Unknown background is 0.5. Imagined yin is 0. An observer's yin bands are masks.
- A gate counts as "on" only strictly above 0.5. Dashed targets start at 0.5 (unknown).

### Dynamic templates orient their content
- A dynamic template rotates its content instead of selecting part of the field. Nested turns add up.
- A serial template gives one oriented slot per strip. Other templates face the sum of their dynamic
  leaves' directions.
- Gate p mixes the oriented and unoriented readings: p oriented, 1-p unchanged.
- Dynamic leaves inside templates orient; they never issue motion.

## 4. The engine ("gameboard": a discrete, board-game-like grid world)
A program is simulated for 16 ticks on a 12x12 board. Each tick:
1. assign roles;
2. scan;
3. compute node views (regions and turned frames);
4. observers perceive last tick's board;
5. motion leaves issue orders;
6. dashed gates propagate (one tick delay, for leaf and group targets alike);
7. bodies and mass move, at most one cell per tick per order;
8. each tree's space is redrawn;
9. each `out` window is read.

### When motion fires
- **Speed** = the motion leaf's gate x the gates of its groups.
- **Rule B:** a switched-on motion leaf orders every body or mass cell that enters its own region.
  - A body gets an order on entering, including re-entry after leaving.
  - A body that stays inside gets no new orders. Only presence is remembered.
- **Direction:** motion runs away from the figure side on both axes.

### Motion kinds
Motion moves drawn mass, whose displacement persists, as well as bodies.

| Leaves | Kind | Behaviour |
| --- | --- | --- |
| top / bottom | bulk | the region moves as one block |
| upper / lower | gradient | layers peel off from the leading edge, one layer per tick; diffusion for mass, a queue for bodies |
| center / periphery | out / in | the two halves move apart or together by half their own size |
| universal | random | random pushes, seeded deterministically (the same video every time) |

### Flags
- Every semantic change sits behind a default-on flag, set by `apply_semantics(cfg["lot"])`.
- Each older config and frozen result pins the behaviour it was run with. A missing key means legacy
  behaviour.

## 5. Likelihood (body-part interface, `parts_scan`)
- **Scanner:** 6 soft detectors per frame.
  - Left / centre / right and upper / centre / lower: sigmoid receptive fields with boundaries at 1/3
    and 2/3 and softness 0.05.
  - Activation = the maximum response over salient points.
  - Observed salient points are skeleton points: joints plus 4 samples along each edge.
  - LoT salient points are rendered cells with value >= 0.75.
- **Parts:** 6 (head, torso, left/right arm, left/right leg).
- **Matching:** per video, the K outputs are matched once to the 6 parts for the whole sequence by
  Hungarian assignment.
  - Costs are padded with empty channels, so unmatched parts and extra outputs pay their energy.
  - The mapping is never re-chosen per frame, so alternation can't be relabelled away.
- **Cost** = mean squared activation difference over frames and detectors, plus w x the same on
  frame-to-frame changes (w = `temporal_weight`, 1.0 in v6).
- **log p(videos | program)** = -β x the sum over the 15 training videos of the matched distance.

## 6. Prior and sampler
- **Prior:** description length, made of the grammar's log-probability and dashed_cost per dashed edge,
  weighted by λ.
  - Lenient v6 limits: 24 nodes, depth 5, 5 dashed edges, dashed_cost 0.5.
- **Sampler:** custom Metropolis-Hastings over trees with exact Hastings corrections.
  - Moves: regenerate, change leaf, change composition, swap, add/remove dashed edge, wrap/unwrap,
    add/remove child.
  - Parallel tempering: 4 chains x 12 replicas x 40k iterations per manner, burn-in 10k, thin 30.
  - There are no gradients and no NUTS: the space is discrete.
- **Selection:** a grid λ{0.25, 0.5, 1} x β{250, 1000} x w{0, 1} is scored by validation-actor
  recognition. Only converged settings count (R-hat <= 1.1).

## 7. Speed (wall clock, 24-core CPU, GPU unused)
- **v6 MH step:** 27 ms before the optimisations, about 11 to 14 ms after. All optimisations are
  bit-for-bit exact.
- **Rollout:** 23.4 to 6.5 ms per program. Changes: cached graph metadata, memoised views, per-tree
  dirty redraws, scalar small reductions.
- **Likelihood:** 0.36 to 0.18 ms per program (a factorised detector scan). It is now about 5% of a
  step; rollouts are about 94%.
- **Not yet done:** Numba, GPU, fewer replicas, early stopping.
- A GPU version would need a lockstep, tensorised engine: the programs are heterogeneous trees.

## 8. Latest result: the 4-manner v6 diagnostic (10k iterations, training actors only)
The best programs use all 6 outputs, each matched to a different part. But their outputs are
**static**: detector change is zero. No setting converged (R-hat 1.24 to 1.73).

A per-tick trace of every motion leaf found three structural reasons:
- **A.** Most dynamic leaves sit in templates, so they orient content and never move it. This is by
  design.
- **B.** The motion is gated by a **lone observer tree**. An observer isn't drawn, so its own space is
  all unknown, and it reads 0.5 forever. Its targets' gates stay at 0.5, and a gate is on only strictly
  above 0.5, so the motion never fires.
- **C.** The motion leaf is on, but its region holds no drawn mass and no body. A motion leaf draws
  nothing itself.

### Open design question (B): what should a lone observer tree perceive?
1. **Nothing (current).** The observer is a constant. Optionally, the grammar could forbid observers
   that share no tree with drawn content.
2. **The shared board:** the overlay of all trees.
3. **The spaces of the trees it gates.** Dashed edges stay the only link between trees, but are read
   both ways: perception goes from target to source, gating from source to target. This allows
   closed-loop oscillation (motion moves mass, an observer sees it, a counter-motion is gated) without
   a timer.
4. **No perception:** the observer's targets are treated as ungated.

## 9. Standing constraints
- The test split stays sealed.
- During diagnostics, don't alter the model, grammar, likelihood, prior or search settings based on
  what the results show.
- Don't tune scanner softness on human similarity data.
- Keep anatomy outside the LoT.
- Never rematch outputs to parts frame by frame.
- Keep the mechanics discrete, like a board game rather than a pixel engine.
- Keep the code concise and easy to revise. Every semantic change goes behind a flag, with old
  results pinned.
