# Probabilistic semantics: one reading throughout (design note, 2026-10-18)

Goal: every value the LoT produces means the same kind of thing, a probability, from the program down to
the comparison with data. This note fixes the reading; the scorer that implements it is a diagnostic
(scratch) until it is checked, then a v7 option behind a flag. Older configs and results stay pinned.

## 1. What values mean

| Quantity | Meaning |
| --- | --- |
| a cell value v ∈ [0, 1] | p(yang) at that cell: 0 known empty (yin), 1 known figure (yang), 0.5 unknown |
| `neutral` (static) | p(yang) = ½ everywhere in its bands: "I don't know whether yin or yang here" |
| a gate p | a mixture weight: with probability p the gated thing is present or on |
| `void*` | p(motion) = 0: "I know there is no motion in this field" |
| `neutral*` | p(motion) = ½: "I don't know whether there is motion in this field" |
| `universal*` | p(motion) = 1, direction unknown: "there is motion here, I don't know which way" |

The language side already follows this reading:
- rendering paints yang and yin by noisy-OR (independent causes);
- a 0.5 cell paints nothing;
- template gates mix the oriented and unchanged readings with weights p and 1 − p;
- a trigram's match 1 − |observed − expected| is the Bernoulli probability of a binary observed cell.

The departures are all in the comparison with data, which is what this note changes.

## 2. A program defines a distribution over boards
A program, together with its *realization*, defines a distribution over binary boards for every frame.

**Discrete unknowns** are latent variables, enumerated exactly.
- `neutral*` is still or moving, ½ each.
- There is **one draw per unit**: all `neutral*` leaves inside the same field or body share it. A leaf outside
  any unit has its own draw.
- At most 2 such draws per program, so at most 4 combinations; a grammar limit would enforce this.
- A draw is decided once per video, not per tick, so a region either moves throughout or stays still.

**Continuous and many-valued unknowns** are summarised by moments within each combination:
- whether each cell with 0 < v < 1 is yang;
- the directions of `universal*` motion.

**Cells keep their state over time.** A cell's draw is coupled across frames: each cell gets one uniform
number u per realization, and it is yang at frame t if u < v_t. An unknown region therefore doesn't flicker;
it changes only where its probability changes. Independent draws per frame would invent frame-to-frame
change that no program asserts.

## 3. The likelihood
For each combination c of the discrete unknowns, draw D realizations with fixed seeds. The same seeds are
used for every program, so each program gets a fixed score: common random numbers keep MH exact with
respect to the score, and keep caching valid.

For each realization, sample the binary boards and compute the part features. Their mean μ_c and variance
σ²_c, per feature, frame and output channel, summarise the prediction. Frame-to-frame changes are
summarised the same way, from the per-draw differences.

Per video v, the matched cost of output k against part p becomes a Gaussian cost with the prediction's own
variance added to a baseline noise s0²:

    C[k, p] = mean_tf [ (μ − o)² · s0² / (s0² + σ²)  +  s0² · log(1 + σ² / s0²) ]   (+ w × the same for changes)

The baseline noise comes from β: s0² = P·T·F / (2β) for frames, and P·(T−1)·F / (2βw) for changes. With
σ² = 0 the cost is exactly today's squared distance. **A certain program, with every cell 0 or 1, no gate
strictly between 0 and 1 and no unknown motion, scores exactly as now.**

The matching stays one Hungarian assignment per video, now per combination. Mixing the combinations:

    log p(v | program) = log Σ_c (1 / |C|) · exp(−β · D_c(v))

Each video takes the combination that fits it, minus log |C|: the price of not committing.

## 4. What this changes, and a consequence to be aware of
1. **No salience threshold.** Today a cell counts as figure only if v ≥ 0.75. Here every cell contributes
   with its probability.
2. **`universal*` is no longer a fixed random path.** Its direction uncertainty enters the variance, so no
   program depends on a lucky draw.
3. **The unknown background becomes uncertain.** With the unknown background, every undrawn cell is 0.5:
   it might be figure. A program that draws a small limb on an unknown background therefore predicts
   "limb plus possible figure anywhere". The program is honestly uncertain, so its features have high
   variance; its score is tolerant but low. Drawing explicit yin ("I know it's empty") commits and, if right,
   scores better. The prior then decides whether the extra yin is worth its cost. This is the intended
   reading, but it changes which programs win, so the diagnostic must show its effect before anything is
   committed.

## 5. Kept as they are
- **Rendering, gates, noisy-OR, and the engine's dynamics** apart from realization seeding: unchanged.
- **Observer yin bands as masks** ("don't look here"). A deliberate exception, kept unless revisited.
- **Observers' matching** stays 1 − |band mean − expected|. The exact per-cell Bernoulli agreement is a
  later, separate change, because it alters perception, not just scoring.

## 6. Diagnostic plan
1. Check that certain programs reproduce the current orient8 scores.
2. Score triples: the same program with `void*`, `neutral*` or `universal*` in one slot, on training videos.
   The expected pattern: `void*` wins on still data, `universal*` on moving data, `neutral*` on mixed data,
   with the price of hedging about log 2.
3. Run the real vs shuffled-time difference-in-differences for the stochastic members.
4. Measure the cost per program.

## 7. Diagnostic results (2026-10-18, scratch scorer, training actors only)
1. **Exactness.**
   - With zero variance, the scorer reproduces the current matched orient8 likelihood to within 7e-6
     (float32 rounding).
   - The batched orient8 of sampled boards matches the existing per-board code to within 1e-14. The
     single-cell convention, a lone point read as vertical, is reproduced.
   - Realization 0 reproduces today's rollouts bit for bit.
2. **Moments fail for unknown direction.** Averaging `universal*` over random paths predicts "blurred
   stillness". At the noise level β implies (s0² ≈ 1.5 per feature, against prediction variances of about
   0.001), the variance barely counts. On synthetic videos that genuinely move, `void*` beat `universal*`
   by about 90 nats, and `neutral*` only ever paid its log 2.
   **Change:** unknown direction is a discrete latent, enumerated exactly:
   - `universal*`: one way or the other along its axis, ½ each;
   - `neutral*`: still ½, one way ¼, the other ¼.
   
   Moments are kept only for independent cell-level uncertainty (unknown yin or yang).
3. **Synthetic check of the discrete version** (15 videos per truth):

   | Truth | `void*` | `neutral*` | `universal*` |
   | --- | --- | --- | --- |
   | all still | **0.0** | -10.4 | -274.9 |
   | half drifting up, half down | -897.9 | -475.8 | **-465.4** |
   | mixed: 7 still, 8 moving | -478.9 | **-258.4** | -376.3 |

   The price of hedging is exactly log 2 per video: 10.40 = 15 log 2, both against `void*` on still data
   and against `universal*` on moving data.
4. **Real videos** (the triple (parallel A M out) in every MAP slot; orient8 + cycle2; 3 shuffles):

   | Manner | Winner: `void*` / `neutral*` / `universal*` | Gap over `void*`: `neutral*` / `universal*` | Difference-in-differences vs shuffled: `neutral*` / `universal*` |
   | --- | --- | --- | --- |
   | bowing | 0.35 / 0.01 / 0.65 | +29.6 / +28.4 | -0.05 / +0.30 |
   | crisscrossing | 0.41 / 0.02 / 0.57 | +20.3 / +11.6 | -0.04 / -0.08 |
   | jumping | 0.39 / 0.00 / 0.61 | +29.3 / +25.8 | +0.25 / +0.28 |
   | swinging | 0.38 / 0.00 / 0.62 | +32.1 / +27.4 | +0.34 / +0.32 |

   The motion members gain 12-32 nats over `void*`, but **equally on shuffled frames** (difference in
   differences about 0). So the gain is about appearance, not temporal order.

   A likely source: under the probabilistic reading, cells vacated by moving mass become unknown (0.5), so
   in expectation they are half figure. A drifting band thereby spreads expected appearance over its path,
   whatever the order of frames.

**Reading.** The scorer does what the semantics says: certain programs are unchanged, and unknown motion
is credited, at the price of log 2, exactly when the data vary in the predicted way. It doesn't, by
itself, create a temporal fit to the real videos. The crude clips still don't move like limbs, as found in
the dynamics audit.

## 8. Implemented for v7 (behind flags; every older config pins them off)
**Engine:** `lot.probabilistic_motion`, as `embodied_lot.PROBABILISTIC_MOTION`.
- Every imagined `universal*` / `neutral*` leaf belongs to a unit: its innermost field or body node
  (template copies share it), else the leaf itself.
- `ConceptGraph.motion_states` fixes each unit's state for a rollout:
  - `"a"` / `"b"`: a bulk push one way or the other along the leaf's axis, acting as `top*` / `bottom*`;
  - `"still"`: no orders.
- Unset units default to `"a"` (universal) and `"still"` (neutral).
- `ConceptGraph.unknown_motion_units()` lists the units.

**Adapter:**
- `motion_combinations(program, size)` enumerates the joint states with their prior weights: universal a/b at
  ½ each; neutral still ½, a ¼, b ¼.
- `rollout(..., motion_states=...)` renders one combination.

**Likelihood:** `ProbabilisticScanLikelihood` (`inference.likelihood = "probabilistic"`, `inference.cell_draws`).
- It uses the current detector interface (`parts_scan`) on sampled binary boards, then the moment-matched
  cost, one matching per video and combination, and the weighted mixture.
- With every cell 0 or 1 and no unknown motion, it equals `MatchedScanLikelihood` (tested).

**Prior and grammar:**
- `lot.max_motion_units` (2 in v7) caps the number of units, via `program.motion_units`. So there are at most
  9 combinations.
- `neutral*` is a legal leaf; the static-only rule of the earlier draft is gone.

**Tests:**
- states and their weights;
- unit counting and the cap;
- MH exactness with dynamic `neutral` and the cap;
- equality with the matched likelihood for certain programs;
- the triple test: videos drawn from a program's own distribution give `void*` on still data, `universal*` on
  data moving either way, `neutral*` on a mix, with hedging at exactly log 2 per video.

With the flags off, the v6, v3 and legacy-fuzz rollouts are bit-identical. The gameboard explainer pins
both v6 flags.

### Open point: motion leaves knowledge behind
In the engine, cells vacated by moving mass become unknown (0.5). Under this reading they "might be
figure". So a moving program predicts possible figure all along its trail: 64% of cells in the test program.
Its videos are then very uncertain, and on real data, where parts are sparse, motion is penalised for that
uncertainty. Whether vacated cells should instead stay known empty, or return to what the tree draws there,
is a semantics decision still to make.

**Not yet switched over:** the held-out evaluation and the inspection pages still use the matched likelihood.

## 9. 0.5 means "no claim", with a shared background (decided 2026-10-18)
**The decision.** A cell at 0.5 is where the concept makes no claim. It is predicted by a **background base
rate shared by all programs**, not by a coin flip.
- The background is `parts.occupancy_background`: the per-cell mean part occupancy over every training video
  of every manner. It looks like an average body silhouette (mean 0.026, peak 0.31 in the central columns).
- In general p(figure) = |2v − 1| · [v > ½] + (1 − |2v − 1|) · b, which is `likelihood.cell_probability`.
  Claims (0 or 1) are unaffected, and partial values interpolate.
- Config: `inference.background = "occupancy"` (v7).

**What it implies.** Concepts are **deviations from the average body**. A program that claims nothing scores
like the background, and that is a strong baseline. On swinging and crisscrossing, an all-abstaining
6-output program scores about −423 and −489, against −1280 to −1460 for single-band programs. So a claim
has to beat the average body to pay off.

**Vacated cells are known empty.** With abstention this attractive, letting moving mass leave "unknown"
behind turned motion into a way of withdrawing claims. `universal*` then beat `void*` by about +79 nats on
every manner, with a difference-in-differences of about 0, so the gain wasn't temporal.

Under `probabilistic_motion`, cells that moving mass leaves behind therefore read as **known empty** (yin).
Only contradictions, such as figure piled onto empty at an edge, stay at 0.5.

Real-data triples afterwards (training actors, current sampling, 3 shuffles):

| Manner | `universal*` wins | Gap over `void*`: `neutral*` / `universal*` | Difference-in-differences: `neutral*` / `universal*` |
| --- | --- | --- | --- |
| bowing | 0.62 | +11.8 / +4.9 | +0.69 / +1.06 |
| crisscrossing | 0.66 | +13.3 / +7.1 | +0.40 / +0.56 |
| jumping | 0.62 | +13.7 / +7.3 | +1.12 / +1.61 |
| swinging | 0.62 | +11.6 / +4.0 | +0.77 / +1.20 |

The motion bonus is now small, and the difference-in-differences is positive everywhere: some of it reflects
temporal order. With the flags off, v6, v3 and the legacy fuzz are bit-identical, and the explainer is exact
with its v6 flags.
