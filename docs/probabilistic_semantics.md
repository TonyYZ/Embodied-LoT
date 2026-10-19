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

The same holds for bodies (added later, same flag): a body or group instance drawn away from the region it
receives leaves the cells it would claim there, but no longer claims, as known empty, as strongly as they
were claimed. The caterpillar demo's motion is unchanged; its clock observer now rests at 0 between pulses
instead of 0.25-0.5.

Later still (same flag), the trail became a full trail: once a body has moved, every cell it has claimed so
far, at home or anywhere it has passed through since, reads as known empty when it no longer claims it, at
the strongest claim it made there. The renderer keeps each moved body's (or instance's) strongest claim per
cell until `reset_gates()`. Before, only the home region was remembered, so cells a body had merely crossed
went back to "no claim".

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

## 10. Diagnostic: noise from data, and informativity (scratch, 2026-10-18)
**The question.** Does the likelihood reward informative claims? It should do so through normalisation: a
sharp, correct claim gets high density, a vague one less, a wrong sharp one little.

**The noise.** Today's β = 250 implies an observation noise variance of about 1.5 per detector value (P·T·F / 2β).
Estimated from data, the variance between actors doing the same manner (pooled over manners, frames and parts)
is 0.0002-0.012 per detector: **100-7000 times smaller**. Real parts vary across all training videos with
variance 0.0005-0.16 per detector. That is what "no claim" should predict.

**The test.** A normalised Gaussian likelihood (data-estimated noise plus the prediction's own variance;
β = 1; a part with no output is predicted by the background) is compared with the current scorer on
feature-level predictions:
- *right*: the manner's mean, leaving each actor out;
- *vague right*: the same mean with the background's spread;
- *abstain*: the background;
- *wrong*: another manner's mean.

| Manner | Gaussian: right / vague / abstain / wrong | Current: right / abstain / wrong |
| --- | --- | --- |
| bowing | 3283 / **8744** / 2553 / -22157 | -109 / -285 / -171 |
| crisscrossing | **14987** / 12297 / 9804 / 9838 | -150 / -335 / -174 |
| jumping | **11815** / 11122 / 8836 / 9608 | -196 / -386 / -220 |
| swinging | **21171** / 13885 / 11854 / -2051 | -76 / -267 / -142 |

**Reading the test.**
- With data-estimated noise the likelihood rewards informativity: right > vague > abstain in 3 of 4
  manners, and sharp wrong claims are heavily punished.
- In bowing, the vague right claim wins, because bowing actors vary more than the pooled noise allows. A
  claim more precise than reality is punished, which is correct. It does suggest per-manner noise or heavier
  tails.
- The current scorer rewards even a wrong manner's mean over abstaining. With noise that large, specificity
  is never costly.

**Programs under the Gaussian likelihood** (unclaimed cells filled from real part trajectories):

| Program | bowing | crisscrossing | jumping | swinging |
| --- | --- | --- | --- | --- |
| background only | 2553 | 9804 | 8836 | 11854 |
| claims nothing, 6 outputs (filled from real parts) | -7971 | 6744 | 5181 | 9088 |
| diag4 MAP, as certain claims | -21654 | 3411 | -4372 | 8567 |
| a still band | -21575 | -14037 | -18773 | -14269 |
| a central column, 6 outputs | -85276 | -66142 | -69229 | -59921 |

**Reading the programs.**
- **No program beats the background.** The diag4 MAPs come closest (swinging, crisscrossing).
- **There is information to capture:** the right manner's mean does beat the background in 3 of 4 manners.
  The current programs just don't capture it.
- **The two background versions disagree.** The "claims nothing" program is filled from rasterised parts,
  and it scores below the analytic background built from point-based part features. The rasterised fill and
  the point-based observation features don't match exactly, and that should be fixed before relying on it.

**Before any change is committed:**
1. a background consistent with the observation features;
2. per-manner (or heavier-tailed) noise;
3. β as a pure temperature, calibrated so the posterior isn't absurdly peaked: at β = 1 with this noise,
   the log-likelihoods are in the thousands;
4. a search under the new likelihood, to see whether programs can beat the average body.

### 10b. Second pass: one feature path, per-manner noise or heavy tails, β as temperature

**What changed** (scratch: `informativity2.py`):
- **One feature path for both sides.** Each observed part is first quantised to the LoT's 12 × 12 cells (a
  cell is occupied if any of the part's skeleton points falls in it). Its detector features then come from the
  same `scan_field` that scores LoT outputs. Unclaimed cells are filled from these real part boards.
- **Check:** with enough fill draws, a program that claims nothing now scores the same as the analytic
  background (within 4-6 per manner, from time-varying moments). With 24 draws it did not: the Monte Carlo
  variance of detector 2 (a near-constant detector, variance 0.0003) came out as 0, which spuriously
  sharpened or broke the prediction. A real scorer needs **a variance floor** or many draws.
- **Noise:** either per manner (across-actor variance within the manner), or pooled with Student-t tails
  (ν = 4).
- **β is a pure temperature** on the log density. It does not change any ordering. Numbers below use
  β = 1/(T·F) = 1/96, i.e. the mean log density per frame and detector.

**Feature-level test** (right / vague right / abstain / wrong manner; summed over 15 training videos):

| Manner | per-manner Gaussian | pooled Student-t |
| --- | --- | --- |
| bowing | **135.0** / 101.4 / 70.9 / 80.6 | **165.4** / 108.2 / 90.2 / 146.7 |
| crisscrossing | **252.7** / 136.2 / 117.2 / 233.1 | **181.9** / 120.0 / 100.8 / 169.9 |
| jumping | **234.1** / 122.9 / 104.2 / 216.6 | **168.0** / 112.1 / 92.5 / 158.7 |
| swinging | **282.2** / 147.3 / 128.7 / -25303 | **207.6** / 129.3 / 112.1 / 153.2 |

- **Both rank right first in every manner,** and right > vague > abstain everywhere. Bowing is fixed:
  per-manner noise gives bowing its own, larger spread.
- **"Wrong" here is the next manner's mean,** and neighbouring manners are close (crisscrossing → jumping,
  jumping → swinging), so a sharp wrong claim can still beat abstaining. Student-t forgives wrong claims
  most: they come second in every manner.
- **Swinging's −25303 is the variance floor again.** Swinging's detector 2 has zero across-actor variance, so a
  tiny mismatch is priced as impossible. A floor (e.g. a fraction of the background variance) is needed.

**Programs** (unclaimed cells filled from real part boards, 200 draws; diag4 MAP as certain claims):

| Program | bowing | crisscrossing | jumping | swinging |
| --- | --- | --- | --- | --- |
| *per-manner Gaussian* | | | | |
| background only | 70.9 | 117.2 | 104.2 | 128.7 |
| claims nothing, 6 outputs | 74.7 | 113.1 | 100.4 | 125.2 |
| diag4 MAP | 33.1 | **120.6** | **147.9** | **156.2** |
| a still band | -160.0 | -209.5 | -172.7 | -85.4 |
| a central column, 6 outputs | -358.7 | -241.9 | -185.6 | -729.4 |
| *pooled Student-t* | | | | |
| background only | 90.2 | 100.8 | 92.5 | 112.1 |
| claims nothing, 6 outputs | 88.0 | 95.1 | 86.9 | 106.7 |
| diag4 MAP | **115.1** | **126.0** | **124.5** | **165.1** |
| a still band | 65.0 | 74.9 | 62.7 | 88.2 |
| a central column, 6 outputs | 1.5 | 7.7 | -10.2 | 14.2 |

- **With one feature path, real programs can now beat the average body.** The diag4 MAPs beat the
  background in 3 of 4 manners under the per-manner Gaussian (bowing is the exception), and in all 4 under
  Student-t. Crude programs (a still band, a central column) lose under both.
- **Scale.** At β = 1/(T·F), a good program gains about 2-4 nats per video over the background, i.e. some
  30-60 over a manner's 15 videos. That is the same order as prior differences between programs, so the
  posterior is no longer absurdly peaked. This is a starting point, not a fitted value.

**Recommendation for a v7 likelihood (not implemented):**
1. Observations go through the cells: quantise points to cells first. This changes the observation features,
   so it needs its own flag.
2. Per-manner Gaussian noise with a variance floor. It ranks right > vague > abstain everywhere and punishes
   wrong claims, while Student-t forgives them. Per-manner noise is estimated from the training actors the
   program is fitted to (empirical Bayes); it never touches the test split.
3. Background = pooled real part features of the training videos. Unclaimed cells filled by Monte Carlo
   need many draws or a variance floor.
4. β = 1/(T·F) as the initial temperature.
5. Then a search under the new likelihood.

## 11. Implemented for v7: the normalised likelihood (2026-10-18)

The recommendation of 10b, behind flags:
- **`interface.observation_cells = true`.** Parts are quantised to cells first (`scanner.part_boards`; points
  beyond the window go to the border cells), then scanned with `scan_field`, like LoT outputs.
- **`inference.likelihood = "gaussian"`** (`GaussianScanLikelihood`):
  - per state combination, `cell_draws` boards: claims with strength |2v − 1|, unclaimed cells filled from
    real part boards of the training videos (`scanner.feature_background`, cached with the pooled feature
    moments);
  - per-detector noise from the manner's own videos, floored at `noise_floor` × the background variance;
  - parts with no output predicted by the background;
  - one matching per video, temperature 1/(T·F) by default; combinations mixed by prior weight.
- **One fix found on real data:** if each output draws its own fills, the matching picks, for every part,
  whichever Monte Carlo variant of "no claim" happens to fit. "Claims nothing ×6" then beat the background
  by 8-27. With fills shared across outputs and 256 draws, it is within ±3 of the background (about 0.2 per
  video).

**Real-data check** (training actors, `build()` path, summed over 15 videos):

| Program | bowing | crisscrossing | jumping | swinging |
| --- | --- | --- | --- | --- |
| background only | 69.4 | 115.1 | 101.1 | 126.6 |
| claims nothing, 6 outputs | 67.6 | 117.3 | 103.8 | 126.9 |
| a moving band (universal*) | -80.5 | -108.0 | -52.6 | -66.2 |
| a still band | -146.7 | -210.1 | -154.2 | -91.2 |
| a central column, 6 outputs | -258.7 | -287.3 | -224.2 | -401.8 |

**Cost:**
- A short tempered chain (400 iterations × 4 replicas, jumping) took 100 s, about 60 ms per evaluation, so
  a full v7 run (40000 iterations × 12 replicas) is many hours per chain.
- `cell_draws` is the main lever: 64 draws are 4× cheaper but put "claims nothing" 3-5 below the
  background.

The MAP "distance" printed by `finish_manner` is now the matched NLL × temperature per part, which can be
negative.
