# GRACE x LoT: results

> **Status (2026-10-15).** Revision 2 (dashed-edge search, description-length prior, tempering, manner-level
> inference with a 15/5/5 actor split) has been run through **validation only**. The test actors are sealed.
> Full report: `results/grace_v2/validation_report.md`. The first-prototype section below is kept for the record;
> it predates the split and should not be compared with the validation numbers.

## Revision 2: validation summary

| | value |
| --- | --- |
| Synthetic recovery (15 actors, noise 0.05, 40k iterations) | search succeeds 23/24; MAP renders like the truth in 96% |
| Synthetic recovery with actor-like variability (noise 0.08, time shifts of up to 3 frames) | search succeeds 24/24; only 29% MAP-equivalent: exact programs are not identifiable |
| Grammar coverage (selected bounds) | 918/942 families visited, none with real prior mass missed |
| Top-ranked by validation recognition | lambda 1, beta 4000, bounds 14 nodes / depth 5 / 4 dashed edges (MCMC not converged) |
| **Frozen (user decision)** | **lambda 1, beta 250, default bounds**: the best converged setting (val top-1 0.35); human comparison not yet run for it |
| Validation recognition, top-1 (chance 0.04) | LoT 0.56, kinematic prototype 0.78, kinematic 1-NN 0.62 |
| Human similarity (training-actor posteriors) | LoT flat 0.22, LoT full 0.14, tree measures about 0; kinematic prototype 0.62; noise ceiling 0.90 |
| LoT beyond kinematics | no measure adds variance (dR2 <= 0.01) |
| Mixing | **only beta = 250 converges** (median R-hat about 1.06; best val top-1 0.35). The selected beta = 4000 setting does not (median 1.74; 1.64 even with a 24-replica ladder) |

### Reproducing revision 2 (validation only)

```
python -m grace_lot.grace.preprocess --all --workers 12                 # pose + trajectories, all 650 videos
python -m unittest test_grace_lot                                       # exactness and reachability checks
python -m grace_lot.inference.recovery --n 24                           # synthetic recovery
python -m grace_lot.inference.select --out results/grace_v2             # lambda/beta grid + variants, on validation actors
python -m grace_lot.analysis.run --results results/grace_v2/select/<chosen>    # human comparison (training-actor posteriors)
python -m grace_lot.analysis.validation_report --root results/grace_v2
python -m grace_lot.inference.freeze --root results/grace_v2           # -> configs/grace_frozen.toml
# NOT YET RUN (needs your go-ahead): fit on train+val with the frozen config, then
# python -m grace_lot.inference.run --config configs/grace_frozen.toml --split trainval
# python -m grace_lot.analysis.run --config configs/grace_frozen.toml --results results/grace_v2/final --unseal-test
```

## First prototype (exemplar videos, before the split)

Run: `configs/grace_poc.toml`, Study-3 exemplars (one video per manner), 4
chains x 40k iterations per manner. Full tables are in
`results/grace_poc/analysis/summary.md`.

## How to reproduce

```
pip install -r requirements-grace.txt          # plus data/models/pose_landmarker_heavy.task
python -m grace_lot.grace.preprocess --preview  # ~2 min
python -m grace_lot.inference.run               # ~4 min on 6 workers
python -m grace_lot.analysis.run                # ~2.5 min
```

## Headline numbers (Spearman rho with the human 26 x 26 matrix, 325 pairs)

| | rho | Mantel p | notes |
| --- | --- | --- | --- |
| human split-half noise ceiling | 0.895 | | |
| kinematic: joint DTW | 0.534 | <0.001 | |
| kinematic: field-sequence distance | 0.479 | <0.001 | the same representation the LoT is fitted to |
| kinematic: joint-velocity DTW | 0.212 | 0.02 | |
| LoT `full` (posterior expectation) | 0.157 | 0.046 | dR2 over the 3 kinematic baselines = 0.038, p = 0.026 |
| LoT `flat` | 0.116 | 0.11 | |
| LoT `tree_ordered` / `tree_unordered` | 0.07 / 0.06 | ~0.2 | |
| LoT `full`, MAP program only | 0.068 | 0.16 | |

Leave-one-manner-out cross-validated rho: kinematic only 0.515;
kinematic + LoT `full` 0.551.

## Reading these results honestly

* **There is a weak LoT signal beyond kinematics, and it is fragile.** The
  `full` measure adds a small, permutation-significant increment. However,
  20 model variants were tested, and p ~ 0.03-0.05 would not survive a
  multiple-comparison correction. Treat this as a pilot effect, not evidence.
* **Posterior averaging matters.** MAP-only similarity is at chance, while
  the posterior expectation is not. This is consistent with the posteriors
  being very diffuse: almost every one of the 4000 samples per manner is a
  different program.
* **The signal comes from body geometry, not composition.** The ablations
  that remove it are the ones that remove body parameters
  (`no_body_params`: rho 0.065) or merge serial orientations
  (`collapse_serial`: 0.069). Removing sibling order or the dynamic flag
  changes nothing or helps slightly. So what predicts the judgments is
  mostly *where and how large the figure's windows are* (a coarse
  kinematic summary), not hierarchical composition as such.
* **Why the structure is weakly identified:** under its current generative
  read-out the LoT can draw static band layouts and constant-velocity
  windows that clamp at the edge. It has **no oscillation or periodicity
  primitive**, and a SerialNode is inert without a field, so a body inside
  a Serial never moves. Every gait is a cyclic limb movement, so the best
  programs fit the time-averaged silhouette (see `analysis/fits/*.png`).
  Many structurally different programs render the same silhouette, so the
  data cannot choose between them.
* **Mixing:** R-hat of the log-posterior is below 1.5 for 14/26 manners.
  It is much worse for dragging (15), groining (10), jumping (4.7),
  hopscotching (4.3), scurrying (3.9) and mermaiding (3.7), where chains
  settle in different modes. Results for those manners rest on unconverged
  posteriors.
* **Bootstrap CIs** resample participants, which adds noise to the human
  matrix and pulls rho down. That is why the intervals sit below the
  point estimates. Read them as conservative.
* **Scurrying** uses `11F_scurrying.mp4` in place of the unavailable
  `10F_scurrying.mp4` (for the kinematic baselines too).

## What would most sharpen the test (not done; each changes or extends the model)

1. Give the generative fragment a way to express rhythm. Options: an
   oscillating body (velocity that reverses at the edge or on a period), or
   time-varying gates via dashed edges between imagined nodes. This is a
   change to LoT semantics and needs your decision.
2. Fix the identifiability problem before interpreting structure:
   * canonicalise render-equivalent programs, or
   * compare posteriors by *rendered behaviour* as well as by tree.
3. Better mixing: parallel tempering, or more chains with random restarts
   (cheap here: rollouts take ~1 ms).
4. Manner-level inference over all 25 actors
   (`preprocess --all` + `inference.run --manner-level`). This is already
   supported. It turns "one video per manner" into the intended
   `p(r_m | v_m1..v_mN)`.
5. Grammar ablations: rerun inference with a modified `[lot]` section
   (e.g. `max_depth = 1` for no hierarchy, or a reduced `primitives` list)
   into a separate `--out` directory, then run `analysis.run --results` on
   it.
