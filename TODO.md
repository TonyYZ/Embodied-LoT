# To do

Open work on the motion scoring (`inference.likelihood = "descriptors"`), from the analysis of the 5-manner fit
`results/grace_v7_parts/desc_fix_5` (30 Oct 2026). Rules for every item: training actors only for anything learned,
one rule stated before looking at results, all variants reported, judged on held-out (validation) actors, the test
actors sealed, nothing tuned on human data.

- [~] **Rhythm blind spot** (pilot done: `inference.descriptors = "full"`, job 90600 `full_5`, 6000 iterations; needs a full-length fit). Only once- and twice-per-cycle amplitudes are compared, so a single step across the clip
  matches the numbers without following the rhythm (bowing: one bow per clip against the actors' two; jumping: a
  sawtooth; see `grace_lot/analysis/motion_view.py`). Compare more rhythms (including once per clip), or how the
  strength is spread across rhythms.
- [~] **Posture terms** (in the same `full` option and pilot). Each part's average position relative to the torso, its size and its tilt (timing-free), so
  programs look like bodies again; positions are not scored at all now.
- [~] **Library inside the search** (built 30 Oct: `grace_lot/inference/library.py`, `inference.library`;
  seed library `configs/library_seed.json` = 7 learned pieces + hand-built oscillator, diagonal shuttle, vertical
  pair; prior = shortest description, limits on the description; paired moves `lib_insert`/`lib_remove`, checked
  exact on 400 cases). With `descriptors="full"` and `warmup=16` oscillators beat still and small units on
  17/30 manner x part pairs after cost (motion-only scoring favours one-tick start-up blips instead). The seed
  library was compressed from a hand-written candidate list under the old scoring, so it is not used: the loop
  below starts from the search's own programs. Round 0 (no library, new scoring) finished 31 Oct: `round0_5`, 28-45% of the way to the ideal, recognition 0.92, no program repeats at the actors' rhythm; on the page. Next: compress its programs (not written yet). The learned pieces (the three-strip shuttle skeleton with its dashed edges, the
  self-gating strip `(parallel ? ? (body ?)) | .1>.0`) in the prior and as one-step proposals; refit the 5 manners and
  compare with `desc_fix_5` on held-out actors. A library learned from four manners shortens the fifth's limb units by
  22-48%; assembled six-part programs still fit worse than the search's best (why is not known yet: the limb pairs
  add little, so it is not only that limbs chosen one at a time miss them).
- [~] **Library learning loop (DreamCoder-like).** Fit -> compress the fits' programs -> refit, a few rounds,
  learned pieces only; hand-built pieces as a separate condition. Pieces with subtree holes; learned piece weights.
- [ ] **Search speed (31 Oct).** Two exact passes are in (`tools/bench_search.py`, `tools/tail_stats.py`, `tools/check_rollouts.py`). The unit
  pile-up is fixed in jumping (worst rollout 40 s -> 4.6 s); ordinary programs are flat Python now, about 2-6 min per 1000
  iterations per ladder, so under 1 hour needs a change to the search. Options, none applied: a budget on units, fewer
  replicas, a shorter warm-up. Distance-based pruning of off-board bodies cannot fire (see the 31 Oct report).
- [ ] **Unknown and motion size (31 Oct, nothing refit).** The average-body fill gives unclaimed cells free body knowledge. A generic
  blob baseline (unclaimed scores 0) rewards confident still shapes; every oscillator and moving window loses, because actors
  swing 0.03-0.97 cells and the smallest move is 1 cell. Oracle rectangles swung by sub-cell amounts beat still ones on 30/30
  pairs (-193 -> +2672). So: sub-cell motion, `lot.window_crop`, and the generic baseline need testing together.
- [ ] **Rhythmic tilt.** Actors' torso and legs tilt rhythmically (0.07-0.19); shuttles only translate (~0.01).
- [ ] **26-manner comparison against templates.** On 5 manners recognition is at ceiling: programs and program-free
  baselines all get 0.96.
- [~] **Gestures with limb-pair timing** (done on 5 manners: no match; schema-level test on 26, `analysis/gesture_schema.py`: at chance, but the own-vs-foil margin follows Norming Study 1, rho 0.47, p 0.015 - a lead). Under motion scoring, gestures do not match their manners (own manner's mean
  rank 4.0 of 5, chance 3.0; amplitudes only). Try matching limb pairs to hand-channel pairs (couplings).
- [x] **Faster measuring on the cluster** (copied 30 Oct). `grace/descriptors.board_signals` as one matrix product (identical scores,
  1.7x faster scoring) is local only; copy it before the next fit.
- [~] **Open cells carry no motion** (`inference.open_cells = "expected"`, decided 30 Oct before scoring). Under the
  fills, outputs claiming 1-2% of the board borrowed real footage's motion and tuned it through the cell-by-cell mix.
  Now an output is read as its expected board (open cells: a real part's average presence, no motion). Checks: default
  scores unchanged, ceiling unchanged, scoring 5-20x faster; the motion fit's programs drop (most in swinging and
  jumping); whole-cell shuttle limbs overshoot the actors' sub-cell swings, so small swings must come from graded
  claims. Pilot fit `expected_5` (job 90601) cancelled for the sub-cell one below; run it later to separate the two changes.
- [~] **Sub-cell drawing** (`lot.subcell = s`, engine switch `SUBCELL`, default 1 = whole cells, decided 30 Oct).
  A tree is drawn s times finer and each claim pooled back, a cell claimed as strongly as the claim covers it, so
  content smaller than a cell neither vanishes nor speaks as loud as a whole cell; lines stay one board cell thick.
  Default unchanged (488 fingerprints identical); s = 3 runs all test programs, ~17% slower. With expected-board
  scoring, the best small leg units improve a lot (swinging -60 to -28, shuffling +9 to +34) and are no longer
  shuttles; swinging and jumping legs still need larger programs. Pilot fit with `subcell = 3` and
  `open_cells = "expected"`: job 90606 `subcell_5` (24000 iterations), started 30 Oct with adjacent figure bands
  as one template slot (90605, with the old slots, was cancelled; its partial output is `subcell_5_oldslots`).
- [~] **Missions cross the parent region** (`lot.mission_parent`, engine switch `MISSION_PARENT`, default off, 30 Oct).
  A moving leaf's mission runs to the far edge of the nearest region enclosing the leaf's own (stopping inside it), and
  content on a mission from one region takes no new one from another until it arrives (same-region leaves still
  combine). Off: a mission ends once its content leaves the issuing leaf's region, so neighbouring strips that push
  opposite ways just bounce content at their border. On, the two-strip program
  `(template upper/V upper/H (parallel (serial_h (parallel (template center/H center/V) bottom/H* top/V*) (parallel top/H* bottom/V*)) out))`
  shuttles corner to corner of the top-right region (period 14) without observers or dashed edges. Default
  unchanged (488 fingerprints); on: 0 errors, 143 of 494 test programs and 4 of 41 examples move differently.
  Pilot fits (expected boards reading claim strength, whole cells): job 90953 `claims_5` (missions as before)
  and job 90954 `claims_mission_5` (missions cross the parent region), queued after it. 90606/90607 cancelled.
- [ ] **Sub-cell figures cannot move.** Motion only carries substance that is more likely figure than not (v > 0.5),
  so a sub-cell dot (e.g. a third of a cell over `void`, cell value ~0.1) is drawn but never moved. Letting motion
  carry fractions of a cell would make `subcell > 1` usable for motion; until then pilots use `subcell = 1`.
- [x] **Expected boards read claim strength** (30 Oct): windows also output how strongly each cell is claimed
  (`rollout(..., claims=True)`); expected presence = strength x claims' share of figure + (1 - strength) x average
  presence. Rollouts unchanged (488 fingerprints); whole-program and component strengths agree.
- [~] **Half leaves** `near` (1 0: bottom/left half) and `far` (0 1: top/right half), 30 Oct. Static only (no moving
  version; `near*`/`far*` have prior -inf), drawn as filled halves (no lone-band line), one template slot each, so
  `(template far/H far/V ...)` is exactly the top-right quadrant. Added to `lot.primitives` in
  `configs/grace_v7_parts.toml` (11 leaves); earlier runs keep their saved primitive lists. Existing programs draw
  as before (488 fingerprints); the grammar's sampler and log-probabilities agree. Not yet used in a fit.

