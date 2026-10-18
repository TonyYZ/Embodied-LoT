# Dynamic-reachability audit, v6 (2026-10-17)

This is a diagnostic only. Nothing in the language, observers, likelihood, prior or sampler was
changed. It uses the `configs/grace_v6_parts.toml` semantics and the `results/grace_v6_parts/diag4`
posteriors (4 manners, 10k iterations, training actors only).

**Question:** can v6 produce time-varying output detector activations, and where do dynamics disappear?

The causal chain checked, per program:
1. **dyn:** the program has a dynamic leaf.
2. **active:** an imagined, non-template motion leaf has gate x group gates > 0.5 at some tick.
3. **issued:** that leaf creates a body or mass order, or a random push.
4. **displaced:** a body moves, or a mass cell is displaced.
5. **out change:** an `out` field differs between ticks.
6. **det change:** the scanner's detector activations differ between ticks.

## 1. Minimal programs (16 ticks, 12x12)
In this notation, `center/H` is a central *column* band and `center/V` a central *row* band.

| Program | Stops at | What happens |
| --- | --- | --- |
| `(parallel center/H bottom/V* out)` | full chain | Bulk motion moves the band up one row per tick. The salient-row centroid goes 5.5 → 2.0 by tick 7. The detectors change from tick 1 to tick 15 (lower 1 → 0, v-centre 0.91 → 0). |
| `(parallel bottom/V* out)` | issued | The leaf is on, but its region holds nothing (a motion leaf draws nothing itself). |
| `(parallel (body center/H) bottom/V* out)` | displaced | An order is issued, but the body fills its whole region (the board) and has no room to move. |
| `(parallel (serial_v (body center/H) void/V) bottom/V* out)` | full chain | The body moves up one cell per tick; detectors change. With `top/V*` the body already sits at the edge it is pushed towards, so it is clamped. |
| `(serial_v (parallel center/V bottom/H*) out)` | out change | Mass moves in the top strip. The `out` window covers only the bottom strip, so it never sees the motion. |
| `(parallel void/H bottom/V* out)` | det change | Only yin and unknown cells move. The field changes, but no cell reaches salience (0.75). |

### Dashed gating

| Program | Result |
| --- | --- |
| `(parallel center/H top/V* out center/H) \| .3>.1` | **On.** The observer sees the band and the gate goes 0.5, 0.5, 1.0 (the dashed delay). Motion starts at tick 2, and the detectors change until tick 15. |
| `(parallel center/H universal/H* out periphery/H) \| .3>.1` | **Off.** The observer doesn't match; the gate goes 0.5 → 0.0, and nothing moves. |
| `(parallel center/H bottom/H* out center/H) \| .3>.1` | **On, then off.** The gate goes 1.0 at tick 2 and drops to 0.0 by tick 7 as the band leaves the centre. The bulk order issued at onset still runs to completion. |
| `(parallel center/H universal/H* out center/H) \| .3>.1` | **Sustained.** Random pushes come every tick while on; the gate decays 1.0 → 0.63, and the detectors keep changing. |
| `periphery/H (parallel center/H universal/H* out) \| .0>.1.1` | **Never on.** A lone observer tree sees only unknown, so the gate is stuck at 0.5. |

**Every stage works in isolation, and so does dashed on/off gating.** I found no implementation bug.

Two edge effects come from the rules, not from bugs:
- Bulk motion moves the whole region, yin ground included, and piles it at the board edge. Where
  moving yang lands on yin, the cell reads 0.5, so a slid band can fade out at the edge.
- A body that fills its region, or that is pushed towards the edge it already touches, cannot move.

## 2. Posterior fractions (diag4)
Posterior mass, with the unique-program fraction in brackets:

| Manner | dyn | active | issued | displaced | out change | det change |
| --- | --- | --- | --- | --- | --- | --- |
| bowing | 0.98 (0.98) | 0.42 (0.43) | 0.18 (0.18) | 0.14 (0.14) | 0.46 (0.47) | 0.09 (0.08) |
| crisscrossing | 0.96 (0.96) | 0.30 (0.30) | 0.03 (0.03) | 0.00 (0.00) | 0.58 (0.57) | 0.02 (0.02) |
| jumping | 0.99 (0.99) | 0.45 (0.45) | 0.15 (0.14) | 0.15 (0.13) | 0.60 (0.60) | 0.14 (0.12) |
| swinging | 1.00 (1.00) | 0.25 (0.25) | 0.04 (0.04) | 0.03 (0.03) | 0.59 (0.58) | 0.02 (0.02) |

### Where the chain first breaks (posterior mass)
- **At "active":** 0.54 to 0.75.
- **At "issued":** 0.21 to 0.30.
- **Complete:** 0.00 to 0.12.

### Fate of dynamic leaves (posterior-weighted share)
- **In a template:** 45 to 53%. These orient their content and never issue motion, by design.
- **Observers:** 14 to 18%.
- **Gated off (≤ 0.5):** 8 to 12%.
- **Gate stuck at exactly 0.5:** 4 to 8%.
- **`void`, which has no motion kind:** 4 to 8%.
- **On at some tick:** only 9 to 19%.

### Timing is the decisive finding
**All detector change in all four posteriors happens at ticks 1 and 2 only.** No sampled program
changes its detectors after tick 2.

"Out change" is common (0.46 to 0.60), but mostly without any displacement. It is gate start-up
(dashed targets begin at 0.5) and gate drift changing drawn intensities: small changes (median mean
|Δy| 0.01 to 0.04) that rarely cross salience.

## 3. Top programs traced
- **The four MAP programs** (traced earlier) have no displacement:
  - bowing: all 4 dynamic leaves are templates;
  - jumping: the only imagined motion leaf (`bottom/H*`) sits under a plan gated by a lone observer,
    stuck at 0.5;
  - swinging: the motion leaves are templates, an observer, or gated by lone observers;
  - crisscrossing: `periphery/V*` is on for all 16 ticks, but its strip holds nothing to move.
- **The highest-ranked programs with displacement:**
  - bowing rank 1;
  - jumping rank 2, with the content `periphery/V*` in a plan whose template is a 3-strip serial;
  - swinging rank 232.
  
  All issue orders once (ticks 0 to 2), then never again. In jumping rank 2, the leaf has 9 small
  template slots. Centripetal motion moves each half by span/4 = 1 cell: 18 one-cell orders, all done
  in one tick. Mass that stays inside a lit leaf's region gets no further orders (rule B), so the
  output is static from tick 2 on.

## 4. Hand-built oscillation (can v6 say "up, down, up"?)
Four quick attempts at observer-gated reversal failed.
- **Paired observers that don't match** keep the gates off.
- **One observer gating `top/V*` and `bottom/V*`** makes the two motions cancel.
- **`center/V` gating the downward slide and `lower/V` gating the return** gets the slide going
  (centroid 5.5 → 10 by tick 8), but the band piles and fades at the edge, so the return never comes.
- **Sustained change does occur** with `universal` jitter (a wandering centroid), and as a one-way
  slide lasting up to about 12 ticks.

## 5. Verdict
- **Implementation bug: no evidence.** Every link of the chain works in minimal programs.
- **Semantics: the main cause.**
  - About 80% of dynamic-leaf placements are inert by rule: template, observer, `void`, or gated at
    ≤ 0.5 (including lone-observer gates stuck at 0.5).
  - Of the leaves that are on, most have nothing in their region, since a motion leaf draws nothing.
  - When motion does fire, it is one-shot: one order per entry, sized by the leaf's own region.
    Nested slots give 1-cell moves finished in a tick.
  - Repeated motion needs universal jitter or gate toggling, and edge piling erases what would come
    back.
- **Search or likelihood: possible but untested.** The sampler does reach displacing programs (up to
  15% of mass), so dynamics are reachable. Whether a sustained moving program would score better than
  the static ones has not been measured. A next diagnostic, still with no changes, would score
  hand-built moving programs against their static twins on training videos.

## 6. Does the likelihood reward the motion v6 can make? (training videos only)
Setup: v6 likelihood (β 250, w 1), the MAP of each diag4 manner, and a library of 360 single-tree
clips.
- The clips are `(parallel A M out)` and `(parallel A M out A) | .3>.1` for every static A and every
  dynamic M, kept if their detectors change; 358 of them still change after tick 3.
- Each clip's **static twin** is its frame 0 held for all 16 frames.
- Scores are computed on rendered output stacks: the MAP's outputs with one channel swapped for a clip.

| | bowing | crisscrossing | jumping | swinging |
| --- | --- | --- | --- | --- |
| MAP log-lik | -200.2 | -217.7 | -290.5 | -150.0 |
| moving clip beats its static twin (share of 2160 swaps) | 0.54 | 0.51 | 0.53 | 0.52 |
| median (moving - static) | +0.8 | +0.5 | +0.9 | +0.6 |
| best single moving swap vs MAP | -25.0 | -29.7 | -13.0 | -27.7 |
| best single static swap vs MAP | -67.0 | -78.5 | -84.0 | -71.6 |
| greedy assembly from the MAP, moving or static clips | no improving swap | none | none | none |

**Change term and the ceiling on motion.** The MAP's static outputs pay the full frame-to-frame change
term: 71.6, 106.4, 159.6 and 52.0 nats, about a third to half of the log-likelihood. (This equals the
MAP's log-likelihood at w = 1 minus its log-likelihood at w = 0.)

The best a deterministic program could do is to give each part the mean change sequence across the 15
actors. Even that ignores the cost of matching frames, and it would recover only:
- bowing: 18.9 nats (26%);
- crisscrossing: 8.9 nats (8%);
- jumping: 28.5 nats (18%);
- swinging: 3.3 nats (6%).

The ceiling is low because actors are not aligned in time: one fixed 16-frame program can't be in
phase with all 15 videos. For swinging, the per-actor swings largely cancel in the mean.

### Reading
Within the library, moving and static are a coin flip. Motion helps when its timing happens to fit and
hurts when it doesn't, and no crude motion improves on the MAP's static outputs.

Even a perfect deterministic motion is worth only 3 to 29 nats on the likelihood. A few extra nodes in
the prior cost about that much at λ = 0.5. So beyond the semantic bottlenecks of section 5, **the
likelihood itself gives little reason to move**: the change term mostly punishes every program equally,
because unaligned actor timing leaves almost nothing shared to fit. This is a property of the per-video,
fixed-time comparison, not a bug.

Scripts (diagnostic, scratchpad): `audit.py`, `posterior_audit.py`, `change_audit.py`,
`motion_value.py`.

## 7. Temporal-alignment test (diagnostic only; training videos; likelihood unchanged)
Hypothesis: actor phase misalignment suppresses the reward for motion.

Setup:
- the same 360 moving clips and static twins as in section 6, each swapped into one diag4 MAP output;
- each video may shift the whole program output (all channels) by one τ, with the best τ chosen per
  video;
- two kinds of shift:
  - **circular:** τ ∈ 0..15, frames rolled (the end wraps onto the start);
  - **local:** τ ∈ [-2, 2], with the edge frame held.

Scripts: `align_test.py`, `align_bound.py`, `align_null.py` (scratchpad).

### 7.1 Moving vs static twin, before and after alignment
Moving − static log-lik over 2160 swaps:

| Manner | Moving wins: fixed / local / circular | Median: fixed / local / circular | Best moving swap vs MAP: fixed / local / circular |
| --- | --- | --- | --- |
| bowing | 0.54 / 0.59 / 0.58 | +0.8 / +2.8 / +1.2 | -25.0 / -18.8 / -22.4 |
| crisscrossing | 0.51 / 0.53 / 0.52 | +0.5 / +2.2 / +1.0 | -29.7 / -23.0 / -29.7 |
| jumping | 0.53 / 0.56 / 0.54 | +0.9 / +2.7 / +1.1 | -13.0 / +3.1 / -9.6 |
| swinging | 0.52 / 0.54 / 0.53 | +0.6 / +1.6 / +0.9 | -27.7 / -21.1 / -27.3 |

**Circular alignment barely changes anything:** the median moves by under 0.4 nats.

**The local-shift gain is mostly an artifact.**
- Almost every video, for moving and static stacks alike, picks τ = -2: 98-100% of videos for static
  twins.
- τ = -2 drops the first two frames, where programs have their gate start-up transient (dashed targets
  begin at 0.5).
- The swap that gains most in every manner is `(parallel periphery/H periphery/H* out)`, with gains of
  +160 to +200. Its gain is already there at fixed time, and its best circular shift is 0 in almost
  every video. What helps it is its later state fitting better than its first frame, not timing.

### 7.2 Best shift per video
Per-video shifts from the observation-only alignment, with an alternating mean template. The 15
training actors are listed in split order.

| Manner | Circular τ | Local τ |
| --- | --- | --- |
| bowing | 13 15 15 15 1 0 0 0 15 15 15 13 0 0 0 | -1 2 1 2 2 2 2 2 2 2 2 -1 2 2 2 |
| crisscrossing | 1 2 9 2 0 6 5 12 12 2 8 12 12 3 9 | -2 -2 -1 -1 2 1 2 2 -2 1 -2 -2 2 -2 -1 |
| jumping | 12 2 7 10 5 5 13 8 0 10 5 10 5 5 15 | -2 -2 1 -1 1 1 -2 2 -2 -1 1 -1 1 -1 2 |
| swinging | 1 4 9 0 2 0 1 1 1 11 7 12 4 0 0 | 1 -2 -1 0 2 0 1 1 1 -2 2 2 -2 0 0 |

Bowing's shifts are near zero (within ±3 frames, circularly). The periodic manners scatter over the
whole cycle.

### 7.3 Maximum recoverable change-term gain, and a null control
Share of the static change cost (bowing 71.6, crisscrossing 106.4, jumping 159.6, swinging 52.0 nats)
that one shared deterministic change sequence recovers in-sample. The **null** runs the same search on
videos whose frames are shuffled, which destroys all timing; its gain is what the extra freedom buys by
itself.

| Manner | Fixed (null) | Local ±2 (null) | Circular (null) | Shift ±4 (null) | Shift ±4 + speed ×0.67-1.5 (null) |
| --- | --- | --- | --- | --- | --- |
| bowing | 26% (6%) | 56% (40%) | 48% (46%) | 64% (53%) | 90% (84%) |
| crisscrossing | 8% (5%) | 48% (36%) | 54% (38%) | 57% (50%) | 87% (84%) |
| jumping | 18% (5%) | 40% (33%) | 51% (39%) | 54% (46%) | 88% (82%) |
| swinging | 6% (6%) | 32% (26%) | 35% (26%) | 45% (39%) | 84% (79%) |

**Most of the alignment gain is search freedom, not shared timing.**
- Above the null, alignment adds only about 2-16 percentage points.
- Swinging and crisscrossing do not stop cancelling: aligned swinging beats the null by +9 points at
  most. Crisscrossing (+16 points, circular) shows the most shared structure.
- The speed option's large in-sample gains are matched by the null too. Slower replay interpolates
  between frames, which shrinks every change, so it can't show real speed differences.

### 7.4 Phase vs speed/period
Shifting is **not sufficient**. Actors differ in tempo and in how much of the movement each 16-frame
clip covers, as well as in phase.
- **Sampling:** each clip is an actor's whole valid segment resampled to 16 frames. Segment lengths vary
  2-3x within a manner (source frames):
  - bowing: 111-265;
  - crisscrossing: 95-270;
  - jumping: 86-150;
  - swinging: 206-450.
  
  So the 16 frames hold a different number of cycles for each actor.
- **Periods vary too.** Autocorrelation periods (frames out of 16, per actor):
  - bowing: 5-8;
  - swinging: 4-9;
  - jumping: 2-5;
  - crisscrossing: 2 for most actors.
- **The fast manners may be aliased.** Strides of about 7 source frames give a period of 2 frames for
  crisscrossing and many jumping actors. That is the shortest period the sampling can represent, so
  their motion may be undersampled.

### Reading
The hypothesis is only partly supported. Phase misalignment is real: the best shifts scatter over the
cycle for the periodic manners. But a single shift per video recovers little shared change beyond what
pure search freedom gives. That is because actors also differ in tempo and in cycles per clip, and the
fast manners are near the 16-frame sampling limit.

Before any likelihood change, worth checking:
1. **Sampling:** a fixed temporal stride, or a fixed number of cycles per clip, instead of resampling
   each whole segment to 16 frames.
2. **Honest gains:** leave-one-actor-out estimates, since the in-sample bounds above are optimistic.

## 8. Temporal-preprocessing diagnostic (diagnostic only; training actors; nothing changed)
Question: is the whole-segment → 16-frame resampling destroying shared motion structure?

All samplers give 16 frames of the same full-rate canonical joints, then the unchanged scanner.

| Sampler | What it samples |
| --- | --- |
| **current** | the whole valid segment, linearly resampled to 16 frames. It reproduces the stored observations exactly (checked). |
| **stride_max** | fixed stride per manner: the largest integer stride whose 16 frames fit every actor's segment, window centred. |
| **stride_4** | fixed stride 4: dense, an aliasing control. |
| **cycle1 / cycle2** | 1 or 2 full-rate cycles, centred, resampled to 16 frames. |

**Cycle detection** is the same generic procedure for every manner:
- the autocorrelation of the detrended joint signal;
- the period is the first peak reaching 80% of the highest peak, with r ≥ 0.3 and a lag of at least 6;
- the result is flagged ambiguous when the highest peak isn't a multiple of the first;
- when it fails, the actor falls back to `current`.

Script: `temporal_pre.py`.

### 8.1 Coverage, period, aliasing, consistency
Coverage is in source frames, min-max (median). The 16-frame period is a median, with its range. Alias is
the median share of full-rate signal power above the sampling Nyquist. Pairwise correlation is between
actors' detector-change sequences; the second figure takes the best circular shift, with the
frame-shuffled null in brackets.

| Manner | Sampler | Coverage | Stride | Period in 16 frames | Alias | Pair corr: fixed / best shift (null) |
| --- | --- | --- | --- | --- | --- | --- |
| bowing | current | 111-265 (173) | 7.3-17.6 | 7 (5-8) | 0.07 | 0.24 / 0.50 (0.45) |
| | stride_max | 106 | 7 | 10 (8-11) | 0.05 | 0.23 / 0.47 (0.53) |
| | stride_4 | 61 | 4 | none found | 0.06 | 0.20 / 0.53 (0.47) |
| | cycle1 | 53-122 (80) | 3.5-8.1 | 8, but found in 2 of 15 actors only | 0.04 | 0.21 / 0.50 (0.47) |
| | cycle2 | 105-199 (149) | 6.9-13.2 | 8 (7-8) | 0.04 | 0.26 / 0.66 (0.49) |
| crisscrossing | current | 95-270 (111) | 6.3-17.9 | **2** (2-5) | **0.59** | 0.02 / 0.53 (0.44) |
| | stride_max | 91 | 6 | 2 (2-5) | 0.06 | 0.05 / 0.61 (0.43) |
| | stride_4 | 61 | 4 | 4 (3-4) | 0.03 | 0.04 / 0.63 (0.46) |
| | cycle1 | 24-33 (30) | 1.5-2.1 | 8 (7-8) | 0.02 | 0.01 / **0.81** (0.47) |
| | cycle2 | 47-65 (59) | 3.1-4.3 | 4 | 0.03 | 0.01 / 0.78 (0.47) |
| jumping | current | 86-150 (107) | 5.7-9.9 | **2** (2-5) | 0.26 | 0.13 / 0.49 (0.38) |
| | stride_max | 76 | 5 | 3 | 0.13 | 0.01 / 0.62 (0.42) |
| | stride_4 | 61 | 4 | 4 (3-4) | 0.14 | 0.01 / 0.63 (0.42) |
| | cycle1 | 15-33 (30) | 0.9-2.1 | 7 (7-8) | 0.06 | 0.04 / 0.60 (0.41) |
| | cycle2 | 29-65 (59) | 1.9-4.3 | 4 (4-7) | 0.09 | 0.06 / 0.67 (0.39) |
| swinging | current | 206-450 (263) | 13.7-29.9 | 6 (4-9) | **0.40** | 0.00 / 0.37 (0.31) |
| | stride_max | 196 | 13 | 7 (3-11) | 0.19 | 0.00 / 0.33 (0.31) |
| | stride_4 | 61 | 4 | 9 (8-11), found in 3 actors | 0.06 | -0.01 / 0.35 (0.29) |
| | cycle1 | 78-228 (103) | 5.1-15.1 | 11 (5-11) | 0.08 | 0.00 / 0.50 (0.33) |
| | cycle2 | 155-319 (205) | 10.3-21.2 | 7 (7-9) | 0.19 | 0.00 / 0.52 (0.31) |

### Full-rate period detection

| Manner | Detected | Periods (source frames) | Notes |
| --- | --- | --- | --- |
| crisscrossing | 15 of 15 | 23-32 | |
| jumping | 15 of 15 | 27-32 | one actor at 14, probably a half-period |
| bowing | 13 of 15 | 52-99 | 2 failures: no peak of 0.3 or more, likely bows that don't repeat in the segment |
| swinging | 14 of 15 | 77-128 | 1 failure; one actor at 159, possibly a double period |

No detection was flagged ambiguous.

### Aliasing evidence (current sampler)
- **crisscrossing:** 59% of the full-rate power lies above Nyquist.
- **swinging:** 40%.
- **jumping:** 26%.

Crisscrossing and jumping appear with a 2-frame period, the alternation limit of the sampling, although
their true periods (about 23-32 source frames at strides of 6-18) would give 1.5 to 5 frames per cycle.
Fixed strides and cycle sampling bring the alias share down to 2-19%.

### 8.2 Leave-one-actor-out ceiling on the change term
Each held-out actor is scored against the other 14 actors' template, as a percentage of that sampler's
own static change cost. "Aligned" builds the template with circular-shift alignment of the 14 and lets the
held-out actor pick its best shift. The null is frames shuffled, scored the same way as a share of its own
static cost.

| Manner | current | stride_max | stride_4 | cycle1 | cycle2 |
| --- | --- | --- | --- | --- | --- |
| bowing: fixed / aligned (null) | 16 / 34 (25) | 17 / 35 (39) | 18 / 45 (23) | 12 / 36 (20) | 18 / 49 (33) |
| crisscrossing | -5 / 42 (21) | -3 / 50 (17) | -4 / 49 (27) | -8 / **67** (33) | -6 / **67** (36) |
| jumping | 6 / 34 (13) | -4 / 51 (21) | -6 / 49 (21) | -2 / 46 (21) | -1 / **55** (16) |
| swinging | -7 / 16 (9) | -7 / 8 (9) | -9 / 15 (4) | -7 / **36** (13) | -7 / **35** (10) |

Aligned gain over the null, in percentage points:

| Manner | current | stride_max | stride_4 | cycle1 | cycle2 |
| --- | --- | --- | --- | --- | --- |
| bowing | +9 | -4 | +22 | +16 | +16 |
| crisscrossing | +21 | +33 | +22 | +34 | +31 |
| jumping | +21 | +30 | +28 | +25 | +39 |
| swinging | +7 | -1 | +11 | +23 | +25 |

### 8.3 Moving clips vs static twins
These are the 360 clips of section 6, at fixed time, against the diag4 MAP.

There is **no change under any sampler**. Moving beats its twin in 51-55% of swaps, with a median of
+0.5 to +0.9 nats, and the best moving swap is still 11-31 nats below the MAP. The crude library, and a
MAP fitted under `current`, can't use structure that only appears when phase is aligned.

### Reading
1. **The current resampling does damage shared structure for the periodic manners.**
   - It aliases crisscrossing (59% of power above Nyquist) and jumping, and undersamples swinging.
   - It fits a different number of cycles into each actor's 16 frames.
   - Sampling a fixed number of detected cycles raises the genuine, held-out shared change structure (above
     the null) from +7 to +23-25 points for swinging, +21 to +31-34 for crisscrossing, and +21 to +39 for
     jumping. Pairwise aligned correlation rises to 0.78-0.81 for crisscrossing.
2. **Phase still has to be aligned.** At fixed time, the held-out ceiling stays at or below 0 for every
   periodic manner under every sampler. Only bowing, the one non-periodic manner, shares structure at
   fixed time (12-18%). Better sampling alone doesn't give a fixed-time program anything to fit. It needs
   phase alignment as well, or a phase-free comparison.
3. **Cycle detection is generic and mostly works.** It succeeds for every crisscrossing and jumping actor,
   with one likely half-period. It fails for 2 bowing actors and 1 swinging actor, and gives one likely
   double period for swinging. Those fall back to `current`, or would need a stated rule.
4. **Hand-built moving programs don't gain more over their static twins under any sampler.**

## 9. Cycle-normalised observations + latent phase (diagnostic only; likelihood-only, no MCMC)
Score: `score(v, r) = max_τ log p(v | roll(R(r), τ))`.
- τ is a circular shift over 16 frames, one per video for the whole output stack.
- Observations are the cycle1 and cycle2 sets of section 8. Fallback actors: 2 bowing, 1 swinging.
- The clips, static twins and diag4 MAP are as in section 6. The MAP and the static twins are scored
  with the same phase search.
- **Null control:** each video's frames shuffled, then the same phase search.

Script: `phase_score.py`.

### Results
Moving − static log-lik over 2160 swaps.

| Manner / sampler | Moving wins: fixed → phase (null phase) | Median: fixed → phase (null phase) | Phase gain, median: moving / static (null moving) | Best moving vs MAP, phase |
| --- | --- | --- | --- | --- |
| bowing / cycle1 | 0.55 → 0.60 (0.58) | +0.9 → +1.3 (+1.8) | +1.05 / 0.00 (+1.25) | -21.0 |
| bowing / cycle2 | 0.53 → 0.57 (0.58) | +0.6 → +1.3 (+1.9) | +0.73 / 0.00 (+1.45) | -22.1 |
| crisscrossing / cycle1 | 0.51 → 0.52 (0.52) | +0.5 → +1.0 (+1.0) | +0.25 / 0.00 (+0.40) | -30.2 |
| crisscrossing / cycle2 | 0.52 → 0.52 (0.52) | +0.5 → +1.0 (+1.0) | +0.16 / 0.00 (+0.35) | -30.4 |
| jumping / cycle1 | 0.53 → 0.54 (0.55) | +0.8 → +1.0 (+1.4) | +0.58 / 0.00 (+0.89) | -11.1 |
| jumping / cycle2 | 0.54 → 0.54 (0.54) | +0.9 → +1.3 (+1.4) | +0.32 / 0.00 (+0.92) | -11.6 |
| swinging / cycle1 | 0.51 → 0.52 (0.52) | +0.5 → +0.7 (+0.8) | +0.08 / 0.00 (+0.06) | -28.9 |
| swinging / cycle2 | 0.52 → 0.53 (0.53) | +0.5 → +0.7 (+0.8) | +0.05 / 0.00 (+0.11) | -28.2 |

Notes on the table:
- Static twins and the MAP gain exactly 0 from phase: they are constant over time.
- The mean moving-minus-static stays negative, between -2 and -10. Motion that doesn't fit costs more
  than motion that fits gains.

### Answers
- **Is there a consistent motion advantage? No.** It is still about 50/50 (0.52-0.60 of swaps).
- **Is the phase gain real temporal structure? No.** Moving clips gain only 0.05-1.05 nats (median) from
  latent phase, and the same gain or more appears on the shuffled-time null (0.06-1.45). It is search
  freedom, not shared timing.
- **Does any hand-built moving program clearly beat its static explanation? No.**
  - The largest moving-minus-static is still `(parallel periphery/H periphery/H* out)`, at +156 to +176.
    It is the same at fixed time and under phase, so it isn't temporal: its later state fits better than
    its first frame.
  - Swapped in, it stays 104-188 nats below the MAP.
  - The best moving swap of any kind stays 11-30 nats below the MAP, unchanged from fixed time.

### Reading
Cycle-normalised sampling plus phase invariance creates potential reward for shared periodic change.
Section 8's leave-one-actor-out ceilings are 35-67% with alignment. But none of the hand-built,
v6-producible moving programs can collect it.

Those programs move once: one-way slides, a one-off collapse, or random jitter. None is periodic. A
circular shift of a one-way slide only moves its single jump and adds a seam. Section 4 found no
reversing oscillator with the current semantics.

So the bottleneck is expressive: v6 can't produce periodic detector changes. It is not the temporal
preprocessing or the likelihood's fixed time alone, although those also suppress the reward (sections
7-8).

## 10. Periodic-expressivity search over current v6 (diagnostic only; no semantics changed)
Each program is rolled out under the v6 semantics. Its detector output A (T, K, 6) is classified, testing
in this order:

| Class | Rule |
| --- | --- |
| constant | A never changes. |
| periodic | Conservative: some start s and period p ≥ 2 with A[t] = A[t+p] for every t ≥ s, at least two full cycles after s (T - s ≥ 2p), and non-constant from s on. |
| transient | Changes at 2 or fewer ticks, then constant. |
| monotonic | Every detector value is monotone over time. |
| irregular | Anything else. |

The primary horizon is T = 16. Every non-constant program is also run for 48 ticks. **Robust** means
periodic at both 16 and 48 ticks with the same period. "Universal" means the program has a *dynamic*
universal leaf (random pushes).

Scripts: `periodic_search.py`, `periodic_targeted.py`.

### Coverage of the space

| Set | n | constant | transient | monotonic | irregular | periodic@16 | robust |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact prior draws (v6, λ 0.5) | 20,000 | 0.961 | 0.026 | 0.006 | 0.007 | 7 (0.035%) | **2 (0.01%)** |
| grammar draws with 3-5 dashed edges | 20,000 | 0.958 | 0.036 | 0.003 | 0.003 | 3 | 0 |
| diag4 posterior, all distinct | 3,876 | 0.939 | 0.061 | 0 | 0 | 0 | 0 |
| diag4 posterior, still legal | 338 | 1.000 | 0 | 0 | 0 | 0 | 0 |
| targeted: observer-gated opposing motions | 2,996 | 0.544 | 0.026 | 0.303 | 0.124 | 10 | 0 |
| targeted: same, moving a whole-region body | 2,995 | 1.000 | 0 | 0 | 0 | 0 | 0 |
| targeted: two-strip self-observing loop | 2,992 | 0.930 | 0.012 | 0.036 | 0.021 | 1 | 0 |
| targeted: three-strip loop | 3,000 | 0.933 | 0.012 | 0.039 | 0.016 | 0 | 0 |
| targeted: cross-gated tree pairs | 3,000 | 0.514 | 0.015 | 0.347 | 0.123 | 0 | 0 |

Two notes on the periodic counts:
- Most "periodic@16" hits are coincidences: a period of 2-3 that starts at tick 9-12, which the 48-tick
  run then breaks.
- Only the 2 robust programs are periodic in the strict sense within 16 frames.

### Does a genuinely periodic program exist? Yes, two kinds, and neither is a useful rhythm at 16 frames

**1. Period-2 boundary flicker.** These are the only robust periodic programs at 16 frames: 2 of 20,000
prior draws. Neither uses universal motion.

The smallest (14 nodes, period 2 from tick 1):

```
out (plan upper/H (parallel (parallel universal/H lower/V center/H*) out lower/H)) top/H* upper/H center/V | .1.1.0.1>.1.1.2 .4>.3
```

The other (20 nodes, period 2 from tick 1):

```
top/H (plan void/V* (parallel void/V universal/V) upper/V) (plan (serial_h (serial_h out out) (serial_v periphery/V upper/H) universal/V) (parallel center/H* upper/H)) | .0>.1.0
```

*Mechanism.* The gates are constant, so this isn't a gating loop.
- A centrifugal `center/H*` leaf sits under a template with many small slots (9 in the second program).
  It pushes each slot's two halves apart by one cell.
- A boundary cell pushed into the neighbouring slot counts as a fresh entry there (rule B). There it
  gets the opposite order and returns. Back in its own slot it is fresh again, and so on.
- Mass offsets alternate (1, -1) ↔ (0, 0) every tick, and a detector flips with them. In the first
  program the right detector goes 1.0 ↔ 0.0.

This is a bookkeeping side effect of rule B at slot boundaries, not a designed oscillator.

**2. Body shuttles: real oscillation, but slower than 16 frames allow.**
- The engine's own oscillator test, made legal for v6, is robustly periodic at 48 ticks with period 14:
  `(serial_h (parallel bottom/H* center/H out) (parallel top/H* center/H (body center/H))) | .0.1>.0.0 .1.1>.1.0`
- *Mechanism:*
  - each strip has an observer (`center/H`) that watches its own strip and gates that strip's motion leaf
    through a dashed edge;
  - while the body is in the right strip, that observer is on and `top/H*` pushes the body left, one cell
    per tick, across the boundary;
  - in the left strip, the other observer turns on, and `bottom/H*` pushes it back.
- *Periods:*
  - the two-strip loop gives 13-20;
  - narrower strips (the three-strip loop) give 9-12;
  - observer-gated opposing motions in one tree give 9.
- None of these uses universal motion, and the smallest has 10 nodes.
- **Shortest found: period 9.** Two cycles after start-up need about 20 or more frames, so at the
  data's 16 frames these show as "irregular".
- Cross-gated tree pairs reach periods of 3-4, but only after about 38 ticks, and only with universal
  jitter.

### How common under the prior
- Periodic outputs of any kind: 0.035% of prior draws at 16 frames, 0.01% robust. 96% of draws are
  constant.
- Without universal motion: both robust programs, and every body shuttle.

### Rules that block short-period recurrence
1. **Speed and travel.**
   - Motion moves at most one cell per tick, at full gate.
   - Reversal needs the body to cross into a second observer's region, and bulk orders move the full span
     of the leaf's region. So the period is at least about 2 × (span + 1).
   - The depth limit of 5 and at most 3 children cap how narrow strips can get, to about 4 cells, which
     gives a period of about 9 at the very least.
2. **Rule B gives no restoring force inside a region.** A body that stays inside a lit leaf's region gets
   no new orders. Reversal needs two regions, two observers and two dashed edges, a structure of 10 or
   more nodes that the prior seldom draws.
3. **Orders run to completion after the gate turns off,** so half-cycles can't be cut short.
4. **Lone observers are stuck at 0.5,** so cross-tree gating needs observers in the same tree as drawn
   content.
5. **Mass-based oscillators erase themselves,** by piling at edges (yin + yang reads unknown). Moving
   content must be a body, and a body that fills its region can't move: opposing motions on a
   whole-region body were 100% constant.
6. **Start-up costs 1-2 ticks:** dashed targets start at 0.5, plus the one-tick dashed delay.

### Conclusion
Current v6 *can* express genuine recurrence.
- The intended mechanism, two self-observing strips shuttling a body, works and needs no universal
  motion.
- But its period is at least 9 ticks, so it can't complete two cycles in 16 frames.
- The only short-period recurrences are a period-2 flicker from rule B at template-slot boundaries. That
  is rare (0.01% of the prior) and has nothing to do with body rhythm.

The blockers are speed and travel (rule 1) and the lack of a within-region restoring force (rule 2).
These hold together with the 16-frame horizon and the observed periods of about 4-8 frames per cycle
under cycle sampling.

## 11. Counterfactual oscillator-speed diagnostic (scratch monkeypatches only; nothing committed, no MCMC)
Four relaxations, applied one at a time:

| Condition | What changes |
| --- | --- |
| **cancel** | A motion leaf's pending orders, for bodies and mass, are dropped when its effective gate falls to 0.5 or below. |
| **reissue** | Every tick, a lit motion leaf re-orders everything inside its region, replacing its old orders. Currently only what newly enters is ordered. |
| **fine** | Finer reversal regions: hand-built programs beyond the depth limit of 5 and the 3-children limit. The engine is unchanged. |
| **speed2** | Control: orders move up to 2 cells per tick (rate doubled, per-tick cap 2). |

Programs tested:
- hand-built oscillator families, about 1,500 each:
  - two-strip and three-strip self-observing body shuttles;
  - opposing motions gated by observers in one tree;
  - for **fine**, the shuttle nested in a narrower outer strip, and 4-6-strip loops;
- the canonical shuttle `(serial_h (parallel bottom/H* center/H out) (parallel top/H* center/H (body center/H))) | .0.1>.0.0 .1.1>.1.0`
  and its variants;
- 3,000 exact prior draws per condition (a relaxed grammar for **fine**), to check for accidental
  periodicity.

**Robust** means periodic at both 16 and 48 ticks with the same period. **Shuttle** means a body's
position repeats with that period over at least 2 cells.

Script: `osc_speed.py`.

| Condition | Shortest robust period (16 ticks) | Smallest program | Mechanism | Periods in 4-8? | Canonical shuttle | Accidental periodicity (prior) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | none robust; at 48 ticks, shortest 9 (three-strip) and 10-14 (canonical) | 10 nodes (at 48) | shuttle | no | 14 (two strips), 10 (three) | 0 / 3000 |
| cancel | **7** (2 of 1500 three-strip programs) | 11 | shuttle | yes, rarely | **breaks**: stalls at the boundary (constant or transient) | 0 / 3000 |
| reissue | **6** (2 of 1500 two-strip programs) | 10 | shuttle | yes, rarely | unchanged: 14 and 11 | 0 / 3000 |
| fine | **8** (canonical nested in a half-width strip); 6 (one random nested program) | 12-13 | shuttle at half width; narrower gives a period-4 **mass flicker** (third width), then transient or constant (quarter, sixth) | yes | 14 → **8** at half width | 0 / 3000 |
| speed2 (control) | **6** (canonical + void strip), 8 (canonical) | 10 | shuttle, in every family | yes, consistently | 14 → **8**, 10 → **6** | 0 / 3000 |

### Reading
1. **Speed and travel set the excessive period.**
   - A shuttle's period is about 2 × (cells travelled / cells per tick + 1).
   - Doubling the speed (the control) halves every shuttle's period, canonical included, with no new
     artifacts.
   - Halving the travel instead (**fine**, canonical nested in a half-width strip) gives the same
     period of 8 at unchanged speed.
2. **cancel isn't a clean fix.**
   - It shortens a few programs, to 7 at best.
   - It **breaks** the canonical shuttle: as the body starts to leave its strip, that strip's observer
     drops to 0.5, the push is cancelled mid-boundary, and the body straddles both strips with neither
     observer on.
   - The current shuttle relies on orders running to completion. Under cancel it would need an extra
     hysteresis rule to keep pushing.
3. **reissue isn't a clean fix either.** A few programs reach period 6, but the canonical shuttle and most
   families are unchanged. Re-ordering doesn't shorten the trip.
4. **fine works only while the strip still fits the body's travel.**
   - At half width it gives a real shuttle with period 8.
   - At a third of the width it gives a period-4 mass flicker, an artifact like the one in section 10.
   - At a quarter or a sixth it stops moving.
   - It also needs the depth and child limits raised: about 2 extra levels per halving.
5. **No relaxation creates widespread accidental periodicity:** 0 robust periodic programs in 3,000 prior
   draws for every condition.

**Cleanest single relaxation:** to shorten the period, change the speed and travel (rule 1).
- A speed of 2 cells per tick is the clearest, but it is a blanket change.
- Finer regions match it at half width without touching speed, but only by raising the structural limits,
  and they degenerate into flicker when narrower.
- The order-bookkeeping rules (cancel, reissue) are not the cause, and changing them alone doesn't fix it.

## 12. Latent tempo + latent phase (Option A; likelihood-only diagnostic, nothing changed)
Realization variables, chosen per video for the whole output stack:
- engine tick = r · frame + τ, with r ∈ {1, 2, 3, 4} engine ticks per video frame;
- τ ∈ 0..16r−1 is an offset into a longer run (31r + 1 ticks), so there is no wrap seam;
- 160 (r, τ) options in all.

Scores per video:
- **fixed:** r = 1, τ = 0;
- **max:** the best of the 160 options;
- **marginal:** log of the mean over the 160 options of p(v | r, τ), i.e. a uniform prior over realizations.

Inputs:
- observations: cycle1 / cycle2 (section 8), plus a null with frames shuffled;
- programs: the 360 single-tree clips plus 12 body shuttles that oscillate at 48 ticks (section 11
  baseline), each swapped into each of the 6 diag4 MAP outputs (2232 swaps);
- static twin: frame 0 held. Static twins and the MAP are constant, so all three scores agree for them.

Script: `tempo_score.py`.

Moving − static twin. "Best vs MAP" is the best moving swap under that score.

| Manner (cycle1; cycle2 within ±1) | Fixed: win / median | Max: win / median / best vs MAP | Marginal: win / median / best vs MAP | Null, marginal: win / median / best vs MAP |
| --- | --- | --- | --- | --- |
| bowing | 0.53 / +0.6 | 0.71 / +26.4 / -14.4 | 0.59 / +15.3 / -14.6 | 0.59 / +15.4 / -14.7 |
| crisscrossing | 0.50 / -0.1 | 0.57 / +13.6 / -5.7 | 0.48 / -4.7 / -8.1 | 0.48 / -4.4 / -7.9 |
| jumping | 0.52 / +0.7 | 0.69 / +23.6 / **+28.4** | 0.55 / +4.6 / **+24.4** | 0.55 / +4.5 / **+24.4** |
| swinging | 0.50 / -0.1 | 0.61 / +10.3 / -13.3 | 0.46 / -4.1 / -15.3 | 0.44 / -4.4 / -15.3 |

**Body shuttles, the genuine oscillators.** Under the marginal score they beat their static twins in
0-11% of swaps. The median is -71 to -124 nats, and they sit 160-226 nats below the MAP. The null scores
the same.

**Chosen tempo.** In every top swap, all 15 videos pick r = 1.

### Reading
1. **The real gains equal the null gains** under max and marginal, in every manner. Whatever the
   realization variables buy, it isn't shared temporal structure.
2. **The τ offset adds a new artifact: choosing a state, not a timing.**
   - Because τ can start the window later in the run, a moving program can present its later, settled
     state in place of its first frame.
   - That's why jumping's best moving swap beats the MAP by +24 to +28: it is just as large on shuffled
     time, and every video picks r = 1.
   - Marginalising shrinks this (the bowing median falls from +26 to +15), but doesn't remove it.
   - A cleaner version would restrict τ to a phase within the cycle, or score a fixed number of
     post-transient frames.
3. **Latent tempo alone doesn't rescue the existing oscillators.**
   - The body shuttles are single-channel strip shuttles, and their detector patterns don't resemble any
     body part's.
   - With faster replay they still lose heavily to static explanations, on real data and on the null alike.
4. **What this does and doesn't show.**
   - It does show that the likelihood plus realization variables gives no reward for real dynamics to any
     *hand-built* moving program here.
   - It doesn't show that no v6 program could earn one. That would need a search, such as MCMC, under the
     tempo-augmented score, so that oscillators are shaped and placed to match the parts. That search
     should also restrict τ to avoid the state-selection artifact, and keep the shuffled-time null.

## 13. Shape + periodic dynamics in one program (hand-built probes; diagnostic only)
Question: is there a v6 program whose output has both a part-like spatial signature and genuine periodic
dynamics?

1. **Oblique shapes work.** `(parallel (plan (parallel top/H* bottom/V*) center/H) out)` draws a slanted
   bar.
2. **Its template gate is a two-pose switch.** At gate ≤ 0.5 the bar is upright (72 → 48 salient cells
   across the switch; detectors L/R 0.3); at gate ≥ 0.75 it is oblique (L/R 1.0). At gate 0.5 both mix,
   and few cells stay salient.
3. **A gate loop through an observer doesn't oscillate.** With an observer gating the template
   (`... center/V) | .2>.0.0` or `center/H`), the gate settles at a fixed point (0.46 or 0.66) by tick 3.
   - Dashed edges copy activations linearly, and observers return graded matches, so the feedback is
     damped.
   - There's no threshold or memory on gate-driven appearance; motion onset has a threshold, appearance
     doesn't.
4. **A legal shape + oscillator exists.** A body shuttle in one tree has its left observer also gate the
   limb's template in a second tree:
   `(serial_h (parallel bottom/H* center/H out) (parallel top/H* center/H (body center/H))) (parallel (plan (parallel top/H* bottom/V*) center/H) out) | .0.0.1>.0.0.0 .0.1.1>.0.1.0 .0.0.1>.1.0.0`
   - It is **robustly periodic: period 14** (10 with a third strip) over 48 ticks.
   - The limb channel switches upright → oblique → upright once per cycle.
   - But only for 2 of every 10-14 ticks: the observer is fully on only while the body sits squarely in
     its strip.
   - Within 16 frames that is one pulse, not two cycles.
5. **Carrying the oblique limb on the moving body is illegal:**
   `(serial_h ... (parallel top/H* center/H (body (plan (parallel top/H* bottom/V*) center/H)))) ...`.
   The plan would sit at depth 4, where only leaves are allowed, because a single tree starts at depth 1.
   Even when forced through, it is irregular rather than periodic over 48 ticks.

### The barriers
- **Only motion orders keep memory.** A dynamic needs memory (hysteresis) to sustain an oscillation; a
  graded gate loop just relaxes to a fixed point. In v6 the only such memory is that motion orders run
  to completion. So every genuine oscillator is a body shuttle, and its period is set by speed and travel:
  9 ticks or more (section 11).
- **Shape changes can't oscillate on their own.** They have to be slaved to a shuttle, inheriting its slow
  period, and they show only while an observer is fully on: short pulses.
- **A template gives two poses, not a swing.** Its gate gives upright vs oriented. A left↔right swing needs
  two templates gated in antiphase by two observers: more nodes and edges, the same slow period.
- **The depth and child limits block the direct construction.** Shape inside mover inside reversal
  structure, with an output outside the mover, needs about 6-7 levels against the 5 allowed.

**Answer.** Yes, such a program exists. In v6 its period is ≥ 9-14 ticks and its shape change is a brief
two-pose pulse. The binding constraints are the lack of memory in gate-driven appearance and the speed of
the only memory-carrying mechanism. Oblique-line capacity isn't the problem.

## 14. Memory field + pose readout (clock-driven action field; diagnostic only)
Programs: legal v6, two trees.
- **Memory/clock tree:** a body shuttle (two strips horizontal, two strips vertical, or three strips). Its
  two strip observers are the scanners.
- **Action tree:** `(parallel P1 P2 out)`. Each pose plan `(plan T C)` is gated by one scanner through a
  dashed edge. T and C are drawn from a pool of static and diagonal templates, with pairs chosen so that
  their static detector signatures differ by at least 0.5.
- **Size:** 900 programs (300 pose pairs × 3 clocks).
- **Conditions:** v6 as is, and a scratch counterfactual with all-or-none dashed gates.

Scoring:
- latent tempo r ∈ 1-4 and phase τ restricted to **one detector cycle after start-up**, so there is no
  state selection;
- max and marginal scores;
- the twin is the **best single frame of the program's own cycle**, held;
- cycle1 / cycle2 observations, and the shuffled-time null.

Script: `clock_pose.py`.

### Periodicity and visibility
- **Every program is periodic** over 48 ticks. Under v6, periods are 14 (600 programs), 10 (148) and
  5 (152). Under all-or-none, 14 and 10.
- **The swing between mirrored diagonals is invisible to the scanner.** The 6 detectors max-pool x and y
  separately, so / and \ give identical detector vectors. Only the frames between poses register.
- **How much of each cycle shows a limb:**

| Condition | Visible limb (share of cycle) | Distinct detector states per cycle |
| --- | --- | --- |
| v6 | 0.86 | 4 |
| all-or-none | 0.29 | 3 |

  All-or-none makes this worse, because the scanners exceed 0.5 only briefly and both poses are off for
  most of the cycle.

### Moving vs best static frame of the same program
**0 of 5400 swaps** win, for every manner, sampler, score and condition.

| Condition | Median marginal − twin | Best moving swap − twin | Best moving swap vs MAP |
| --- | --- | --- | --- |
| v6 | -225 to -263 | -99 to -111 | -112 to -139 |
| all-or-none | -189 to -203 | -95 to -105 | -188 to -243 |

- The null scores the same.
- Every video picks r = 1.

### Why: real motion barely moves the detectors
| | Mean \|change\| per frame per detector | Changes > 0.5 |
| --- | --- | --- |
| observed parts, swinging (cycle1) | 0.029 | 0.9% |
| observed parts, crisscrossing (cycle1) | 0.039 | 0.0% |
| clock-pose limb channels | 0.284 | 28.5% |

The scanner's sharp sigmoids, with max-pooling over a part's many skeleton points, mostly saturate:
a part that spans several bins keeps its detectors near 1 while it moves. Real motion therefore shows up
as small, graded detector changes. A discrete pose switch is about 10 times larger than anything observed,
and it is penalised on every frame.

### Reading
- **Your architecture works mechanically, in legal v6:** a memory field, scanners on it, and an action
  field gated by them, attached to `out`. It is genuinely periodic, and phase-coded.
- **Its predictions don't fit the data.** That is not because of tempo, phase or memory. The detector
  interface represents a swinging limb as almost static, and as orientation-blind (/ = \).
- **The binding constraint has moved to the observation interface.** The coarse, saturating 6-detector
  code throws away most of the motion information that the LoT would have to explain. Until that changes,
  the likelihood can reward only very small, graded detector changes, which no discrete pose change
  produces.
- **Options are interface decisions, not tuned on human similarity:**
  - finer or overlapping bins;
  - mean instead of max pooling, or occupancy-weighted responses;
  - orientation-sensitive detectors;
  - comparing at the occupancy level per part.

## 15. Observation-interface diagnostic (training actors only; nothing in the LoT or likelihood changed)
All five interfaces are part-level and anonymous: the same features for every part. They are computed
from each part's skeleton points: joints plus 4 samples per edge, in window coordinates.

| Interface | Features per part | Total (6 parts) | Content |
| --- | --- | --- | --- |
| max6 (current) | 6 | 36 | 3 sigmoid fields per axis, max over points. Reproduces the stored observations exactly. |
| mean6 | 6 | 36 | Same fields, mean over points. |
| fine10 | 10 | 60 | 5 overlapping Gaussian fields per axis, mean over points. |
| orient8 | 8 | 48 | mean6 + elongation-weighted orientation of the part's principal axis (e·cos2θ, e·sin2θ). |
| endpoint4 | 4 | 24 | Proximal and distal endpoint (x, y): the points nearest to and farthest from the torso centre. For head and torso, the principal-axis ends, upper end first. |

Script: `interface_diag.py`, `interface_report.py`.

### Mirrored diagonals (a straight segment / vs \ with the same centre and length)
- **max6, mean6, fine10:** identical (difference 0.000). Any representation built from separate x and y
  profiles is mirror-blind.
- **orient8 and endpoint4:** distinguishable (max difference 2.0 and 0.4).

### Frame-to-frame change (current sampling)
Mean |change| per frame, raw / normalised by each feature's spread over all training frames.

| | bowing | crisscrossing | jumping | swinging |
| --- | --- | --- | --- | --- |
| max6 | 0.066 / 0.83 | 0.086 / 0.83 | 0.108 / 0.85 | 0.045 / 0.39 |
| mean6 | 0.052 / 0.83 | 0.063 / 0.72 | 0.076 / 0.82 | 0.044 / 0.47 |
| fine10 | 0.063 / 0.81 | 0.075 / 0.71 | 0.084 / 0.79 | 0.058 / 0.49 |
| orient8 | 0.106 / 0.92 | 0.083 / 0.66 | 0.097 / 0.75 | 0.086 / 0.54 |
| endpoint4 | 0.037 / 0.82 | 0.047 / 1.01 | 0.053 / 1.10 | 0.031 / 0.56 |

**Correction to section 14.** Relative to each feature's own spread, max6 doesn't compress motion more
than the alternatives do: about 0.4-0.85 spreads per frame for all of them. Its raw changes are small
because its spread is small. The section 14 mismatch is better stated this way: discrete pose switches are
large compared with the observed spread, and max6 can't see orientation.

### Shared temporal structure
Leave-one-actor-out aligned share of the static change cost, with the frame-shuffled null in brackets.
"Excess" is the mean over the four manners of real minus null.

| Interface | Sampler | bowing | crisscrossing | jumping | swinging | Excess |
| --- | --- | --- | --- | --- | --- | --- |
| max6 | current | 34% (25) | 42% (21) | 34% (13) | 16% (9) | +14 |
| | cycle1 | 36 (29) | 67 (22) | 46 (22) | 36 (19) | +23 |
| | cycle2 | 49 (15) | 67 (24) | 55 (18) | 35 (5) | **+36** |
| mean6 | cycle2 | 49 (18) | 58 (21) | 48 (14) | 41 (9) | +33 |
| fine10 | cycle2 | 40 (11) | 62 (17) | 38 (7) | 37 (6) | +34 |
| orient8 | cycle2 | 44 (16) | 49 (11) | 27 (7) | 38 (4) | +30 |
| endpoint4 | cycle2 | 31 (14) | 56 (18) | 42 (17) | 33 (7) | +27 |

Excess for every sampler:

| Interface | current | cycle1 | cycle2 |
| --- | --- | --- | --- |
| mean6 | +11 | +22 | +33 |
| fine10 | +9 | +26 | +34 |
| orient8 | +9 | +14 | +30 |
| endpoint4 | +11 | +14 | +27 |

Pairwise correlation after the best circular shift follows the same pattern. For crisscrossing under
cycle1 it is 0.81 for max6 and 0.60-0.74 for the others, against a null of about 0.35-0.45.

### Manner and actor predictability (current sampling, all 26 manners)
Nearest centroid on z-scored per-video summaries. Chance is 0.04 for manner and 0.07 for actor.

| Interface | Manner (leave actor out): static + dynamics / dynamics only | Actor (leave manner out): static + dynamics / dynamics only |
| --- | --- | --- |
| max6 | 0.84 / 0.83 | 0.22 / 0.15 |
| mean6 | 0.81 / 0.81 | 0.27 / 0.17 |
| fine10 | 0.85 / 0.81 | 0.27 / 0.19 |
| orient8 | 0.84 / 0.82 | 0.24 / 0.19 |
| endpoint4 | 0.83 / **0.72** | 0.24 / 0.17 |

### Sensitivity (swinging)
Feature change from the perturbation, in units of a typical frame-to-frame change.

| Interface | Translation x + 5% | Size × 1.1 |
| --- | --- | --- |
| max6 | 0.61 | 0.42 |
| mean6 | 0.62 | 0.34 |
| fine10 | 0.80 | 0.34 |
| orient8 | **0.24** | **0.13** |
| endpoint4 | 0.81 | 0.36 |

### Example: actor 01M swinging, right arm, cycle1 frames 0 / 2 / 4 / 6
The arm swings from down-left to up and back down. Shoulder → hand:
(0.44,0.41)→(0.38,0.57); (0.41,0.43)→(0.49,0.31); (0.46,0.47)→(0.50,0.33); (0.47,0.48)→(0.48,0.63).

| Interface | t0 | t2 | t4 | t6 | What it shows |
| --- | --- | --- | --- | --- | --- |
| max6 | L .28, Lo .14 | L .31, U .61 | L .08 | Lo .31 | Small, mixed changes. |
| orient8 | orientation (−.76, −.65) | (−.68, −.72) | (−.98, −.20) | (−.99, +.05) | The swing appears as a rotation of the principal axis, crossing zero on the second component. |
| endpoint4 | hand y .57 | .31 | .33 | .63 | The hand's up-down excursion directly. |

### Reading
1. **Sampling matters far more than the interface for shared structure.** Moving from current to cycle2
   roughly triples the excess over the null in every interface (+9-14 → +27-36). Among interfaces, the
   spread at cycle2 is small (+27 to +36), and the current max6 is not worse.
2. **Only orient8 and endpoint4 preserve relational limb geometry:** they tell / from \, and they show a
   swing as rotation or as an endpoint excursion.
3. **None of the alternatives is more actor-specific.** Leakage is 0.22-0.27 for every interface (chance
   0.07). Manner predictability is unchanged, about 0.81-0.85, except endpoint4's dynamics-only score
   (0.72).
4. **orient8 is the most invariant** to translation and size (about 3 times less sensitive than the rest).
   endpoint4 and fine10 are the most translation-sensitive.

### Recommendation for later re-scoring of the clock/memory-field programs
1. **orient8:** mean-pooled 3+3 fields plus a 2-dimensional orientation per part, 48 dimensions.
   - It is mirror-aware and shows a swing as rotation.
   - It is the least sensitive to translation and size, with no extra actor leakage, and keeps nearly all
     the shared structure at cycle2 (+30).
   - On the LoT side the same features come straight from the salient cells (their mean fields and
     principal axis), so the interface stays anonymous and symmetric.
   - An untested variant keeps the current max6 fields and only adds the 2 orientation dimensions.
2. **endpoint4:** the smallest, at 24 dimensions, and it shows limb excursions most directly.
   - Its proximal/distal ordering needs a reference point. Observed parts use the torso, but anonymous
     LoT outputs have no torso. They would need an anatomy-free rule, such as ordering the principal-axis
     ends by distance from the window centre. That asymmetry should be settled before using it.

Either should be combined with **cycle2** sampling, which the sampler comparison favours across the
board.

## 16. Clock + pose-readout programs re-scored with orient8 + cycle2 (diagnostic only)
Changes from section 14:
- **Interface:** orient8, with the same features on both sides. Observed parts use their skeleton points.
  LoT outputs use the centres of cells ≥ 0.75, which is the current salience rule. A symmetry check
  passes: a straight / or \ gives the same orientation signs from skeleton points and from rendered
  cells.
- **Matching:** the committed matched likelihood (Hungarian per video, frames + changes, β 250) applied
  to these features.
- **Sampling:** cycle2.
- **Realization:** tempo r ∈ 1-4, and phase within one *orient8* cycle. orient8 sees the true swing period,
  which for 132 programs differs from what max6 saw: max6 halves it because it can't tell the mirrored
  poses apart.
- **Twin:** the best frame of the program's own cycle, held.
- **Programs:** 900 (300 pose pairs × 3 clocks). Pairs are now picked by orient8 signature and include
  every mirror pair available (8 pairs, 144 swap rows per manner).
- **Periods in orient8:** 5 (51 programs), 10 (249), 14 (600).

Script: `orient_clock.py`.

| Manner | Moving beats twin (marg / max) | Median marg − twin: real (null) | Best marg − twin: real (null) | Mirror-pair rows, median | Best moving swap vs MAP |
| --- | --- | --- | --- | --- | --- |
| bowing | 0 / 0 | -85.5 (-79.8) | -26.0 (-25.7) | -91 | +96* |
| crisscrossing | 0 / 0 | -101.9 (-99.7) | -23.9 (-23.8) | -96 | -26 |
| jumping | 0 / 0 | -94.7 (-94.7) | -24.2 (-24.2) | -89 | -70 |
| swinging | 0 / 0.001 | -86.2 (-90.0) | **-15.6 (-21.5)** | -83 | -74 |

\*The diag4 MAP was fitted under max6, so under orient8 even static swaps beat it for bowing. This says
nothing about motion.

In every top swap, all 15 videos pick r = 1.

### Reading
1. **orient8 narrows the gap about 2.5- to 4-fold** (median -225…-263 → -86…-102; best -99…-111 →
   -16…-26). But a moving program still never beats its own best static pose.
2. **Real data and the null are almost the same.** The only hint of genuine temporal fit is swinging, where
   real beats null by about 4 nats (median) and 6 nats (best). Crisscrossing and jumping are identical to
   the null.
3. **Mirror swings, now visible, don't help** (median -83 to -96). A switch between two whole-limb poses,
   with blended frames in between, still doesn't resemble the gradual rotation of a real arm. In the
   observed data, the orientation features rotate smoothly over several frames (section 15 example).
4. **What remains is the motion's shape, not the interface or the timing.** The hand-built action field
   switches between two static poses. A real limb sweeps continuously. A program would need graded
   intermediate poses: more scanners on the memory track, each gating its own intermediate pose, or a
   motion whose output rotates gradually.

Whether v6 can express that compactly enough to be found is the next question. It would take either a
richer hand-built family (3-4 intermediate poses per cycle) or a search under the orient8 + cycle2 +
latent-tempo score. No MCMC was run.

## 17. Richer multi-pose clock family + a search under the tempo/phase score (diagnostic only)

### 17a. Richer family: 3-4 graded poses per cycle
Program structure:
- **Clock:** the 2-strip body shuttle.
- **Readouts:** 3-4 position-selective observers (top/upper/center/lower/bottom/periphery H patterns) spread
  over the two strips. Each one is a dashed source gating one pose plan.
- **Poses:** drawn from a pool of (plan T C), where T includes rotations (top/V*, top/H*, …) and both
  diagonals.
- **Kinds:** half the programs order their poses by orientation along the track ("ordered": a pendulum
  such as \ → | → / → |); half are random.
- **Probe example:** readouts top/H and bottom/H on the left and center/H on the right, gating /, | and \,
  give an orientation cycle (0,+.9) → (0,0) → (0,−.9) → … with period 14.
- **Legality:** every program in this family exceeds the 3-children limit, so it is a counterfactual on
  legality only. The engine is unchanged.

**Scoring** as in section 16: orient8 + cycle2, tempo r 1-4, phase within one orient8 cycle, marginal;
twin = best frame of the own cycle; frame-shuffled null.

**Size:** 1200 programs, all periodic in orient8; 7056 swap rows per manner.

| Manner | Moving beats twin: real (null) | Median: real (null) | Best: real (null) | Per row, real − null: median / share > 0 |
| --- | --- | --- | --- | --- |
| bowing | 0 (0) | -102 (-97) | -0.6 (-0.6) | -3.9 / 0.16 |
| crisscrossing | 0 (0) | -124 (-123) | -0.6 (-0.6) | -0.2 / 0.34 |
| jumping | 0 (0) | -116 (-116) | -0.3 (-0.3) | -0.1 / 0.39 |
| **swinging** | **0.2% (0)** | -109 (-113) | **+3.7 (-0.5)** | **+1.8 / 0.80** |

**Swinging, the first positive evidence of rewarded dynamics:**
- **16 swaps beat their static twins (+2.0 to +3.7), and all 16 are ordered pendulums.** On the null, the
  same swaps lose by 1.2-4.6.
- **They genuinely move.** orient8 changes by 0.055-0.073 per frame, close to the observed 0.086.
- **They use faster replay.** Their videos pick r = 2 or 4, rather than r = 1 everywhere as in section 12.
- **Across all swinging rows,** real beats null in 80% of cases (median +1.8 nats), for ordered and random
  programs alike.

The other three manners show no such effect. For crisscrossing and jumping, real and null are
indistinguishable. For bowing, the null is better, as you'd expect for a non-periodic manner.

**Reading.** The effect is small: a few nats, against a moving-minus-twin median of about -110. But it is
consistent, it is specific to swinging, and it needs several graded poses plus a free tempo. The
two-pose programs of sections 14 and 16 never reached it. So cycle sampling, orient8, latent tempo and a
phase-coded multi-pose readout together *can* make the existing matched likelihood reward a real periodic
limb, though barely, and only for the most pendulum-like manner.

The near-zero best scores (-0.3 to -0.6) in the other manners come from programs whose poses barely
differ, so they are nearly static.

### 17c. Translation family + per-video difference-in-differences (10 shuffles, actor bootstrap)
DiD per video = (moving − static | real) − mean over 10 shuffles of (moving − static | shuffled).
- "Static" is the program's best frame (chosen by total score), held.
- Scores are marginal over tempo r 1-4 and phase within one orient8 cycle; orient8 + cycle2.
- The 95% confidence interval resamples actors (videos).
- **Held-out check:** the top 20 rows are picked on actors 1-8 by moving − static on real data, then
  evaluated on actors 9-15, so selection doesn't flatter them.

Families (all periodic in orient8: 1043 of 1050):
- **rich:** 146 of the section 17a rotation pendulums;
- **xorder:** 3-4 poses ordered left to right along the track (translation);
- **yorder:** poses ordered top to bottom (a bounce);
- **antix:** two outputs in anti-phase, the same x-ordered poses forward and reversed, swapped jointly into
  two MAP slots.

| Manner | Family | Mean DiD per row (sum over 15 videos), 95% CI over actors | Videos with DiD > 0 | Held-out top 20: DiD / still beat static |
| --- | --- | --- | --- | --- |
| swinging | rich | +2.3 [+0.9, +4.1] | 0.55 | +0.6 / 0.10 |
| swinging | xorder | +2.6 [+1.0, +4.1] | 0.57 | +1.2 / 0 |
| swinging | yorder | +2.9 [+1.3, +4.8] | 0.58 | +1.1 / 0 |
| swinging | antix | **+4.3 [+1.8, +6.9]** | 0.64 | +2.2 / 0 |
| crisscrossing | rich | -0.2 [-0.5, +0.1] | 0.41 | 0.0 / 0 |
| crisscrossing | xorder | -0.8 [-1.1, -0.3] | 0.35 | 0.0 / 0 |
| crisscrossing | antix | -1.4 [-2.0, -0.8] | 0.33 | -0.4 / 0 |
| jumping | rich / xorder / yorder | +0.1 / -0.2 / -0.1 (all CIs span 0) | 0.41-0.45 | about 0 / 0 |
| bowing | rich / yorder | -3.5 [-5.3, -1.7] / -4.4 [-6.4, -2.6] | 0.25-0.27 | -0.0 / -0.8, 0 |

### Reading
1. **The swinging effect survives the stricter test.** With 10 shuffles averaged and actors resampled,
   every family's DiD is positive and excludes 0, and it stays positive on held-out actors. But it is
   **not specific to rotation.** Translation and bounce families earn as much, and the anti-phase pair
   earns most.
   - So what swinging rewards is *ordered periodic change at a fitted tempo*, not the pendulum geometry
     as such.
   - Part of this is that shuffling swinging doesn't make it jumpier (real ÷ shuffled change ≈ 1.0).
     For the other manners, shuffling does make the data jumpier, which tilts their DiD downward.
2. **Translation doesn't rescue crisscrossing or jumping.** Crisscrossing's translation and anti-phase
   families are *more* negative than the rotation family. Jumping stays at 0. So the earlier explanation,
   that these manners translate while the family only rotated, isn't enough.
   - **The likely culprit is abruptness.** Every family switches between discrete poses. Real
     crisscrossing and jumping are smooth (84% and 91% of the shuffled change). Abrupt switches resemble
     shuffled frames more than real ones, whatever the direction of motion.
3. **Held-out, no family beats its own static explanation.** One exception: 10% of the held-out swinging
   rotation programs do. The temporal reward is real but small (about 1-4 nats over 15 videos). Each
   program's static mismatch is much larger.

**Implication.** The next limit is *graded* motion. The clock-and-pose architecture produces stepwise
changes. Smooth limb trajectories would need either many finely spaced poses (more readouts) or a
continuous mechanism: for example, motion of a limb-shaped body along the track, read through a window,
instead of discrete pose switching.

### 17d. Feature-space tweening as part of the realization (geometry interpolation; diagnostic only)
**The tween.**
- The program stays discrete.
- Its orient8 output (tick level) is smoothed by a causal moving average over k video frames, so a state
  change A → B becomes a linear ramp over k frames.
- That ramp is linear interpolation of the position fields and of the doubled-angle orientation. It
  interpolates geometry; it doesn't blend images.
- k ∈ {1, 2, 3, 4} is a realization variable, marginalised together with tempo and phase. "Rigid" means
  k = 1 only.

**Same setup as 17c:** cycle2, 10 shuffles, actor bootstrap, held-out actors.

**Speed.** The scoring now runs **28-113× faster** with exact optimal matching (`fastlik.SwapLik`). Only the
swapped channels vary across options, so the best assignment of the unchanged channels is precomputed for
every set of leftover columns. The results match the previous scorer to within 1e-5, and all 1189 programs
(full families) were scored.

**Results.** Each cell is DiD summed over 15 videos (95% CI over actors), then the share of all rows that beat
the static twin, then the held-out top-20 win rate.

| Manner / family | Rigid | Tween |
| --- | --- | --- |
| swinging / rich | +0.89 [+0.20, +1.70]; 1.0%; held-out wins 0.65 | +0.20 [+0.07, +0.37]; 6.8%; 1.00 |
| swinging / antix | +2.74 [+0.99, +4.46]; 0.3%; 0.45 | +0.39 [+0.16, +0.64]; 7.1%; 1.00 |
| swinging / xorder, yorder | +1.4, +1.6 (CI > 0); 0.4-0.7%; 0.45-0.65 | +0.25, +0.30; 3.5-5.1%; 1.00 |
| crisscrossing / rich, xorder, antix | -0.4, -0.9, -1.5 (CI < 0); about 0%; 0-0.10 | +0.02, +0.05, +0.04 (about 0); 3.6-4.4%; 1.00 |
| jumping / rich, xorder, yorder | -0.1, -0.3, -0.3 (CIs span 0); about 0%; 0.05-0.45 | +0.03, +0.08, +0.06; 4.1-7.1%; 1.00 |
| bowing / rich, yorder | -2.5, -3.2 (CI < 0); 0.5%; 0.25-0.45 | -0.22, -0.18; 7.3-12.3%; 0.85-1.00 |

Median moving − static roughly halves with tweening (e.g. crisscrossing anti-phase: -121 → -71). Most videos
pick k = 3 or 4.

**Reading.**
1. **Tweening makes discrete programs beat their single-frame static twins often,** in every manner,
   held-out included.
2. **But the gain is the same on shuffled frames, so DiD falls to about 0.** It even appears for bowing,
   which isn't periodic. So the gain doesn't come from temporal order.
3. **The likely cause is blending.** A heavily tweened program partly *blends* several poses, and a blend
   can fit better than any single held pose. The single-frame twin is then an unfair comparator.
4. **Swinging keeps a positive DiD under tweening,** but a smaller one: averaging dilutes the ordered
   contrast.

The fair test gives the static side the same blending: the cycle mean and every k-frame blend, held still.
It is running and is reported in 17e.

### 17e. Tweening with a fair static comparator
The static side may now also blend: the cycle mean, and every k-frame blend of the cycle's poses, held
still. This is the strongest static explanation the same program could give.

**Result: in every manner and family, no row beats its static explanation (0.000), and held-out win rates
are 0.00.** Medians stay at -40 to -130. The tween's earlier wins (17d) came entirely from blending poses
into an average, not from time.

**DiD is unchanged by this, and that is expected.** A static prediction scores the same frame term on real
and shuffled data (the same set of frames). Its change term equals the data's own change energy, whatever
the static vector is. So the static twin cancels out of DiD exactly: DiD isolates the contribution of
temporal *order*, independent of which static comparator is used.

**Reading.** Tweening doesn't make the matched likelihood reward discrete state structure. It mainly lets
a moving program imitate a blended static pose. Temporal order is still rewarded only for swinging, by
about 0.2-0.4 nats with tween and 0.9-2.7 rigid.

### 17b. Search under the orient8 + cycle2 + latent tempo/phase score (scratch likelihood; v6 grammar)
Setup:
- tempered MH: 8000 iterations × 12 replicas, 4 chains per condition, about 3.4 h per chain;
- swinging and crisscrossing, on real observations and on frame-shuffled (null) observations;
- the score marginalises tempo r 1-4 and a circular phase over 16 frames.

| Condition | R-hat | Acceptance | Posterior mass that moves* | Tempo chosen by the top 200 programs (r = 1/2/3/4) | MAP: moving − its best static frame |
| --- | --- | --- | --- | --- | --- |
| swinging, real | 1.01 | 0.16 | **0.26** | 1292 / 351 / 436 / 921 | 0.0 (the MAP is static) |
| swinging, null | 1.01 | 0.18 | 0.00 | 2475 / 36 / 266 / 223 | 0.0 |
| crisscrossing, real | 1.01 | 0.17 | 0.00 | 2551 / 211 / 194 / 44 | 0.0 |
| crisscrossing, null | 1.00 | 0.19 | 0.00 | 1436 / 1125 / 342 / 97 | 0.0 |

\* The share of the top-200 posterior programs whose orient8 output changes by more than 0.02 per frame
after start-up, at each program's chosen tempo.

**Reading.**
- These are the first converged runs of the diagnostic (R-hat about 1.01).
- **Under this score, a quarter of the swinging posterior genuinely moves, and none of it does on
  shuffled data.** That matches the per-video DiD: swinging carries rewardable order, and the search can
  find programs that use it.
- The MAP is still static, and crisscrossing has no moving posterior mass at all. So motion is
  competitive in swinging but not dominant, and absent in crisscrossing.
- These runs used the v6 grammar. Whether v7 (separate grouping and agency) shifts this is the obvious
  follow-up.
