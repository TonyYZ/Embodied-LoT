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
