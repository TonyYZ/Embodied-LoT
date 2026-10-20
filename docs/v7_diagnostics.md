# v7 diagnostics (four manners, training actors only)

The manners are bowing, crisscrossing, jumping and swinging, run as `grace_lot.inference.run --split train`.
Every run uses 4 chains × 12 temperature replicas and 10,000 iterations, with a burn-in of 2,500. The test
split stays sealed.

Each table reports:
- **R-hat:** the convergence of the log-posterior across chains; below about 1.1 counts as converged.
- **Distinct programs:** how many different programs appear among the 1,000 retained samples. 1,000
  means the chains never revisit a program.
- **Mass of the top program:** the share of samples that are the single most frequent program.
- **Median log-lik / log-prior of samples,** next to the same for the MAP (the best single program found).
- **Samples that move:** the share of 150 sampled programs whose output features change after start-up,
  in some unknown-motion state.
- **field / body / template:** the share of sampled programs that use each node.

## Why the first run was rerun
In the first v7 diagnostic, the likelihood had too little weight. The temperature 1/(T·F) = 1/128 turns
each video's fit into an average per frame and detector.
- **For any single program, the prior wins.** Richer programs fit better than the empty one by about +5
  to +35 nats, but pay about −50 nats of prior, so the MAP is a near-empty program.
- **For the posterior as a whole, sheer numbers win.** λ = 0.5 doesn't counter how fast the number of
  programs grows with size, so the chains drift among about 1,000 large programs and never converge.

The reruns set λ = 1, a proper description-length prior, and vary the likelihood weight: temperature
1/16 (per frame), 1 (the full Gaussian) and 1/128 (the original).

### Baseline: λ = 0.5, temperature 1/(T·F) = 1/128 (first v7 diagnostic)

λ = 0.5, temperature = default 1/(T F), 10000 iterations (burn-in 2500), 4 chains × 12 replicas

| Manner | R-hat | Distinct programs | Mass of the top program | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move | field / body / template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.48 | 1000 | 0.001 | 94.6 / -57.4 | 78.9 / -4.3 | 0.01 | 0.51 / 0.49 / 0.91 |
| crisscrossing | 1.44 | 999 | 0.002 | 103.8 / -57.7 | 89.1 / -12.6 | 0.11 | 0.58 / 0.56 / 0.66 |
| jumping | 3.17 | 999 | 0.002 | 106.6 / -57.7 | 89.7 / -5.0 | 0.25 | 0.46 / 0.37 / 0.81 |
| swinging | 1.47 | 1000 | 0.001 | 74.9 / -58.6 | 71.7 / -3.0 | 0.00 | 0.45 / 0.53 / 0.90 |

MAP programs:

- **bowing:** `(body out)`
- **crisscrossing:** `(template (serial_h (parallel bottom/H*)) out) neutral/V`
- **jumping:** `bottom/H out`
- **swinging:** `out`

### Run 1: λ = 1, temperature 1/16 (per frame)

λ = 1.0, temperature = 0.0625, 10000 iterations (burn-in 2500), 4 chains × 12 replicas

| Manner | R-hat | Distinct programs | Mass of the top program | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move | field / body / template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.52 | 990 | 0.002 | 873.0 / -88.7 | 953.1 / -62.8 | 0.01 | 0.63 / 0.50 / 0.79 |
| crisscrossing | 1.38 | 988 | 0.002 | 1004.5 / -90.2 | 1085.2 / -82.3 | 0.00 | 0.44 / 0.47 / 0.95 |
| jumping | 1.18 | 993 | 0.002 | 919.4 / -85.8 | 1080.2 / -69.4 | 0.07 | 0.43 / 0.60 / 0.82 |
| swinging | 1.18 | 1000 | 0.001 | 807.4 / -76.8 | 858.6 / -58.6 | 0.17 | 0.41 / 0.45 / 0.61 |

MAP programs:

- **bowing:** `(field (parallel out out (parallel out center/V center/H))) (body (template (parallel out out) (template bottom/V center/H center/H))) | .0.0.2>.0 .1.0.1.0>.0.0.2`
- **crisscrossing:** `center/H* (template (template top/H* (parallel out out out) (serial_v periphery/V out)) (parallel (serial_h periphery/H* center/V void/V) (serial_v out out) upper/V)) | .0>.1.0 .1.0.1>.1.1.0.0 .1.1.2>.1.1.0.1`
- **jumping:** `periphery/V top/V (parallel (serial_v void/H* out) (parallel (parallel out center/H) center/V out) (template (body out) (serial_v out upper/H universal/H) center/H))`
- **swinging:** `(field center/V) (field (parallel (serial_v out out) (parallel out out out) (serial_v void/H center/H periphery/H))) | .0.0>.1.0.2.1 .1>.1.0.2.1 .1.0.2.2>.1.0.2.1`

**Reading.**
- **The likelihood now shapes the programs.** The best programs are structured: about six output windows
  each, matched to parts, using fields, bodies and templates.
- **Measure trees appear.** Swinging's MAP has a separate `(field center/V)` tree that gates the other
  tree through a dashed edge.
- **Convergence improves but is not reached.** R-hat is 1.18 for jumping and swinging, against 1.47 and
  3.17 before, but still above 1.1. The chains still see about 990-1000 distinct programs.
- **Motion appears where the audit expected it.** 17% of swinging's samples move, against 0% before.
  Jumping is at 7%, and bowing and crisscrossing stay at about 0.

### Run 2: λ = 1, temperature 1 (full Gaussian)

λ = 1.0, temperature = 1.0, 10000 iterations (burn-in 2500), 4 chains × 12 replicas

| Manner | R-hat | Distinct programs | Mass of the top program | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move | field / body / template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.10 | 985 | 0.002 | 15040.8 / -95.9 | 15980.2 / -90.0 | 0.20 | 0.53 / 0.63 / 0.41 |
| crisscrossing | 1.37 | 984 | 0.002 | 16239.6 / -84.7 | 18391.6 / -84.0 | 0.04 | 0.36 / 0.35 / 0.45 |
| jumping | 2.27 | 992 | 0.002 | 15359.5 / -81.8 | 17168.9 / -54.7 | 0.06 | 0.52 / 0.53 / 0.46 |
| swinging | 1.67 | 997 | 0.002 | 13894.3 / -89.4 | 15633.8 / -87.4 | 0.00 | 0.51 / 0.48 / 0.68 |

MAP programs:

- **bowing:** `center/V (parallel out (parallel out out) (serial_v (serial_v bottom/V center/V neutral/H) center/H periphery/V*)) (body (parallel out (serial_v center/H out) (parallel bottom/V out))) | .0>.1.2.1 .0>.2 .1.2.0.2>.2`
- **crisscrossing:** `(template (body out) (template top/V (field out) out) (serial_v void/H void/V (template bottom/V center/H center/H))) (parallel (parallel center/H out out) out center/V) | .0>.1 .0.2.2.1>.1`
- **jumping:** `out (parallel out (parallel out (serial_v void/V center/H out) out) (serial_h void/V (template periphery/H top/H) void/H))`
- **swinging:** `upper/V bottom/V (template (parallel out out (serial_v out center/H)) (field (serial_h out)) (serial_v void/V (serial_h bottom/V center/H void/H) (parallel neutral/V* void/V neutral/V))) | .2.2.1.0>.2.2.2.1 .2.2.2.2>.2.2.2.1`

**Reading.**
- **The likelihood now dominates.** The scale is about 15,000 nats, against a prior of about −90.
- **Mixing gets worse, not better.** Only bowing reaches R-hat 1.10; the others are at 1.37-2.27, worse
  than run 1 for jumping and swinging.
- **The chains sit far from the best fit they ever reach.** Typical samples are 900-2,100 nats below the
  MAP's log-likelihood, and nearly every sample is still a distinct program. That pattern fits chains
  stuck in separate local regions of a very sharp posterior.
- **The temperature ladder is mis-scaled for this likelihood.** It is set from `beta`, which the Gaussian
  likelihood doesn't use. Its hottest replica still weights the likelihood by about 0.005 × 15,000 ≈ 80
  nats, so even that replica stays sharp and can't move freely between those regions.
- **Motion moves to bowing.** 20% of bowing's samples move, swinging's drop to 0%, and crisscrossing and
  jumping are at 4-6%. With chains this unconverged, those shares shouldn't be over-read.

### Run 3: λ = 1, temperature 1/128 (the original likelihood weight)

λ = 1.0, temperature = 0.0078125, 10000 iterations (burn-in 2500), 4 chains × 12 replicas

| Manner | R-hat | Distinct programs | Mass of the top program | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move | field / body / template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.00 | 1000 | 0.001 | 98.0 / -76.4 | 78.9 / -6.1 | 0.01 | 0.66 / 0.55 / 0.71 |
| crisscrossing | 1.17 | 1000 | 0.001 | 109.3 / -84.8 | 89.1 / -6.1 | 0.02 | 0.63 / 0.62 / 0.71 |
| jumping | 1.01 | 998 | 0.002 | 90.2 / -60.8 | 89.7 / -6.1 | 0.01 | 0.54 / 0.51 / 0.49 |
| swinging | 1.05 | 991 | 0.005 | 79.3 / -61.2 | 71.7 / -6.1 | 0.00 | 0.56 / 0.54 / 0.58 |

MAP programs:

- **bowing:** `out`
- **crisscrossing:** `out`
- **jumping:** `out`
- **swinging:** `out`

**Reading.**
- **λ = 1 fixes convergence at the original weight.** R-hat is 1.00-1.17 in every manner, against
  1.44-3.17 at λ = 0.5.
- **But what converges is an almost data-free posterior.** Every MAP is the bare `out`, and samples are
  still about 1,000 distinct mid-sized programs. At temperature 1/128 the likelihood is too weak to select
  structure, whatever the prior.

## Summary of the three reruns

| Likelihood weight (λ = 1) | Convergence | Best programs | Samples that move |
| --- | --- | --- | --- |
| 1/128 (original) | converged (R-hat 1.00-1.17) | empty (`out`) | about 0-2% |
| **1/16 (per frame)** | partial (1.18-1.52) | structured: 6 outputs, fields, bodies, measure trees | swinging 17%, jumping 7% |
| 1 (full Gaussian) | poor, except bowing (1.10-2.27) | structured, sharp | bowing 20%, swinging 0% |

**What this shows:**
- **The original weight was the main problem.** At 1/128 the data barely constrain the programs; λ only
  decides whether the chains agree on that near-prior posterior.
- **Per-frame weighting (1/16) is the most useful setting.** The data select structured programs, and
  motion shows up where the audit predicted it: swinging.
- **Full weighting (1) is too sharp for the current sampler.** The tempering ladder is set from `beta`,
  which the Gaussian likelihood doesn't use, so even its hottest replica stays sharp.

These next steps were run as Run 4, below.

## Run 4: the calibrated ladder, 40,000 iterations

Changes since Run 1, all in place for this run:
- **Calibrated ladder** (`inference.ladder = "calibrated"`). The hottest replica is set from the likelihood itself:
  32 prior programs are scored, and the hottest replica shrinks the spread of their log-likelihoods to about one nat
  (here tau_min ≈ 0.001, against 0.016 from `beta`).
- **Split R-hat.** Each chain is cut in half, so chains that agree but are still drifting are flagged. Runs 1-3
  above report the earlier, unsplit statistic. Rescored split, Run 1 is 1.53-2.51, not 1.18-1.52.
- **Semantics.** Moving bodies leave a full trail of known emptiness, which content moving in covers.
- **Speed.** Component rollouts and per-channel cost rows make each likelihood evaluation 2.6 times faster, with
  bit-identical rollouts.

### Run 4: λ = 1, temperature 1/16, calibrated ladder, full trails

λ = 1.0, temperature = 0.0625, 40000 iterations (burn-in 10000), 4 chains × 12 replicas. Wall time: 2.3 to 6 hours
per chain, on 16 workers.

| Manner | Split R-hat | Distinct programs | Mass of the top program | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move | field / body / template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.53 | 3915 | 0.001 | 979.5 / -91.8 | 1015.6 / -76.6 | 0.12 | 0.49 / 0.37 / 0.63 |
| crisscrossing | 2.46 | 3851 | 0.001 | 1121.9 / -86.3 | 1149.0 / -75.0 | 0.02 | 0.29 / 0.36 / 0.91 |
| jumping | 2.30 | 3880 | 0.001 | 1102.5 / -83.8 | 1157.8 / -74.7 | 0.03 | 0.52 / 0.55 / 0.95 |
| swinging | 1.31 | 3956 | 0.001 | 982.3 / -82.0 | 1029.4 / -74.3 | 0.00 | 0.51 / 0.41 / 0.71 |

MAP programs:

- **bowing:** `bottom/H (parallel (parallel center/H (serial_v neutral/V out) out) (parallel center/V out out) (template (serial_v top/H out) center/H (serial_h bottom/H lower/V out))) | .1.2.1>.1`
- **crisscrossing:** `(template (template (body out) (serial_v periphery/H out) out) (parallel (serial_v neutral/V out) out out) (serial_v void/V (template center/H center/H universal/H) (parallel void/H bottom/V)))`
- **jumping:** `(parallel (serial_v void/H neutral/V) out) (parallel (parallel out out (serial_v void/H out)) (field (template center/H center/V center/H)) (parallel out (serial_v center/H* out)))`
- **swinging:** `(parallel (template (parallel out out) (body out) (template center/H center/H center/V)) (serial_v out (parallel out out)) (template bottom/V void/H)) periphery/H | .0.0.2.1>.0.0.1 .1>.0.0.0`

Mean log-posterior of each chain over the four quarters after burn-in, and its swap acceptance between neighbouring
replicas (pair k joins replicas k and k + 1; tau falls from 1 by a factor of about 0.54 per step):

| Manner | Chain means by quarter | Swaps, pairs 0-4 | Swaps, pairs 5-10 |
| --- | --- | --- | --- |
| bowing | 858→892, 845→927, 824→899, 862→896 | 0.01-0.65 | 0.95-0.99 |
| crisscrossing | 1026→1036, 949→1055, 936→1045, 963→1033 | 0.01-0.85 | 0.95-0.99 |
| jumping | 1019→1047, 930→1076, 852→1004, 939→1025 | 0.00-0.95 | 0.94-0.99 |
| swinging | 817→916, 831→887, 872→910, 848→942 | 0.00-0.94 | 0.98-0.99 |

**Reading.**
- **Not converged, and still climbing.** Split R-hat is 1.31-2.46. Most chains' means still rise after burn-in
  (jumping chain 2 by 150 nats over 30,000 iterations), and chains that level off do so at different heights
  (bowing: 892 against 927). The chains are both non-stationary and in different modes.
- **The calibrated ladder is too hot.** Six of the twelve replicas (tau below about 0.05) swap 95-99% of the time:
  they all sample nearly the prior, so they add cost without moving programs. The cold end (tau 1 to 0.16) swaps
  only 1-20% of the time, so programs found by hot replicas rarely reach tau = 1. A spread of one nat at the
  hottest replica was the wrong target; the useful range of the ladder is tau = 1 to about 0.05.
- **Better fits, almost no motion.** Samples fit about 100 nats better than Run 1's (median log-lik 980-1120
  against 810-1000), but almost none move (0-12%; swinging 0%, against 17% in Run 1). Static programs with
  templates and five or six output windows win. This matches the audit: without latent phase, motion cannot be
  rewarded across videos whose cycles are not aligned. The full-trail semantics may also cost moving programs
  (vacated cells read as known empty); this run cannot separate the two.
- **Cost.** Late programs take about 36 ms to score, against about 9 ms early on, so iterations slow down as the
  chains find larger programs: the first 2,000 iterations took 2-3 minutes, the last ones 6-8 minutes per 1,000.

**Next steps, not yet run:**
1. **Space the ladder by its swap rates.** Adapt the spacing during burn-in so that neighbouring replicas swap at
   similar rates (a standard adaptive parallel-tempering scheme; adaptation stops at burn-in, so the sampler stays
   exact afterwards), with tau_min near 0.05 and the replicas concentrated near tau = 1.
2. **Add latent phase** (and probably tempo) to the v7 likelihood, so motion can be rewarded.
3. **Separate the trail semantics from the missing phase** with a short counterfactual: score the Run 1 and Run 4
   moving samples with and without full trails.
4. **Use checkpoints** (`python -m grace_lot.inference.run --status <run dir>`) to stop the next run once split
   R-hat is below 1.1, instead of running a fixed number of iterations.
5. **Run all 26 manners** once one setting converges.

## Synthetic recovery: is it the sampler and language, or the fit to real data?

`python -m grace_lot.inference.recovery --config configs/grace_v7_parts.toml --n 4 --actors 15 --from-run
results/grace_v7_parts/diag5_l1_t16_trail --manners bowing crisscrossing jumping swinging --set
inference.iterations=10000 inference.burn_in=3000 inference.chains=2 inference.lambda_size=1.0 inference.temperature=0.0625`
(results in `results/grace_v7_parts/recovery_v7/`).

**Design.** Eight known programs: Run 4's four MAPs (21-23 nodes, the size real fits reach) and four moving prior
draws. For each, 15 synthetic actors are drawn from the v7 likelihood's own generative model: a motion realization
by its prior weight, a board per output channel (claimed cells figure, unclaimed cells filled from a real part board,
fresh randomness per actor), scan features, Gaussian noise at the real manners' level, and the channels shuffled as
parts. Phases are aligned. Inference is exactly the v7 setting (adaptive ladder, temperature 1/16, λ = 1),
10,000 iterations, 2 chains. If inference cannot recover programs from data they generated, the trouble is search
(sampler and language), whatever the real-data fit.

| True program | Nodes | Truth log-post | Best found (each chain) | Same rendering? | Split R-hat | Hours |
| --- | --- | --- | --- | --- | --- | --- |
| bowing MAP | 22 | 512.3 | 509.9, 488.5 | no | 1.47 | 6.6 |
| crisscrossing MAP | 23 | 455.1 | 400.0, 404.1 | no | 2.28 | 1.6 |
| jumping MAP | 23 | 618.2 | 576.9, 533.4 | no | 1.92 | 7.5 |
| swinging MAP | 21 | 679.4 | 663.6, 682.8 | no | 1.35 | 3.0 |
| prior draw | 20 | 85.5 | 130.2, 134.2 | no | 1.06 | 5.7 |
| prior draw | 23 | 26.6 | 93.0, 88.0 | no | 1.01 | 3.6 |
| prior draw | 5 | -10.1 | 3.0, -2.0 | yes (`out`) | 1.00 | 1.0 |
| prior draw | 15 | 69.7 | 111.4, 111.4 | no | 1.04 | 1.2 |

**Reading.**
- **Search fails on realistic programs.** For three of the four MAP-sized truths no chain reaches the truth's
  log-posterior: 2, 51 and 41 nats short. The fourth finds a different program scoring 3 nats better. The two chains
  end 5-44 nats apart, and split R-hat is 1.35-2.28: the same non-convergence as on real data, here with data the
  model generated itself and aligned phases. So the real-data non-convergence does not need a misfit likelihood to
  happen. **The bottleneck is search over large programs**: the moves and the brittleness of the language.
- **Small posteriors converge.** For the prior draws, chains agree (split R-hat 1.00-1.06) and beat the truth by
  40-65 nats with much smaller programs. Most of a random program's detail (neutral leaves, invisible fields, edges
  that change nothing) leaves no trace in the features, so the posterior rightly prefers a shorter description:
  this is Occam's razor, not failure.
- **Together:** the language's large programs sit in a landscape the current local moves cannot cross, even under
  tempering. When the posterior is on small programs, sampling is fine.

**What this points to (not yet done):**
1. **Larger moves.** Moves that rewrite whole subtrees or trees in one step (for example regenerating a whole tree of
   a forest, or swapping trees between programs held by different replicas) and moves aware of the matching
   (re-pointing an output at a different part).
2. **Better starting points.** Start chains from data-driven programs (one output per part, fitted greedily), so
   search begins near the high-scoring region.
3. **A less brittle language.** Fewer ways to write the same rendering, and smaller jumps in output for one-node
   edits.
4. **Model-side, independently:** latent phase for motion, and a weaker temperature if the posterior stays this sharp.

## Run 5: the adaptive ladder

λ = 1, temperature 1/16, 40,000 iterations (burn-in 10,000, during which the ladder adapts), 4 chains × 12 replicas,
`inference.ladder = "adaptive"` with tau from 1 to 0.05. About 11 hours: the recovery experiment shared the
machine for its first 8.

| Manner | Split R-hat | Distinct programs | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move | field / body / template |
| --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.82 | 3903 | 996.4 / -95.8 | 1039.9 / -79.9 | 0.05 | 0.34 / 0.41 / 0.73 |
| crisscrossing | 1.66 | 3939 | 1089.4 / -88.0 | 1141.0 / -80.2 | 0.00 | 0.39 / 0.47 / 0.62 |
| jumping | 1.99 | 3899 | 1118.9 / -86.0 | 1140.7 / -75.5 | 0.08 | 0.47 / 0.41 / 0.89 |
| swinging | 1.68 | 3936 | 977.0 / -84.9 | 1037.7 / -74.1 | 0.00 | 0.42 / 0.52 / 0.75 |

Chain means of the log-posterior over the four post-burn-in quarters:

| Manner | Chain 0 | Chain 1 | Chain 2 | Chain 3 |
| --- | --- | --- | --- | --- |
| bowing | 902 → 929 | 880 → 892 | 867 → 925 | 867 → 948 |
| crisscrossing | 982 → 1012 | 951 → 1022 | 962 → 997 | 964 → 1042 |
| jumping | 944 → 941 | 1043 → 1049 | 1022 → 1045 | 943 → 1007 |
| swinging | 864 → 902 | 784 → 872 | 919 → 936 | 808 → 852 |

**Reading.**
- **No better than Run 4.** Split R-hat is 1.66-1.99 (Run 4: 1.31-2.46), fits are about as good, and motion stays at
  0-8%.
- **Separate modes, not just slow climbing.** Jumping's chain 0 sits flat at about 942 for 30,000 iterations while
  chain 1 sits at about 1047: a 100-nat gap that tempering never bridged. Other chains are still climbing at the end.
- **The ladder adapted only partly.** The middle of the ladder evened out, but in most chains the last gap (down to
  tau = 0.05) stayed wide, with swap rates near 0 there: 10,000 iterations of shrinking adaptation steps were not
  enough to settle it while the chains were still climbing.
- **Consistent with the recovery experiment:** swaps were never the main problem. The moves cannot reach better
  regions of large programs, on synthetic or real data alike.
