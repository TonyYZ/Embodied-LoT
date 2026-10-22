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

## Benchmark: crossover and data-driven starts

Recovery of Run 4's four MAP programs (same synthetic data for every variant, as in the recovery experiment),
5,000 iterations (burn-in 1,500, adaptive ladder), 2 chains per program; `results/grace_v7_parts/bench_moves/`.
- **crossover** (`inference.crossover = 0.5`): on half the sweeps, two neighbouring replicas exchange one subtree
  each (possibly a whole tree of the forest), accepted by the joint tempered rule. Exact (toy test).
- **starts** (`inference.start = "parts"`): each replica starts from a forest of trees fitted greedily, one per
  observed part (`grace_lot/inference/starts.py`). Only the burn-in changes.

| Variant | Best found minus truth: mean / median | Last-quarter chain mean minus truth | Split R-hat (mean) | Crossovers accepted |
| --- | --- | --- | --- | --- |
| baseline | -72.2 / -87.7 | -110.1 | 1.50 | - |
| crossover | -58.2 / -53.8 | -93.6 | 1.55 | 19% |
| crossover + starts | -46.9 / -48.7 | -86.6 | 1.38 | 17% |

Per program (best found minus truth, chain 1 / chain 2):

| Program | baseline | crossover | crossover + starts |
| --- | --- | --- | --- |
| bowing | -15 / -10 | -21 / -27 | -16 / **+1** |
| crisscrossing | -86 / -89 | -66 / -104 | -76 / -42 |
| jumping | -96 / -120 | -108 / -33 | -62 / -44 |
| swinging | -26 / -135 | -47 / -61 | -82 / -53 |

**Reading.**
- **Both help, modestly.** Crossover closes about 15 nats of the gap on average, and starts another 10. One chain
  (bowing, crossover + starts) reaches the truth's score. Crossovers are accepted 17-19% of the time, so pieces do
  travel between replicas.
- **Neither solves it.** After 5,000 iterations every variant is still 50-90 nats short on average, and the chains
  are still climbing. With 2 chains on 4 programs, differences of 10-15 nats are within the run-to-run spread.
- **Why starts are weak:** fitted one part at a time, a whole tree barely pays for its prior (about 4 nats of
  tempered fit for most parts against about 10 nats of prior for a 3-node tree), while good programs share trees
  across several windows. Starting programs score 20-90 nats above random ones, far below good programs.
- **So:** the landscape of large programs needs more than moves that recombine existing pieces. The remaining
  options are a less brittle language (fewer near-equivalent ways to write a rendering, smaller output jumps per
  edit), a weaker temperature, or fitting smaller programs per manner.

## Do chains in different modes agree on what matters?

Per-chain samples are not saved (posteriors pool the chains), so this uses each chain's best program: 4 chains in
each of Runs 4 and 5, so 8 independently found programs per manner, 32 in all. Model-side only: the human ratings
are not used, so no choice here is tuned on them.

**Structure.** Mean similarity of two chain-best programs:

| Measure | Same manner, different chains | Different manners | AUC (same vs different) |
| --- | --- | --- | --- |
| tree_unordered | 0.614 | 0.610 | 0.51 |
| flat | 0.254 | 0.245 | 0.52 |
| full | 0.527 | 0.521 | 0.54 |

**Behaviour.** Each manner's likelihood (temperature 1/16) scores all 32 programs:

| Manner's videos | Its own programs | Other manners' programs | AUC own vs other | Own programs beating every other |
| --- | --- | --- | --- | --- |
| bowing | 982 .. 1040 | 611 .. 1009 | 0.96 | 5 / 8 |
| crisscrossing | 1091 .. 1149 | 725 .. 1131 | 0.97 | 3 / 8 |
| jumping | 1023 .. 1158 | 846 .. 1149 | 0.77 | 1 / 8 |
| swinging | 946 .. 1038 | 710 .. 949 | 0.99 | 7 / 8 |

21 of the 32 programs fit their own manner's videos best.

**Reading.**
- **The modes agree on behaviour.** Whichever mode a chain lands in, its program fits its own manner far better than
  other manners' programs do (AUC 0.77-0.99).
- **They do not agree on structure.** Programs for the same manner are no more alike than programs for different
  manners (AUC 0.51-0.54, chance) on every tree measure. The data identify what programs render, not how they are
  written.
- **So non-convergence matters for the human comparison as it stands.** `analysis/run.py` compares manners by the
  structure of their programs (tree measures). With structure this arbitrary, those similarities would mostly
  reflect which mode each chain fell into.

**Options:**
1. **Normalise programs before comparing them:** remove what has no effect on the rendering (neutral fillers,
   gates and edges that change nothing, child order) and compare the normal forms. Cheap and model-side; it shows
   whether a rendering-preserving normal form makes structure identifiable.
2. **A less redundant language,** so that fewer different programs render the same boards.
3. **Behavioural similarity** (how well one manner's programs explain another manner's videos) is robust to the
   mode, but it is close to a kinematic baseline and gives up the structural claim of the LoT account.

### Normal forms do not help

Each of the 32 chain-best programs was simplified greedily: remove a dashed edge, remove a child subtree, or unwrap
a single-child composition, keeping an edit only if the program stays in the language and renders exactly the same
boards under every motion realization (channel order ignored).

| | Mean size | Structure AUC (tree_unordered / flat / full) | Distinct programs per manner |
| --- | --- | --- | --- |
| original | 21.3 nodes | 0.51 / 0.52 / 0.54 | 8 of 8 |
| normal form | 20.2 nodes | 0.52 / 0.51 / 0.55 | 8 of 8 |

**Reading.** Very little of each program is removable without changing what it draws, and the normal forms are
still no more alike within a manner than across manners. The chains' programs are not one rendering written in
different ways: they draw different boards that fit about equally well. So the structure is unidentified because
the data constrain it only weakly (8 features per part, matched to whichever output fits best, and no phase), not
because the language writes the same thing many ways. A less redundant notation alone would not fix it.

**What would:**
1. **A stronger simplicity prior** (a larger λ, or fewer windows or less depth), so only the simplest adequate
   programs remain. Recovery showed small posteriors converge; the question is whether manners still separate.
2. **Observations that constrain structure more:** latent phase and tempo, temporal features, and finer parts.
3. **Comparing concepts by behaviour,** which is robust (AUC 0.77-0.99), at the cost of the structural claim.

## Run 6: a stronger simplicity prior (λ = 2)

λ = 2, temperature 1/16, 30,000 iterations (burn-in 10,000, adaptive ladder), crossover 0.5, 4 chains × 12
replicas. About 15-20 minutes per chain (Run 5: 6-11 hours). Each chain's own posterior is now saved.

| Manner | Split R-hat | Distinct programs | Mass of the top program | Median log-lik / log-prior of samples | MAP log-lik / log-prior | Samples that move |
| --- | --- | --- | --- | --- | --- | --- |
| bowing | 1.09 | 1968 | 0.032 | 816.0 / -74.3 | 1006.7 / -155.7 | 0.03 |
| crisscrossing | 1.32 | 2265 | 0.024 | 895.7 / -100.3 | 1080.4 / -136.4 | 0.01 |
| jumping | 1.28 | 1929 | 0.084 | 818.8 / -73.9 | 1097.7 / -97.1 | 0.11 |
| swinging | 1.23 | 1414 | 0.170 | 651.9 / -56.3 | 842.7 / -95.2 | 0.00 |

Chain agreement (16 chains; model-side only):

| Measure | AUC on chains' best programs (Runs 4-5 at λ = 1) | AUC on chains' posteriors |
| --- | --- | --- |
| tree_unordered | 0.59 (0.51) | 0.62 |
| flat | 0.57 (0.52) | 0.70 |
| full | 0.57 (0.54) | 0.60 |

Behaviour AUC (each manner's likelihood, chains' best programs): bowing 0.98, crisscrossing 1.00, jumping 0.52,
swinging 0.90; 8 of 16 programs fit their own manner best.

**Reading.**
- **Much closer to convergence, and about 30 times cheaper.** Split R-hat 1.09-1.32 (λ = 1: 1.66-1.99); bowing is
  converged. Chain-best programs average 17 nodes (λ = 1: 21).
- **Structure becomes somewhat identifiable.** Same-manner chains are now more alike than different-manner chains
  (AUC 0.57-0.59 on best programs, 0.60-0.70 on posteriors, against 0.51-0.54 at λ = 1), but the separation is still
  weak.
- **Behaviour mostly holds,** except jumping, whose programs no longer fit jumping better than other manners do.
- **Fits are looser** (median log-lik 650-900 against about 980-1120 at λ = 1), and motion stays rare (0-11%).

## Run 7: λ = 2 with latent phase

As Run 6, plus `inference.phases = 8`: each video is scored at every circular shift of its frames within one cycle
(cycle2 sampling gives two cycles of 16 frames), averaged over the shifts. About 25 minutes in all.

| Manner | Split R-hat (Run 6) | Median log-lik / log-prior of samples | Samples that move (Run 6) |
| --- | --- | --- | --- |
| bowing | 1.42 (1.09) | 786.6 / -68.9 | 0.03 (0.03) |
| crisscrossing | 1.11 (1.32) | 931.6 / -98.8 | **0.21** (0.01) |
| jumping | 1.29 (1.28) | 868.4 / -85.9 | 0.15 (0.11) |
| swinging | 1.18 (1.23) | 796.9 / -87.1 | 0.00 (0.00) |

| Chain agreement | Run 7 | Run 6 |
| --- | --- | --- |
| Structure AUC, best programs (tree_unordered / flat / full) | 0.60 / 0.64 / 0.55 | 0.59 / 0.57 / 0.57 |
| Structure AUC, posteriors (tree_unordered / flat / full) | 0.54 / 0.73 / 0.49 | 0.62 / 0.70 / 0.60 |
| Behaviour AUC (bowing / crisscrossing / jumping / swinging) | 0.77 / 0.96 / 0.62 / 1.00 | 0.98 / 1.00 / 0.52 / 0.90 |

**Reading.**
- **Phase brings some motion back:** crisscrossing's samples move 21% of the time (Run 6: 1%), jumping 15%. Bowing
  and swinging stay static.
- **Fits improve** (median log-lik up 30-145 nats per manner, more than the few nats of generic best-start bonus
  phase gives every program), so the search finds programs that use the freedom.
- **Convergence and structure do not change** in a consistent direction: R-hat 1.11-1.42, structure AUC 0.49-0.73.
  Behaviour AUC moves both ways.

## Where this leaves v7

- **λ = 2 makes inference practical:** near-converged in under an hour instead of a night, and structure becomes
  weakly identifiable (AUC about 0.6). λ = 1 is neither converged nor structurally identified.
- **Structure stays only weakly identified** under every setting tried, while behaviour is identified well. Different
  programs that fit equally well remain the main obstacle for a structural comparison with human similarity.
- **Latent phase helps motion and fit, not identifiability.**

## Runs 8-9: λ = 3, without and with latent phase

As Runs 6-7 with λ = 3. About 10 minutes each.

| Manner | Split R-hat (no phase / phase) | Distinct programs | Mass of the top program | Median log-lik of samples |
| --- | --- | --- | --- | --- |
| bowing | 1.27 / 1.82 | 132 / 337 | 0.84 / 0.76 | 632 / 630 |
| crisscrossing | 1.31 / 1.29 | 927 / 737 | 0.27 / 0.33 | 793 / 772 |
| jumping | 4.46 / 1.57 | 106 / 76 | 0.84 / 0.84 | 718 / 724 |
| swinging | 1.00 / 1.00 | 3 / 3 | 1.00 / 1.00 | 574 / 580 |

Chain-best programs average 5.4 / 4.8 nodes; swinging's posterior is the bare window `out`, and no samples move.
Structure AUC (best programs, tree_unordered / flat / full): 0.69 / 0.63 / 0.68 without phase, 0.61 / 0.62 / 0.61
with phase. Behaviour AUC (bowing / crisscrossing / jumping / swinging): 0.60 / 0.96 / 0.31 / 0.21 without phase,
0.67 / 0.83 / 0.46 / 0.38 with phase; 7-8 of 16 programs fit their own manner best.

**Reading.** λ = 3 is too strong: programs collapse to a few nodes (swinging to an empty window), the data no longer
shape them, and own-manner fit breaks down (swinging's programs fit other manners better than its own). The slightly
higher structure AUC only reflects near-empty programs resembling each other. Swinging's R-hat of 1.00 is trivial:
every chain sits on `out`.

## The λ sweep

| λ | Split R-hat | Chain-best size | Structure AUC (best programs) | Behaviour AUC | Motion |
| --- | --- | --- | --- | --- | --- |
| 1 (Runs 4-5) | 1.31-2.46 | 21 nodes | 0.51-0.54 | 0.77-0.99 | 0-12% |
| **2** (Runs 6-7) | 1.09-1.42 | 17 nodes | 0.55-0.64 | 0.52-1.00 | up to 21% with phase |
| 3 (Runs 8-9) | 1.00-4.46 | 5 nodes | 0.61-0.69 | 0.21-0.96 | 0% |

λ = 2 is the useful setting: near-converged within an hour, with fits that still distinguish most manners. Structure
is only weakly identified at any λ before the programs stop describing the data.

## Finer λ sweep: 1.5 and 2.5 (with latent phase)

As Run 7 (λ = 2 with phase) with λ = 1.5 and 2.5. The decision rule was fixed before the runs: (1) split R-hat
≤ 1.2 for all four manners; (2) own-manner behaviour AUC ≥ 0.75 for every manner; (3) among the λ passing both,
the highest structure AUC; (4) if none passes, keep λ = 2. Model-side measures only.

| λ | Split R-hat (bowing / crisscrossing / jumping / swinging) | Behaviour AUC (same order) | Structure AUC, best programs (tree_unordered / flat / full) | Structure AUC, posteriors | Chain-best size | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5 | 1.11 / 1.18 / 1.27 / 1.22 | 0.98 / 1.00 / 0.75 / 0.81 | 0.60 / 0.47 / 0.74 | 0.65 / 0.67 / 0.67 | 19.0 | about 70 min |
| 2 | 1.42 / 1.11 / 1.29 / 1.18 | 0.77 / 0.96 / 0.62 / 1.00 | 0.60 / 0.64 / 0.55 | 0.54 / 0.73 / 0.49 | 16.7 | about 25 min |
| 2.5 | 1.29 / 1.17 / 1.81 / 2.12 | 0.77 / 0.94 / 0.46 / 0.38 | 0.58 / 0.54 / 0.58 | 0.60 / 0.50 / 0.56 | 9.9 | about 20 min |

Median log-likelihood of samples at λ = 1.5: 864 / 1026 / 924 / 794 (λ = 2: 787 / 932 / 868 / 797).

**Reading.**
- **By the rule, none passes** (no λ has R-hat ≤ 1.2 in all four manners), so the rule keeps λ = 2.
- **But λ = 1.5 is at least as good as λ = 2 on every criterion:** its worst R-hat is lower (1.27 against 1.42), it is
  the only λ whose behaviour AUC is ≥ 0.75 for every manner, its fits are tighter, and its structure AUC is similar
  or higher. Its cost is time: about three times slower per chain.
- **λ = 2.5 is already too strong:** jumping and swinging stop converging and their programs no longer fit their own
  manner best.

## All 26 manners: λ = 1.5, latent phase, crossover

`results/grace_v7_parts/all26_l15_phase` (temperature 1/16, 30,000 iterations, burn-in 10,000, adaptive ladder,
4 chains × 12 replicas). Interrupted once when the computer shut down: the 5 manners finished by then were kept
and the other 21 rerun with the same seeds (the 4 diagnostic manners reproduce the λ = 1.5 sweep run exactly).

**Convergence.** Median split R-hat 1.19; 14 of 26 manners ≤ 1.2, 5 ≤ 1.1. Worst: mermaiding 1.57, scurrying 1.55,
hopscotching 1.42, hopping 1.30.

**Chain agreement (model-side).** Behaviour: median own-manner AUC about 0.85 (20 of 26 ≥ 0.74; weakest trotting 0.48,
swinging 0.56, jumping 0.58, creeping 0.61, twisting 0.62). Structure: same- vs different-manner AUC 0.57-0.64 on
chain posteriors, 0.49-0.56 on chain-best programs.

**Human similarity** (`python -m grace_lot.analysis.run --results results/grace_v7_parts/all26_l15_phase`; run once,
test actors sealed, nothing tuned on the ratings). Spearman ρ over manner pairs, Mantel permutation p, ΔR² over the
kinematic baselines with its permutation p, and leave-one-manner-out cross-validated ρ.

| Model | ρ | 95% CI | Mantel p | ΔR² over kinematic | p(ΔR²) | CV ρ kinematic + model |
| --- | --- | --- | --- | --- | --- | --- |
| noise ceiling (split-half) | 0.895 | ± 0.018 | | | | |
| kin_manner_prototype | 0.616 | [0.559, 0.618] | 0.0001 | | | |
| kinematic baselines only (CV) | | | | | | 0.590 |
| lot_full | **0.302** | [0.253, 0.311] | 0.0020 | 0.026 | 0.040 | 0.602 |
| lot_tree_ordered | 0.285 | [0.238, 0.295] | 0.0025 | 0.027 | 0.039 | 0.602 |
| lot_tree_unordered | 0.273 | [0.227, 0.283] | 0.0034 | 0.020 | 0.059 | 0.593 |
| lot_flat | 0.156 | [0.114, 0.172] | 0.0489 | 0.000 | 0.83 | 0.577 |
| map_full | 0.011 | [-0.021, 0.035] | 0.42 | 0.000 | 0.93 | 0.584 |
| map_tree_unordered | 0.194 | [0.155, 0.212] | 0.0090 | 0.001 | 0.54 | 0.585 |

Ablations of lot_full range from 0.244 (no hierarchy) to 0.350 (no dashed edges); see `analysis/summary.md`.
For comparison, v2 (the last full analysis): lot_full 0.137, tree measures -0.03 to -0.06, lot_flat 0.219.

**Reading.**
- **The structural LoT measures now track human similarity:** lot_full ρ = 0.30 (Mantel p = 0.002), up from 0.14 in
  v2, and the tree measures go from about zero to 0.27-0.29.
- **They add a little beyond kinematics:** ΔR² about 0.02-0.03 (p 0.04-0.06), and cross-validated ρ rises from 0.590
  to 0.602. Small and borderline: one of several measures tested, without correction for multiple comparisons.
- **Posterior averages carry the signal, single programs do not:** MAP-based measures are near zero (0.01-0.19).
  This matches the chain-agreement finding: any single program's structure is close to arbitrary, but averaged over
  the posterior, some structure shared within manners survives.
- **Kinematics remain far better** (0.62), and both sit well below the human ceiling (0.90).
- **Caveats:** 12 of 26 manners are not converged to R-hat ≤ 1.2; the ablation that drops dashed edges (0.35) is a
  post-hoc observation, not a test; the per-manner fit figures failed (the figure code predates multi-output
  programs), the summary and other figures are complete.

## Search failure or real ambiguity? Gaps between the chains' best programs

The chains disagree on structure (see above). That could mean real ambiguity (several structures score about the
same) or a search failure (some chains never reach the best mode). Test, model side only: per manner, each chain's
best log posterior and its gap to the best chain's. A chain Δ nats below another holds a mode with e^-Δ the
posterior mass, so a mixing sampler would have left it. Script: `scratchpad/chain_gaps.py` (not in the repo);
stored scores re-checked from scratch for swinging and skating (exact match).

| Gap to the best chain (78 non-best chains, 26 manners) | Chains |
| --- | --- |
| ≤ 3 nats (≤ 20x odds: a tie) | 3 |
| 3-10 nats | 5 |
| > 10 nats | 70 |
| median | 42.6 nats |

- **Mostly search failure.** Only 3 manners (hopscotching, skating, turning) have two near-tied chains, and their
  tied programs are fairly similar (tree_unordered 0.61-0.75). Elsewhere, three of four chains sit tens to hundreds
  of nats below the best one.
- **Split R-hat missed it:** median 1.19 while the bests are ~40 nats apart. Agreement of the chains' best log
  posteriors is the better check for these runs.
- **So structural identifiability is not yet measurable.** The weak structure agreement (AUC 0.5-0.6) mostly reflects
  which mode each chain got stuck in. Fix search first; only then ask whether the posterior is sharp, and whether
  a reuse prior or skeleton information is needed.

## Search fix, first attempt: population selection, a fixed ladder, longer runs

Five manners (bowing, jumping, swinging, shuffling, scurrying), λ = 1.5, latent phase, crossover as in the 26-manner
run. Changed: each manner gets 4 independent **searches**, each 4 ladders side by side; every 1,000 iterations of
the burn-in the ladder with the lowest τ = 1 log posterior continues from a copy of the best one
(`inference.population`, `select_every`). 60,000 iterations (30,000 burn-in) instead of 30,000 (10,000). A fixed
geometric ladder of 8 replicas from 1 to 0.15 instead of the adaptive 12. About 5x the compute per manner.
`results/grace_v7_parts/search_pop`. (The first two starts ran out of memory: the likelihood memos, keyed by raw
array bytes, reached GBs per worker. Now keyed by a 16-byte hash and capped at 10,000 entries; results unchanged.)

Yardstick, fixed beforehand: all 4 searches' best log posteriors within 3 nats.

| Manner | Best (old run) | Best (this run) | Gaps of the other 3 searches | Similarity of search bests |
| --- | --- | --- | --- | --- |
| bowing | 911.8 | 966.1 | 28.7, 47.6, 77.2 | 0.62 |
| jumping | 1033.2 | 1039.1 | 36.4, 85.1, 162.9 | 0.61 |
| swinging | 925.1 | 932.7 | 7.7, 26.4, 64.9 | 0.65 |
| shuffling | 1628.5 | 1621.1 | 34.2, 156.5, 177.4 | 0.61 |
| scurrying | 1951.2 | 2159.6 | 68.5, 210.6, 347.2 | 0.64 |

- **Fails the yardstick: 0 of 5** (old run: 0 of 5). The median gap of non-best searches halves (127 to 69 nats),
  and the best rises for 4 of 5 manners (scurrying by 208 nats), but searches still end on different peaks, with
  structurally different programs (similarity 0.54-0.79 for the top two, about the level of different manners).
- **The fixed ladder was a mistake.** Swap rates between τ = 1 and the next replica (0.76) are 1-2%, and 2-16% for
  the next pair: the cold replica was nearly cut off from the rest, so tempering did little. The adaptive ladder had
  packed replicas between 1 and 0.985 for a reason: log-likelihood differences between programs are hundreds of
  nats, so even Δτ = 0.02 is a large step. Its hot end was redundant (swaps 0.97-1.0 here too).
- **What the swap rates say about the posterior.** It is extremely sharp: changing τ by a few percent changes which
  programs are plausible. That makes search hard and is itself a modelling question: the Gaussian likelihood
  treats every actor, frame, part and feature as independent evidence, though they are strongly correlated.

## Search fix, second attempt: population selection with the adaptive ladder

As the first attempt, but with the adaptive ladder (12 replicas, τ = 1 to 0.05, spacing adapted during the burn-in;
selection copies a ladder's spacing with its replicas). `results/grace_v7_parts/search_pop_adaptive`, 5.5-7.7 h
per search.

| Manner | Best (old run / fixed ladder / adaptive) | Gaps of the other 3 searches | Split R-hat |
| --- | --- | --- | --- |
| bowing | 911.8 / **966.1** / 952.4 | 7.3, 20.0, 21.5 | 1.35 |
| jumping | 1033.2 / 1039.1 / **1105.1** | 29.5, 48.6, 69.3 | 1.19 |
| swinging | 925.1 / **932.7** / 929.8 | 4.6, 5.4, 6.6 | 1.05 |
| shuffling | 1628.5 / 1621.1 / **1652.0** | 15.0, 53.3, 56.9 | 1.19 |
| scurrying | 1951.2 / **2159.6** / 2032.1 | 14.2, 48.7, 90.2 | 2.00 |

- **Still 0 of 5 at 3 nats, but much closer:** the median gap of non-best searches falls from 127 (old run) and 69
  (fixed ladder) to 21.5 nats. Swinging's four searches are within 6.6 nats.
- **The ladder now mostly works:** swap rates 0.15-0.75 along the ladder for bowing, jumping, scurrying and
  shuffling. Two flaws remain: the fixed hot end leaves one big last step (scurrying's last pair never swaps), and
  swinging's ladder collapsed (9 replicas between 0.991 and 1, then 0.83 and 0.12, swap 0.08 between them).
- **Search still varies between runs:** the fixed-ladder run found programs 14 nats (bowing) and 128 nats
  (scurrying) above anything the four adaptive searches reached. Agreement within one run is not yet trustworthy.
- **Swinging shows a plateau, not one peak.** Across both runs, 6 programs lie within 10 nats of the best; they
  render 5 different behaviours (scan signatures), and the 2 that render identically differ in structure. Here
  the likelihood does not separate several distinct behaviours, and the language adds redundant parses on top.
  Whether that plateau is real ambiguity or a too-coarse likelihood is the next question.
