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

**Next steps, not yet run:**
1. **Fix the temperature ladder.** For the Gaussian likelihood, scale the hottest replica to the
   likelihood's actual size instead of to `beta`.
2. **Rerun temperature 1/16** with more iterations, and perhaps 1/4 between it and the full Gaussian, until
   R-hat falls below 1.1.
3. **Run all 26 manners** once one setting converges, then use validation recognition to choose between
   the remaining settings. Recognition is the one score comparable across likelihood weights.
4. **Add latent phase, and probably tempo,** to the v7 likelihood. The audit showed motion can only be
   rewarded once each video's phase is aligned.
