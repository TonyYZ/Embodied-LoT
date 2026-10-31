"""A one-ladder benchmark of the search, on real data, without the full experiment.

    python tools/bench_search.py --manner bowing --iterations 200 --start results/grace_v7_parts/desc_fix_5

One ladder, set up as ``run._ladder_worker`` sets it up (seed, tempering ladder, crossover), with round 0's settings
unless ``--set`` says otherwise. ``--start`` warms it up from a run's MAP, so the programs are as big as late in a fit.
Prints seconds per sweep and a digest of the chain (every replica's program and log posterior after each sweep): two
versions of the code that give the same digest walked the same chain, draw for draw.

    --cprofile FILE   the whole run under cProfile (read it with ``python -m pstats FILE``)
    --timers          wall time in the pipeline's stages, by wrapping them here: nothing is added to the inner loop
    --corpus FILE     every program the likelihood was asked about, for ``check_rollouts``

For py-spy: ``py-spy record -o prof.svg -- python tools/bench_search.py ...``. Run it from the repo root with
PYTHONPATH=. (or pointing at another copy of the code, to compare)."""

from __future__ import annotations

import argparse
import cProfile
import functools
import hashlib
import json
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

ROUND0 = ['inference.likelihood="descriptors"', 'inference.open_cells="expected"', 'inference.descriptors="full"',
          "inference.warmup=16", "lot.subcell=1", "lot.max_nodes=32", "inference.lambda_size=1.0", "inference.phases=8",
          "inference.crossover=0.5", "inference.chains=2", "inference.population=2", "inference.replicas=12",
          'inference.ladder="adaptive"', "inference.tau_min=0.05", "inference.iterations=24000",
          "inference.burn_in=12000", "inference.select_every=1000", "inference.temperature=1.0",
          'lot.primitives=["top","center","upper","bottom","periphery","lower","universal","void","neutral"]']


def wrap_timers(totals: dict, counts: dict) -> None:
    """Time the stages by wrapping them. The outermost call of a stage is what counts, so recursion is not counted
    twice; stages nest, so the numbers do not add up to the total."""
    import embodied_lot as E
    from grace_lot.inference import descriptor_likelihood as DL
    from grace_lot.inference import proposals, prior as PR
    from grace_lot.lot_adapter import adapter
    from grace_lot.grace import descriptors as D
    import scipy.optimize

    def timed(owner, name, label=None):
        fn, label = getattr(owner, name), label or name
        depth = [0]
        @functools.wraps(fn)
        def inner(*a, **k):
            depth[0] += 1
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                depth[0] -= 1
                if not depth[0]:
                    totals[label] += time.perf_counter() - t
                    counts[label] += 1
        setattr(owner, name, inner)

    timed(proposals.ProposalMixture, "__call__", "proposal")
    timed(PR.SizePrior, "__call__", "prior")
    timed(DL.DescriptorLikelihood, "__call__", "likelihood")
    timed(DL.DescriptorLikelihood, "_channel", "channel")
    timed(DL.DescriptorLikelihood, "_matched", "matched")
    timed(D, "board_signals", "descriptor signals")
    timed(scipy.optimize, "linear_sum_assignment", "hungarian")
    for name in ("motion_combinations", "rollout", "rollout_whole", "_state_signature", "_window_rollout"):
        timed(adapter, name)
    timed(adapter, "compile_program")
    for name in ("step", "_render_all", "node_views", "_issue_orders", "_motion_field", "_move_substance",
                 "_move_bodies_gameboard", "reset_gates", "assign_roles"):
        timed(E.ConceptGraph, name)
    timed(E.ConceptGraph, "_paint")
    for name in ("_claim_values", "_claim_values_of", "_claimed_value", "_view_mask", "_view_field"):
        if hasattr(E, name):
            timed(E, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--manner", default="bowing")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--ladder", type=int, default=0)
    ap.add_argument("--start", default=None, help="a run folder: start from its MAP for this manner")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--cprofile", default=None)
    ap.add_argument("--timers", action="store_true")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--summary", default=None, help="write the timing and digest here as JSON")
    args = ap.parse_args()

    from grace_lot.config import load_config
    from grace_lot.grace.dataset import split_videos
    from grace_lot.inference.mh import _Ladder
    from grace_lot.inference.run import apply_overrides, build, chain_ladder, observations_for, start_init

    sets = ROUND0 + ([f'inference.start="run:{args.start}"'] if args.start else []) + args.set
    cfg = apply_overrides(load_config(args.config), sets)
    inf = cfg["inference"]
    observations = observations_for(cfg, split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")[args.manner])
    _, _, prior, lik, proposal = build(cfg, observations)
    totals, counts = defaultdict(float), defaultdict(int)
    if args.timers:
        wrap_timers(totals, counts)
    taus, adapt_until = chain_ladder(inf, prior, lik, proposal, len(observations), args.manner)
    rng = random.Random(inf["seed"] * 1000 + args.chain + 17 * zlib.crc32(args.manner.encode()))
    for _ in range(args.ladder):
        rng.getrandbits(64)
    corpus = {}
    score = lik
    if args.corpus:
        def score(p):
            corpus[str(p)] = None
            return lik(p)
    init = start_init(inf, args.manner, prior, lik, proposal)
    t0 = time.perf_counter()
    ladder = _Ladder(prior, score, proposal, random.Random(rng.getrandbits(64)), taus, inf.get("swap_every", 1),
                     adapt_until, inf.get("crossover", 0.0), init)
    setup = time.perf_counter() - t0
    digest = hashlib.sha256()

    def sweeps():
        for it in range(args.iterations):
            ladder.sweep(it, inf["burn_in"], inf["thin"])
            digest.update(repr([(str(s.program), s.lp, s.ll) for s in ladder.states]).encode())

    t0 = time.perf_counter()
    if args.cprofile:
        pr = cProfile.Profile()
        pr.runcall(sweeps)
        pr.dump_stats(args.cprofile)
    else:
        sweeps()
    wall = time.perf_counter() - t0
    from grace_lot.lot_adapter import adapter
    nodes = [sum(1 for _ in str(s.program).split()) for s in ladder.states]
    out = dict(manner=args.manner, iterations=args.iterations, setup=setup, wall=wall, per_1000_min=wall / args.iterations * 1000 / 60,
               digest=digest.hexdigest()[:16], best=ladder.log_post, tokens=nodes)
    print(json.dumps(out))
    if args.timers:
        for k in sorted(totals, key=totals.get, reverse=True):
            print(f"  {k:34s} {totals[k]:8.2f} s  {100 * totals[k] / wall:5.1f}%  {counts[k]:9d} calls  {1e3 * totals[k] / counts[k]:8.3f} ms")
    if args.corpus:
        Path(args.corpus).write_text(json.dumps(sorted(corpus)))
    if args.summary:
        Path(args.summary).write_text(json.dumps(dict(out, timers=dict(totals), calls=dict(counts))))


if __name__ == "__main__":
    main()
