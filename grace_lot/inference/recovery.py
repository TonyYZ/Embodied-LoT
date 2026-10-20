"""Synthetic recovery: can inference find programs that the LoT itself generated?

    python -m grace_lot.inference.recovery [--n 24] [--actors 15] [--noise 0.05] [--truths PROGRAM ...]
                                           [--from-run RUN_DIR --manners MANNER ...]

For each "true" program (the MAP programs of ``--manners`` in a fitted run, so the synthetic
videos resemble those GRACE videos; those given with ``--truths``; then n drawn from the prior):
  1. Render it (``lot_trajectory``) and make ``actors`` noisy copies (Gaussian
     noise per cell, clipped to [0, 1]). These stand in for one manner's videos.
     With ``--shift k``, each copy is also shifted in time by up to k frames
     (edge frames repeated), mimicking actors who start a movement at
     different times.
  2. Run exactly the configured inference (prior, likelihood, proposals,
     tempering, iterations).
  3. Score the result:
     search_ok        the best log-posterior found is >= the truth's (search did not fail)
     equiv_mass       posterior mass on programs that render identically to the truth
     exact_mass       posterior mass on the truth itself
     sim_to_truth     E_post[full similarity to the truth], compared with
     sim_prior        E_prior[full similarity to the truth] (the chance level)

Programs that render the same as the truth are indistinguishable by any
likelihood, so ``equiv_mass``, not ``exact_mass``, is the target. A low
``search_ok`` rate or ``equiv_mass`` is an implementation or mixing failure.
Nothing here touches human data.

With the v7 interface (``parts_scan`` and the Gaussian likelihood) each synthetic actor is a draw from that
likelihood's own generative model: a motion realization (by its prior weight), one board per output channel
(claimed cells figure, unclaimed cells filled from a real part board, fresh randomness per actor), scan features,
Gaussian noise at the real manners' level, and the channels in a random order as parts. Equivalence is then
rendering identically under every motion realization, up to channel order; the ladder is the run's
(``run.chain_ladder``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from grace_lot.config import load_config
from grace_lot.inference.grammar import Grammar
from grace_lot.inference.likelihood import TrajectoryLikelihood
from grace_lot.inference.mh import posterior, r_hat
from grace_lot.inference.prior import SizePrior
from grace_lot.inference.proposals import ProposalMixture, _addable
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.distance import DISTANCES, euclid
from grace_lot.lot_adapter.program import Program, size, with_edges
from grace_lot.lot_adapter.similarity import MEASURES

EQUIV_TOL = 1e-4


def sample_prior_program(g: Grammar, prior: SizePrior, rng: random.Random) -> Program:
    """An exact draw from the prior by rejection: propose (generator tree + uniform edges), accept with exp(prior - proposal - M)."""
    while True:
        t = g.sample(rng)
        k = rng.randrange(g.max_dashed + 1)
        options = _addable(Program(t), g)
        if k > len(options):
            continue
        prog = with_edges(t, rng.sample(options, k))
        lp = prior(prog)
        if lp == -math.inf:
            continue
        log_q = g.log_prob(t) - math.log(g.max_dashed + 1) - math.log(math.comb(len(options), k))
        if math.log(rng.random()) < lp - log_q:      # lp <= log_q when lambda >= 1 (description-length prior)
            return prog


def make_truths(g, prior, renderer, n, rng, min_size=3):
    """n prior draws with size >= min_size and a non-empty rendering. Half are required to change over time."""
    truths = []
    while len(truths) < n:
        p = sample_prior_program(g, prior, rng)
        y = renderer(p)
        if size(p.tree) < min_size or y.max() < 0.1:
            continue
        if p.tree.kind == "outputs" and len(p.tree.children) < 2:
            continue                     # multi-output truths: permutation must matter
        moving = bool(np.abs(np.diff(y, axis=0)).max() > 0.05)
        if len(truths) < n // 2 and not moving:
            continue
        truths.append(p)
    return truths


def synthetic_copies(y: np.ndarray, n: int, noise: float, shift: int, rng) -> list:
    """n noisy copies of a rendering (Gaussian noise per cell, clipped), each shifted by up to ``shift`` frames."""
    out = []
    for _ in range(n):
        s = int(rng.integers(-shift, shift + 1)) if shift else 0
        idx = np.clip(np.arange(len(y)) - s, 0, len(y) - 1)
        out.append(np.clip(y[idx] + rng.normal(0, noise, y.shape), 0, 1))
    return out


def scan_signature(p, frames: int, size: int) -> tuple:
    """A program's renderings under every motion realization, up to channel order: equal means indistinguishable."""
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    out = []
    for states, w in motion_combinations(p, size):
        v = rollout(p, frames, size, states)
        v = v if v.ndim == 4 else v[:, None]
        out.append((round(w, 9), tuple(sorted(v[:, k].tobytes() for k in range(v.shape[1])))))
    return tuple(sorted(out))


def scan_actor(truth, cfg: dict, background: dict, s2: np.ndarray, rng) -> np.ndarray:
    """One synthetic actor (T, K, F), drawn from the Gaussian scan likelihood's generative model."""
    from grace_lot.grace.scanner import field_features
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    pre, itf = cfg["preprocess"], cfg["interface"]
    combos = motion_combinations(truth, pre["field_size"])
    w = np.array([c[1] for c in combos])
    v = rollout(truth, pre["frames"], pre["field_size"], combos[rng.choice(len(combos), p=w / w.sum())][0])
    v = v if v.ndim == 4 else v[:, None]                                               # (T, K, H, W)
    fill = background["boards"][rng.integers(len(background["boards"]))].astype(float)   # (T, H, W)
    uniform = rng.random(v.shape[2:])
    boards = np.where(uniform[None, None] < np.abs(2 * v - 1), (v > 0.5).astype(float), fill[:, None])
    feats = field_features(boards, itf["scan_softness"], itf["lot_salience"], itf.get("features", "max6"))
    feats = feats + rng.normal(0.0, np.sqrt(s2), feats.shape)
    return feats[:, rng.permutation(feats.shape[1])]


def recover_scan(cfg: dict, truth, actors: int, s2: np.ndarray, seed: int) -> tuple:
    """Synthetic v7 observations of ``truth``, their likelihood, and an equivalence test."""
    from grace_lot.grace.scanner import feature_background
    from grace_lot.inference.likelihood import GaussianScanLikelihood
    pre, itf, inf = cfg["preprocess"], cfg["interface"], cfg["inference"]
    background = feature_background(cfg)
    rng = np.random.default_rng(seed)
    obs = [scan_actor(truth, cfg, background, s2, rng) for _ in range(actors)]
    lik = GaussianScanLikelihood(obs, pre["frames"], pre["field_size"], itf["scan_softness"], itf["lot_salience"],
                                 background, inf.get("temperature"), inf.get("temporal_weight", 0.0),
                                 inf.get("cell_draws", 256), inf.get("noise_floor", 0.1), features=itf.get("features", "max6"))
    target = scan_signature(truth, pre["frames"], pre["field_size"])
    return obs, lik, lambda p: scan_signature(p, pre["frames"], pre["field_size"]) == target


def recover_one(job) -> dict:
    cfg, truth_str, k, actors, noise, seed, shift, label = job[:8]
    s2 = np.asarray(job[8]) if len(job) > 8 else None
    from grace_lot.lot_adapter.program import parse_program
    truth = parse_program(truth_str)
    inf, pre = cfg["inference"], cfg["preprocess"]
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    g = Grammar.from_config(cfg["lot"])
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    rng = np.random.default_rng(seed)
    y = renderer(truth)
    itf = cfg.get("interface", {})
    parts_mode = itf.get("representation", "occupancy") in ("parts", "parts_scan")
    scan_mode = itf.get("representation") == "parts_scan" and inf.get("likelihood") == "gaussian"
    if scan_mode:
        obs, lik, same_render = recover_scan(cfg, truth, actors, s2, seed)
    elif parts_mode:
        # Anonymous outputs: every actor shows the true channels in its own random order.
        from grace_lot.inference.likelihood import MatchedPartsLikelihood
        yk = y if y.ndim == 4 else y[:, None]
        obs = []
        for _ in range(actors):
            perm = rng.permutation(yk.shape[1])
            s_ = int(rng.integers(-shift, shift + 1)) if shift else 0
            idx = np.clip(np.arange(len(yk)) - s_, 0, len(yk) - 1)
            obs.append(np.clip(yk[idx][:, perm] + rng.normal(0, noise, yk.shape), 0, 1))
        lik = MatchedPartsLikelihood(obs, renderer, inf["beta"], inf.get("temporal_weight", 0.0))
        truth_lik = MatchedPartsLikelihood([yk], renderer, inf["beta"])
        same_render = lambda p: truth_lik.assignments(p)[0][2] < EQUIV_TOL
    else:
        obs = synthetic_copies(y, actors, noise, shift, rng)
        lik = TrajectoryLikelihood(obs, renderer, DISTANCES[inf["distance"]], inf["beta"],
                                   inf.get("temporal_weight", 0.0))
        same_render = lambda p: euclid(renderer(p), y) < EQUIV_TOL
    prior = SizePrior.from_config(g, inf)
    prop = ProposalMixture(g, inf["move_weights"])
    from grace_lot.inference.run import chain_ladder, tempered
    taus, adapt_until = chain_ladder(inf, prior, lik, prop, len(obs), f"truth{k}")
    t0 = time.time()
    chains = [tempered(inf, prior, lik, prop, random.Random(seed * 100 + c), taus, adapt_until)
              for c in range(inf["chains"])]
    post = posterior(chains)
    truth_lp = prior(truth) + lik(truth)
    best_lp = max(c.best_log_post for c in chains)
    top = sorted(post.items(), key=lambda kv: -kv[1])[:500]      # equivalence is costly in v7: the heaviest programs
    equiv = sum(w for p, w in top if same_render(p))
    full = MEASURES["tree_unordered"] if parts_mode else MEASURES["full"]   # output order is arbitrary
    sim_post = sum(w * full(p, truth) for p, w in post.items())
    prng = random.Random(seed)
    sim_prior = float(np.mean([full(sample_prior_program(g, prior, prng), truth) for _ in range(200)]))
    best = max(chains, key=lambda c: c.best_log_post).best
    return {"k": k, "seed": seed, "label": label, "truth": truth_str, "truth_size": size(truth.tree), "truth_edges": len(truth.dashed),
            "truth_log_post": truth_lp, "best_log_post": best_lp, "search_ok": best_lp >= truth_lp - 1e-6,
            "equiv_mass": equiv, "exact_mass": post.get(truth, 0.0),
            "map": str(best), "map_equiv": bool(same_render(best)),
            "sim_to_truth": sim_post, "sim_prior": sim_prior,
            "r_hat": r_hat([c.log_post[inf["burn_in"]:] for c in chains]),
            "chain_best": [c.best_log_post for c in chains], "chain_best_equiv": [bool(same_render(c.best)) for c in chains],
            "swap_rates": [round(s, 2) for s in chains[0].swaps], "crossovers": [c.crossovers for c in chains],
            "chain_log_post": [c.log_post for c in chains], "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--actors", type=int, default=15)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--shift", type=int, default=0, help="max per-actor time shift in frames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides such as inference.iterations=40000")
    ap.add_argument("--truths", nargs="*", default=[], help="hand-picked true programs, recovered before the prior draws")
    ap.add_argument("--from-run", help="a fitted run whose manners' MAP programs serve as true programs")
    ap.add_argument("--manners", nargs="*", default=[], help="manners of --from-run to take MAP programs from")
    args = ap.parse_args()
    from grace_lot.inference.run import apply_overrides
    cfg = apply_overrides(load_config(args.config), args.set)
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    g = Grammar.from_config(cfg["lot"])
    prior = SizePrior.from_config(g, cfg["inference"])
    renderer = LotRenderer(cfg["preprocess"]["frames"], cfg["preprocess"]["field_size"])
    from grace_lot.lot_adapter.program import parse_program, resolved_lot
    labelled = [(f"manner:{m}", parse_program(json.loads((Path(args.from_run) / "posteriors" / f"{m}.json").read_text())["map"]))
                for m in args.manners]
    labelled += [("hand-picked", parse_program(t)) for t in args.truths]
    labelled += [("prior draw", t) for t in make_truths(g, prior, renderer, args.n, random.Random(12345))]
    s2 = None
    if cfg.get("interface", {}).get("representation") == "parts_scan" and cfg["inference"].get("likelihood") == "gaussian":
        from grace_lot.inference.run import build, observations_for, videos_for   # the real manners' noise level
        by_manner, _ = videos_for(cfg, "train")
        s2 = np.mean([build(cfg, observations_for(cfg, by_manner[m]))[3].s2
                      for m in (args.manners or ["bowing", "crisscrossing", "jumping", "swinging"])], axis=0).tolist()
        apply_semantics(cfg["lot"])
    jobs = [(cfg, str(t), k, args.actors, args.noise, 1000 + k, args.shift, label, s2)
            for k, (label, t) in enumerate(labelled)]
    results = []
    with Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(recover_one, jobs):
            results.append(r)
            print(f"[{r['k']:2d}] ok={r['search_ok']!s:5} equiv={r['equiv_mass']:.2f} exact={r['exact_mass']:.2f} "
                  f"sim={r['sim_to_truth']:.2f} (prior {r['sim_prior']:.2f}) R-hat={r['r_hat']:.2f} "
                  f"{r['seconds']:.0f}s  {r['truth']}", flush=True)
    results.sort(key=lambda r: r["k"])
    summary = {
        "n": len(results), "actors": args.actors, "noise": args.noise, "shift": args.shift,
        "lot": resolved_lot(cfg["lot"]), "preprocess": cfg["preprocess"], "inference": cfg["inference"],
        "data": {k: str(v) for k, v in cfg["data"].items()}, "split": cfg["split"], "from_run": args.from_run,
        "interface": cfg.get("interface", {"representation": "occupancy"}),
        "search_ok_rate": float(np.mean([r["search_ok"] for r in results])),
        "mean_equiv_mass": float(np.mean([r["equiv_mass"] for r in results])),
        "map_equiv_rate": float(np.mean([r["map_equiv"] for r in results])),
        "mean_exact_mass": float(np.mean([r["exact_mass"] for r in results])),
        "mean_sim_to_truth": float(np.mean([r["sim_to_truth"] for r in results])),
        "mean_sim_prior": float(np.mean([r["sim_prior"] for r in results])),
        "results": results,
    }
    out = Path(args.out) if args.out else cfg["data"]["results_dir"].parent / "grace_recovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
