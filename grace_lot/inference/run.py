"""Infer posterior programs for each GRACE manner.

    python -m grace_lot.inference.run --split exemplar       # the Study-3 exemplar of each manner (first prototype)
    python -m grace_lot.inference.run --split train          # the 15 training actors of each manner
    python -m grace_lot.inference.run --split trainval       # training + validation actors (final fit)
    python -m grace_lot.inference.run --set inference.beta=4000 --out results/x

Each (manner, chain) pair is one parallel job. Chains use parallel
tempering when ``inference.replicas > 1``. With ``inference.population`` > 1
each job is a search of that many ladders with selection (``run_population``),
and every ladder is reported as a chain, tagged with its search. For each manner this writes
``<out>/posteriors/<manner>.json``:
* the pooled posterior;
* the MAP program per chain;
* acceptance per move type, swap rates, and R-hat of the log-posterior.

It also writes ``<out>/traces/<manner>.npz`` with the log-posterior,
log-likelihood and log-prior of every chain.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import time
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos, study3_exemplars, substituted
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.inference.grammar import Grammar
from grace_lot.inference.likelihood import TrajectoryLikelihood
from grace_lot.inference.mh import (calibrated_ladder, ladder_for, r_hat, run_population, run_tempered,
                                    temperature_ladder)
from grace_lot.inference.prior import SizePrior
from grace_lot.inference.proposals import ProposalMixture
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.distance import DISTANCES
from grace_lot.lot_adapter.program import parse_program, resolved_lot


def apply_overrides(cfg: dict, items: List[str]) -> dict:
    """``section.key=value`` overrides; values are Python literals."""
    cfg = copy.deepcopy(cfg)
    for item in items or []:
        key, value = item.split("=", 1)
        section, name = key.split(".")
        cfg[section][name] = ast.literal_eval(value)
    return cfg


def representation(cfg: dict) -> str:
    """'occupancy' (one field per frame) or 'parts' (skeleton part channels matched to anonymous LoT outputs)."""
    return cfg.get("interface", {}).get("representation", "occupancy")


def observations_for(cfg: dict, videos) -> list:
    if representation(cfg) == "parts_scan" and cfg["interface"].get("observation_cells", False):
        from grace_lot.grace.scanner import scan_part_cells     # parts quantised to cells first, like LoT outputs
        itf = cfg["interface"]
        return [scan_part_cells(load_trajectory(cfg, v), cfg["preprocess"], itf["scan_softness"], itf["lot_salience"],
                                itf["edge_samples"], itf.get("features", "max6")) for v in videos]
    if representation(cfg) == "parts_scan":
        from grace_lot.grace.scanner import scan_parts
        itf = cfg["interface"]
        return [scan_parts(load_trajectory(cfg, v), cfg["preprocess"], itf["scan_softness"], itf["edge_samples"])
                for v in videos]
    if representation(cfg) == "parts":
        from grace_lot.grace.parts import part_fields
        return [part_fields(load_trajectory(cfg, v), cfg["preprocess"]) for v in videos]
    return [load_trajectory(cfg, v).fields for v in videos]


def build(cfg: dict, observations, noise: dict = None):
    """Grammar, renderer, prior, likelihood and proposals. ``noise`` fixes the Gaussian likelihood's observation
    noise (``s2``, ``s2d``) instead of estimating it from ``observations`` (scoring held-out videos)."""
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    inf, pre = cfg["inference"], cfg["preprocess"]
    grammar = Grammar.from_config(cfg["lot"])
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    prior = SizePrior.from_config(grammar, inf)
    if representation(cfg) == "parts_scan" and inf.get("likelihood") == "gaussian":
        from grace_lot.grace.scanner import feature_background
        from grace_lot.inference.likelihood import GaussianScanLikelihood
        itf = cfg["interface"]
        if not itf.get("observation_cells", False):
            raise ValueError("likelihood 'gaussian' needs interface.observation_cells")
        lik = GaussianScanLikelihood(observations, pre["frames"], pre["field_size"], itf["scan_softness"],
                                     itf["lot_salience"], feature_background(cfg), inf.get("temperature"),
                                     inf.get("temporal_weight", 0.0), inf.get("cell_draws", 256),
                                     inf.get("noise_floor", 0.1), noise, features=itf.get("features", "max6"),
                                     phases=inf.get("phases", 1))
    elif representation(cfg) == "parts_scan" and inf.get("likelihood") == "probabilistic":
        from grace_lot.inference.likelihood import ProbabilisticScanLikelihood
        itf = cfg["interface"]
        background = inf.get("background")
        if background == "occupancy":
            from grace_lot.grace.parts import occupancy_background
            background = occupancy_background(cfg)
        lik = ProbabilisticScanLikelihood(observations, pre["frames"], pre["field_size"], inf["beta"],
                                          itf["scan_softness"], itf["lot_salience"], inf.get("temporal_weight", 0.0),
                                          inf.get("cell_draws", 8), background=background)
    elif representation(cfg) == "parts_scan":
        from grace_lot.inference.likelihood import MatchedScanLikelihood
        itf = cfg["interface"]
        lik = MatchedScanLikelihood(observations, renderer, inf["beta"], itf["scan_softness"], itf["lot_salience"],
                                    inf.get("temporal_weight", 0.0))
    elif representation(cfg) == "parts":
        from grace_lot.inference.likelihood import MatchedPartsLikelihood
        lik = MatchedPartsLikelihood(observations, renderer, inf["beta"], inf.get("temporal_weight", 0.0))
    else:
        lik = TrajectoryLikelihood(observations, renderer, DISTANCES[inf["distance"]], inf["beta"],
                                   inf.get("temporal_weight", 0.0))
    return grammar, renderer, prior, lik, ProposalMixture(grammar, inf["move_weights"])


def chain_ladder(inf: dict, prior, lik, proposal, n_observations: int, key: str) -> tuple:
    """The tempering ladder a chain starts with, and how long it adapts (``inference.ladder``)."""
    if inf.get("ladder") == "adaptive":                     # spaced by swap rates during burn-in, from a geometric start
        return temperature_ladder(inf.get("replicas", 1), inf.get("tau_min", 0.05)), inf["burn_in"]
    if inf.get("ladder") == "geometric":                    # fixed: geometric from 1 down to tau_min
        return temperature_ladder(inf.get("replicas", 1), inf["tau_min"]), 0
    if inf.get("ladder") == "calibrated":                   # scaled to this likelihood, the same for a manner's chains
        return calibrated_ladder(inf, prior, lik, proposal.grammar, random.Random(zlib.crc32(key.encode()))), 0
    return ladder_for(inf, n_observations), 0


def tempered(inf: dict, prior, lik, proposal, rng: random.Random, taus, adapt_until: int, report=None):
    """One tempered chain with the configured options: ``crossover`` (probability per sweep of a subtree exchange
    between neighbouring replicas) and ``start`` ("parts": data-driven starting programs, ``start_candidates`` trees
    per pool; otherwise draws from the generator). With ``population`` > 1, a search of that many chains with
    selection every ``select_every`` iterations of the burn-in: a list of chains."""
    init = None
    if inf.get("start") == "parts":
        from grace_lot.inference.starts import part_fit_start
        init = lambda r: part_fit_start(lik, proposal, prior, r, inf.get("start_candidates", 100), inf.get("start_steps", 200))
    args = (prior, lik, proposal, inf["iterations"], inf["burn_in"], inf["thin"], rng, taus, inf.get("swap_every", 1),
            report, inf.get("progress_every", 0), adapt_until, inf.get("crossover", 0.0), init)
    if inf.get("population", 1) > 1:
        return run_population(inf["population"], *args, select_every=inf.get("select_every", 1000))
    return run_tempered(*args)


def save_checkpoint(path: Path, chain) -> None:
    """A chain's traces so far (written atomically, so ``status`` never reads half a file)."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{key: np.array(getattr(chain, key)) for key in ("log_post", "log_lik", "log_prior")})
    tmp.replace(path)


def run_one_chain(job):
    cfg, manner, videos, k, out = job
    inf = cfg["inference"]
    observations = observations_for(cfg, videos)
    _, _, prior, lik, proposal = build(cfg, observations)
    taus, adapt_until = chain_ladder(inf, prior, lik, proposal, len(observations), manner)
    seed = inf["seed"] * 1000 + k + 17 * zlib.crc32(manner.encode())
    t0 = time.time()
    checkpoint = out / "checkpoints" / f"{manner}_{k}.npz"

    def report(it, chain):                                  # every progress_every iterations
        chain = max(chain, key=lambda c: c.log_post[-1]) if isinstance(chain, list) else chain   # a search: its lead
        print(f"  {manner} chain {k}: {it}/{inf['iterations']} iterations, "
              f"{(time.time() - t0) / 60:.0f} min, log-post {chain.log_post[-1]:.1f}", flush=True)
        save_checkpoint(checkpoint, chain)
    cs = tempered(inf, prior, lik, proposal, random.Random(seed), taus, adapt_until, report)
    return manner, k, [{
        "search": k, "samples": [str(p) for p in c.samples], "best": str(c.best), "best_log_post": c.best_log_post,
        "acceptance": c.acceptance_rate, "proposed": dict(c.moves), "accepted": dict(c.accepted),
        "swaps": c.swaps, "taus": c.taus, "crossovers": c.crossovers, "replaced": c.replaced,
        "log_post": c.log_post, "log_lik": c.log_lik, "log_prior": c.log_prior,
        "seconds": time.time() - t0} for c in (cs if isinstance(cs, list) else [cs])]


def finish_manner(cfg, manner, videos, chains: List[dict], out: Path, flags: Dict[str, str]) -> str:
    inf = cfg["inference"]
    counts = Counter(s for c in chains for s in c["samples"])
    total = sum(counts.values())
    best = max(chains, key=lambda c: c["best_log_post"])
    observations = observations_for(cfg, videos)
    _, renderer, prior, lik, _ = build(cfg, observations)
    best_prog = parse_program(best["best"])
    distance = DISTANCES[inf["distance"]]
    if representation(cfg) in ("parts", "parts_scan"):
        map_d = float(np.mean([d for _, _, d in lik.assignments(best_prog)]))
    else:
        map_d = float(np.mean([distance(o, renderer(best_prog)) for o in observations]))
    result = {
        "manner": manner, "videos": [v.path.name for v in videos],
        "substituted_for": flags.get(manner),
        "posterior": [[s, n / total] for s, n in counts.most_common()],
        "map": best["best"], "map_log_post": best["best_log_post"],
        "map_log_lik": lik(best_prog), "map_log_prior": prior(best_prog),
        "map_distance": map_d,
        "chains": [{**{k: c[k] for k in ("search", "best", "best_log_post", "acceptance", "proposed", "accepted",
                                         "swaps", "taus", "crossovers", "replaced", "seconds")},
                    "posterior": [[s, n / len(c["samples"])] for s, n in Counter(c["samples"]).most_common(500)]}
                   for c in chains],                          # each chain's own samples: its 500 most frequent programs
        "r_hat_log_post": r_hat([c["log_post"][inf["burn_in"]:] for c in chains]),
        "unique_programs": len(counts),
    }
    (out / "posteriors").mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "posteriors" / f"{manner}.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out / "traces" / f"{manner}.npz",
                        **{k: np.array([c[k] for c in chains]) for k in ("log_post", "log_lik", "log_prior")})
    return (f"{manner:14s} MAP D={result['map_distance']:.4f} R-hat={result['r_hat_log_post']:.2f} "
            f"unique={len(counts)}  {result['map']}")


def videos_for(cfg: dict, split: str):
    d = cfg["data"]
    if split == "exemplar":
        ex = study3_exemplars(d["similarity_csv"], d["raw_dir"], d["substitutes"])
        return {m: [v] for m, v in ex.items()}, substituted(ex, d["similarity_csv"])
    return split_videos(d["raw_dir"], cfg["split"], split), {}


def run(cfg: dict, split: str, out: Path, manners=None, workers: int = 16, quiet: bool = False) -> None:
    by_manner, flags = videos_for(cfg, split)
    manners = manners or sorted(by_manner)
    jobs = [(cfg, m, by_manner[m], k, out) for m in manners for k in range(cfg["inference"]["chains"])]
    done: Dict[str, List[dict]] = {m: [] for m in manners}            # chains (every ladder of a search)
    searches = Counter()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"fit_split": split, **{k: cfg[k] for k in ("preprocess", "lot", "inference", "split")},
         "lot": resolved_lot(cfg["lot"]),
         "interface": cfg.get("interface", {"representation": "occupancy"})}, indent=1))
    with Pool(min(workers, len(jobs))) as pool:
        for manner, k, chains in pool.imap_unordered(run_one_chain, jobs):
            done[manner] += chains
            searches[manner] += 1
            if not quiet:
                print(f"{manner}: chain {k} done in {chains[0]['seconds'] / 60:.0f} min "
                      f"({searches[manner]}/{cfg['inference']['chains']})", flush=True)
            if searches[manner] == cfg["inference"]["chains"]:
                line = finish_manner(cfg, manner, by_manner[manner], done.pop(manner), out, flags)
                if not quiet:
                    print(line, flush=True)


def status(out: Path) -> str:
    """Convergence so far, from a run's checkpoints (or its finished traces): per manner, the iterations each chain
    has done and the split R-hat of the log-posterior after burn-in, over the length all its chains have reached."""
    inf = json.loads((out / "config.json").read_text())["inference"]
    burn, lines = inf.get("burn_in", 0), []
    traces: Dict[str, Dict[int, np.ndarray]] = {}
    for f in sorted((out / "checkpoints").glob("*.npz")):
        manner, k = f.stem.rsplit("_", 1)
        traces.setdefault(manner, {})[int(k)] = np.load(f)["log_post"]
    for f in sorted((out / "traces").glob("*.npz")) if (out / "traces").exists() else []:
        traces[f.stem] = dict(enumerate(np.load(f)["log_post"]))          # finished: the full traces
    for manner, chains in sorted(traces.items()):
        done = [len(chains[k]) for k in sorted(chains)]
        n = min(done) - burn
        rh = r_hat([chains[k][burn:burn + n] for k in sorted(chains)]) if n >= 100 and len(chains) > 1 else float("nan")
        verdict = "converged" if rh < 1.1 else ("not yet" if rh == rh else "too early")
        lines.append(f"{manner:14s} iterations {done} of {inf['iterations']}  split R-hat {rh:.2f} over {max(n, 0)} "
                     f"post-burn-in iterations  ({verdict})")
    return "\n".join(lines) or "no checkpoints yet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--split", default="exemplar", choices=["exemplar", "train", "trainval"])
    ap.add_argument("--set", nargs="*", default=[], help="overrides such as inference.beta=4000")
    ap.add_argument("--manners", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--status", help="print convergence so far for this run directory, and exit")
    args = ap.parse_args()
    if args.status:
        return print(status(Path(args.status)))
    cfg = apply_overrides(load_config(args.config), args.set)
    out = Path(args.out) if args.out else cfg["data"]["results_dir"]
    run(cfg, args.split, out, args.manners, args.workers)


if __name__ == "__main__":
    main()
