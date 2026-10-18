"""Infer posterior programs for each GRACE manner.

    python -m grace_lot.inference.run --split exemplar       # the Study-3 exemplar of each manner (first prototype)
    python -m grace_lot.inference.run --split train          # the 15 training actors of each manner
    python -m grace_lot.inference.run --split trainval       # training + validation actors (final fit)
    python -m grace_lot.inference.run --set inference.beta=4000 --out results/x

Each (manner, chain) pair is one parallel job. Chains use parallel
tempering when ``inference.replicas > 1``. For each manner this writes
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
from grace_lot.inference.mh import ladder_for, r_hat, run_tempered
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


def build(cfg: dict, observations):
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
                                     inf.get("noise_floor", 0.1), features=itf.get("features", "max6"))
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


def run_one_chain(job):
    cfg, manner, videos, k = job
    inf = cfg["inference"]
    observations = observations_for(cfg, videos)
    _, _, prior, lik, proposal = build(cfg, observations)
    taus = ladder_for(inf, len(observations))
    seed = inf["seed"] * 1000 + k + 17 * zlib.crc32(manner.encode())
    t0 = time.time()
    c = run_tempered(prior, lik, proposal, inf["iterations"], inf["burn_in"], inf["thin"],
                     random.Random(seed), taus, inf.get("swap_every", 1))
    return manner, k, {
        "samples": [str(p) for p in c.samples], "best": str(c.best), "best_log_post": c.best_log_post,
        "acceptance": c.acceptance_rate, "proposed": dict(c.moves), "accepted": dict(c.accepted),
        "swaps": c.swaps, "log_post": c.log_post, "log_lik": c.log_lik, "log_prior": c.log_prior,
        "seconds": time.time() - t0}


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
        "chains": [{k: c[k] for k in ("best", "best_log_post", "acceptance", "proposed", "accepted",
                                      "swaps", "seconds")} for c in chains],
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
    jobs = [(cfg, m, by_manner[m], k) for m in manners for k in range(cfg["inference"]["chains"])]
    done: Dict[str, List[dict]] = {m: [] for m in manners}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"fit_split": split, **{k: cfg[k] for k in ("preprocess", "lot", "inference", "split")},
         "lot": resolved_lot(cfg["lot"]),
         "interface": cfg.get("interface", {"representation": "occupancy"})}, indent=1))
    with Pool(min(workers, len(jobs))) as pool:
        for manner, k, chain in pool.imap_unordered(run_one_chain, jobs):
            done[manner].append(chain)
            if len(done[manner]) == cfg["inference"]["chains"]:
                line = finish_manner(cfg, manner, by_manner[manner], done.pop(manner), out, flags)
                if not quiet:
                    print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--split", default="exemplar", choices=["exemplar", "train", "trainval"])
    ap.add_argument("--set", nargs="*", default=[], help="overrides such as inference.beta=4000")
    ap.add_argument("--manners", nargs="*")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    out = Path(args.out) if args.out else cfg["data"]["results_dir"]
    run(cfg, args.split, out, args.manners, args.workers)


if __name__ == "__main__":
    main()
