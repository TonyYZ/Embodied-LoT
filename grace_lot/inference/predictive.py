"""Held-out log predictive density on the validation actors (Gaussian scan likelihood).

    python -m grace_lot.inference.predictive results/x results/y ... [--config ...] [--draws 1000]

Each run directory holds posteriors fitted to the training actors. For each of its manners m, ``draws`` programs
are drawn from m's pooled posterior (by weight, with replacement), and every validation video of every manner in
the run is scored under m's likelihood, with the observation noise of m's training actors (as in the fit):

    log p(v | m) = log mean_{r ~ posterior of m} p_w(v | r)

Reported per run: the summed log predictive density of each validation video under its own manner (the criterion
for choosing lambda at a fixed likelihood weight w; higher is better) and recognition among the run's manners
(argmax over m). Reads neither human similarity data nor the sealed test actors. Writes
``<run>/predictive.json``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot


def log_predictive(scores: np.ndarray) -> np.ndarray:
    """(draws, videos) log densities -> (videos,) log of their mean over draws."""
    return logsumexp(scores, axis=0) - np.log(len(scores))


def evaluate(cfg: dict, run_dir: Path, draws: int = 1000, seed: int = 0) -> dict:
    from grace_lot.inference.run import build, observations_for
    d = cfg["data"]
    sealed = set(cfg["split"]["test_actors"])
    manners = sorted(f.stem for f in (run_dir / "posteriors").glob("*.json"))
    train, val = (split_videos(d["raw_dir"], cfg["split"], s) for s in ("train", "val"))
    held = [v for m in manners for v in val[m]]
    if any(v.actor in sealed for v in held):
        raise PermissionError("validation set contains sealed test actors")
    truth = np.array([manners.index(v.manner) for v in held])
    held_obs = observations_for(cfg, held)
    rng = random.Random(seed)
    lpd = np.empty((len(manners), len(held)))                       # log p(video | manner)
    for i, m in enumerate(manners):
        lik_train = build(cfg, observations_for(cfg, train[m]))[3]
        lik = build(cfg, held_obs, noise=dict(s2=lik_train.s2, s2d=lik_train.s2d))[3]
        post = json.loads((run_dir / "posteriors" / f"{m}.json").read_text())["posterior"]
        progs, weights = zip(*post)
        sample = rng.choices(progs, weights, k=draws)
        cache = {s: lik.per_video_scores(parse_program(s)) for s in set(sample)}
        lpd[i] = log_predictive(np.array([cache[s] for s in sample]))
        print(f"  {run_dir.name} {m}: {len(cache)} distinct programs scored", flush=True)
    own = lpd[truth, np.arange(len(held))]
    return {"manners": manners, "videos": [v.stem for v in held], "lpd": lpd.tolist(),
            "own_lpd_total": float(own.sum()),
            "own_lpd_by_manner": {m: float(own[truth == i].sum()) for i, m in enumerate(manners)},
            "recognition_top1": float(np.mean(lpd.argmax(axis=0) == truth)), "draws": draws}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    for r in map(Path, args.runs):
        cfg = load_config(args.config)
        run_cfg = json.loads((r / "config.json").read_text())
        for k in ("preprocess", "interface", "split", "inference"):     # the run's own settings and language
            cfg[k] = run_cfg.get(k, cfg[k])
        cfg["lot"] = saved_lot(run_cfg["lot"])
        apply_semantics(cfg["lot"])
        res = evaluate(cfg, r, args.draws)
        (r / "predictive.json").write_text(json.dumps(res, indent=1))
        print(f"{r.name}: held-out log predictive density {res['own_lpd_total']:.1f}  "
              f"recognition {res['recognition_top1']:.2f}  by manner "
              + ", ".join(f"{m} {v:.1f}" for m, v in res["own_lpd_by_manner"].items()), flush=True)


if __name__ == "__main__":
    main()
