"""Why are the fitted programs static? Does the likelihood reward motion at all? Training actors only.

    python -m grace_lot.analysis.motion_check --run results/grace_v7_parts/w32_l1.0_current_cap32

For each manner, a real training actor (the medoid) is used as if it were a program: its six part boards become six
outputs, drawn exactly (no unknown cells), and it is scored on the other training actors with the run's own
likelihood. The same actor frozen at its best single frame is scored the same way. If motion helped, the moving
actor would win. Also reported: the best fitted program (from ``gallery/gallery.json``) and an empty program on
the same actors, each score with and without the frame-to-frame change term (``temporal_weight`` 0). And how far
the parts actually move on the board (centroid range over the 16 frames, in cells, mean over actors).
Writes ``<run>/motion_check.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.parts import PART_NAMES
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.grace.scanner import field_features, part_boards
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot


def board_score(lik, boards: np.ndarray, itf: dict) -> float:
    """Summed log density of ``lik``'s videos under exact boards (T, K, H, W) used as a program's outputs."""
    F = field_features(boards, itf["scan_softness"], itf["lot_salience"], itf.get("features", "max6"))
    dF = np.diff(F, axis=0)
    m = dict(weight=1.0, mu=F, var=np.zeros_like(F), dmu=dF, dvar=np.zeros_like(dF))
    match = lik._matched_warped if getattr(lik, "warp_radius", 0) else lik._matched
    L = -lik.temperature * np.array([nll for _, _, nll in match(m)])
    return float((logsumexp(L.reshape(lik.N, lik.phases), axis=1) - np.log(lik.phases)).sum())


def centroid_range(boards: np.ndarray) -> np.ndarray:
    """(P,) the larger of the x and y range, over time, of each part's centroid, in cells."""
    m = boards.sum(axis=(-1, -2)) + 1e-9
    n = np.arange(boards.shape[-1])
    c = np.stack([(boards.sum(-2) * n).sum(-1) / m, (boards.sum(-1) * n).sum(-1) / m], axis=-1)
    return np.ptp(c, axis=0).max(axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--timing-sd", type=float, help="score with this inference.timing_sd instead of the run's")
    ap.add_argument("--out", help="default: <run>/motion_check.json")
    ap.add_argument("--manners", nargs="*", help="default: every manner in the gallery")
    args = ap.parse_args()
    from grace_lot.inference.run import build, observations_for
    run = Path(args.run)
    cfg = load_config(args.config)
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    cfg["lot"] = saved_lot(rc["lot"])
    apply_semantics(cfg["lot"])
    if args.timing_sd is not None:
        cfg["inference"]["timing_sd"] = args.timing_sd
    itf = cfg["interface"]
    still = copy.deepcopy(cfg)
    still["inference"]["temporal_weight"] = 0.0
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    assert not any(v.actor in set(cfg["split"]["test_actors"]) for vs in train.values() for v in vs)
    gallery = json.loads((run / "gallery" / "gallery.json").read_text())
    res = {}
    for m in args.manners or gallery:
        videos = train[m]
        obs = observations_for(cfg, videos)
        O = np.stack([o[0] if o.ndim == 4 else o for o in obs])                       # phase 0 of per-phase copies
        med = int(np.argmin(((O - O.mean(0)) ** 2).mean(axis=(1, 2, 3))))
        rest = [o for i, o in enumerate(obs) if i != med]
        boards = [part_boards(load_trajectory(cfg, v), cfg["preprocess"], itf["edge_samples"]) for v in videos]
        B = boards[med]
        best = parse_program(gallery[m]["top"][0]["program"])
        row = {"actor": videos[med].stem, "others": len(rest),
               "part_motion_cells": dict(zip(PART_NAMES, np.mean([centroid_range(b) for b in boards], axis=0).round(2).tolist()))}
        for name, c in (("with_changes", cfg), ("positions_only", still)):
            lik = build(c, rest)[3]
            row[name] = {"actor_moving": board_score(lik, B, itf),
                         "actor_frozen": max(board_score(lik, np.repeat(B[t:t + 1], len(B), 0), itf) for t in range(len(B))),
                         "best_program": float(lik(best)), "empty_program": float(lik(parse_program("out")))}
        res[m] = row
        w = row["with_changes"]
        print(f"{m}: actor moving {w['actor_moving']:.1f}, frozen {w['actor_frozen']:.1f}, best program "
              f"{w['best_program']:.1f}, empty {w['empty_program']:.1f}; positions only: motion gain "
              f"{row['positions_only']['actor_moving'] - row['positions_only']['actor_frozen']:.1f}", flush=True)
    (Path(args.out) if args.out else run / "motion_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
