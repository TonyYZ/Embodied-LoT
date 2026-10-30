"""Kinematic baselines for held-out recognition, on the same input and split as ``inference.predictive``.

    python -m grace_lot.analysis.kinematic_baselines --run results/grace_v7_parts/all26_current

No programs: each validation video is compared directly with the training actors' observations (the scanned
part features the LoT likelihood sees, (T, parts, features)).

  prototype  nearest manner mean of the training observations
  1nn        nearest single training video

Distance: mean squared difference plus ``temporal_weight`` x that of the frame-to-frame changes (as in the fit),
minimised over the same circular phase shifts as the likelihood (``inference.phases``). Two scalings: raw, and
each (part, feature) divided by its variance over all training videos. Recognition is scored like the LoT's
(top-1, top-5, mean rank), and the LoT is compared video by video (exact McNemar test). Reads neither human
data nor the sealed test actors. Writes ``<run>/kinematic_baselines.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos


def distance(a: np.ndarray, b: np.ndarray, w: float, phases: int) -> float:
    """min over shifts s < phases of D(roll(a, s), b); a, b (T, P, F). An ``a`` of (phases, T, P, F) holds the
    shifted copies already (``preprocess.frame = "world"``)."""
    best = np.inf
    for s in range(phases):
        x = a[s] if a.ndim == 4 else np.roll(a, -s, axis=0)
        d = ((x - b) ** 2).mean() + (w * ((np.diff(x, axis=0) - np.diff(b, axis=0)) ** 2).mean() if w else 0.0)
        best = min(best, d)
    return float(best)


def summary(D: np.ndarray, truth: np.ndarray) -> dict:
    """D (manners, videos), lower is better."""
    rank = (D < D[truth, np.arange(len(truth))]).sum(0) + 1
    return {"top1": float(np.mean(rank == 1)), "top5": float(np.mean(rank <= 5)), "mean_rank": float(rank.mean()),
            "hits": (D.argmin(0) == truth).tolist()}


def mcnemar(a: list, b: list) -> dict:
    """Exact McNemar test between two per-video hit lists."""
    a, b = np.array(a), np.array(b)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first": only_a, "only_second": only_b, "p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    args = ap.parse_args()
    from grace_lot.inference.run import observations_for
    run = Path(args.run)
    cfg = load_config(args.config)
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    pred = json.loads((run / "predictive.json").read_text())
    manners, stems = pred["manners"], pred["videos"]
    sealed = set(cfg["split"]["test_actors"])
    train, val = (split_videos(cfg["data"]["raw_dir"], cfg["split"], s) for s in ("train", "val"))
    held = {v.stem: v for m in manners for v in val[m]}
    assert list(held) == stems and not any(v.actor in sealed for v in held.values())
    assert not any(v.actor in sealed for m in manners for v in train[m])
    truth = np.array([manners.index(held[s].manner) for s in stems])
    w, phases = cfg["inference"].get("temporal_weight", 0.0), int(cfg["inference"].get("phases", 1))
    first = lambda o: o[0] if o.ndim == 4 else o                                 # phase 0 of a per-phase copy
    tr = {m: np.stack([first(o) for o in observations_for(cfg, train[m])]) for m in manners}
    ho = np.stack(observations_for(cfg, [held[s] for s in stems]))
    scale = np.concatenate(list(tr.values())).var(axis=(0, 1)) + 1e-9          # (P, F)
    lot = np.array(pred["lpd"])
    res = {"lot": summary(-lot, truth), "n": len(stems), "chance_top1": 1 / len(manners)}
    for name, k in (("raw", 1.0), ("standardised", 1 / np.sqrt(scale))):
        T = {m: x * k for m, x in tr.items()}
        H = ho * k
        proto = np.array([[distance(h, T[m].mean(0), w, phases) for h in H] for m in manners])
        nn = np.array([[min(distance(h, u, w, phases) for u in T[m]) for h in H] for m in manners])
        res[f"prototype_{name}"], res[f"1nn_{name}"] = summary(proto, truth), summary(nn, truth)
    for k in [k for k in res if k.startswith(("prototype", "1nn"))]:
        res[k]["vs_lot"] = mcnemar(res["lot"]["hits"], res[k]["hits"])
        r = res[k]
        print(f"{k}: top-1 {r['top1']:.2f}, top-5 {r['top5']:.2f}, mean rank {r['mean_rank']:.1f}; "
              f"LoT-only hits {r['vs_lot']['only_first']}, baseline-only {r['vs_lot']['only_second']}, "
              f"p {r['vs_lot']['p']:.3f}", flush=True)
    print(f"LoT: top-1 {res['lot']['top1']:.2f}, top-5 {res['lot']['top5']:.2f}, mean rank {res['lot']['mean_rank']:.1f}")
    (run / "kinematic_baselines.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
