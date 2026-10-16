"""Diagnostic rerun of the selected setting with a denser tempering ladder.

    python -m grace_lot.inference.dense_ladder_check

The selected setting (lambda 1, beta 4000, bigger search) mixed poorly: the
cold-end swap rate was 0.07 and the median R-hat 1.74. This reruns it with
24 replicas instead of 12 and reports validation recognition and R-hat
only. No human data and no test actors are read.
"""

import json
from pathlib import Path

import numpy as np

from grace_lot.analysis.heldout import evaluate
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.inference.run import apply_overrides, run

if __name__ == "__main__":
    root = Path("results/grace_v2/select")
    best = json.loads((root / "selection.json").read_text())["best"]
    cfg = apply_overrides(load_config(), best["overrides"] + ["inference.replicas=24"])
    out = root / (best["tag"] + "_dense_ladder")
    d = cfg["data"]
    train = split_videos(d["raw_dir"], cfg["split"], "train")
    val = split_videos(d["raw_dir"], cfg["split"], "val")
    if len(list((out / "posteriors").glob("*.json"))) < len(train):
        run(cfg, "train", out, workers=24, quiet=True)
    res = evaluate(cfg, out / "posteriors", train, val)
    posts = [json.loads(p.read_text()) for p in (out / "posteriors").glob("*.json")]
    rh = [p["r_hat_log_post"] for p in posts]
    swaps = np.mean([c["swaps"] for p in posts for c in p["chains"]], axis=0)
    summary = {"tag": out.name, "overrides": best["overrides"] + ["inference.replicas=24"],
               "lot": res["lot"], "r_hat_median": float(np.median(rh)), "r_hat_max": float(np.max(rh)),
               "n_r_hat_below_1.1": int(sum(r < 1.1 for r in rh)), "swap_rates": swaps.round(2).tolist()}
    (out / "dense_ladder_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
