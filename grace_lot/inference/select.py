"""Choose lambda (prior strength) and beta (likelihood sharpness) on the validation actors.

    python -m grace_lot.inference.select [--out results/grace_v2]

Stage 1: every (lambda, beta) in ``[selection]``.
Stage 2: at the best (lambda, beta), each entry of ``[selection].variants``
(search bounds, MCMC length, proposal mixture, ...), given as config overrides.

For each setting:
  1. infer each manner's posterior from the 15 training actors;
  2. recognise the 5 validation actors' videos (``analysis.heldout``).
The chosen setting has the best top-1 accuracy (ties: mean reciprocal rank).
Neither human similarity data nor the test actors are read. Results go to
``<out>/select/`` and are rewritten after every setting. Settings whose
posteriors already exist are reused, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from grace_lot.analysis.heldout import evaluate
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.inference.run import apply_overrides, run


def _key(r):
    return r["lot_top1"], r["lot_mrr"]


def write(out: Path, rows) -> list:
    best = max(rows, key=_key)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps({"rows": rows, "best": best, "chosen_dir": best["dir"]}, indent=1))
    lines = ["| setting | top-1 | top-5 | MRR | kin prototype top-1 | kin 1-NN top-1 |",
             "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['tag']} | {r['lot_top1']:.3f} | {r['lot_top5']:.3f} | {r['lot_mrr']:.3f} | "
              f"{r['kin_prototype_top1']:.3f} | {r['kin_1nn_top1']:.3f} |" for r in rows]
    lines += ["", f"Selected: {best['tag']} (validation actors only; chance top-1 = 0.038)"]
    (out / "selection.md").write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default="results/grace_v2")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    base = load_config(args.config)
    out = Path(args.out) / "select"
    d = base["data"]
    train = split_videos(d["raw_dir"], base["split"], "train")
    val = split_videos(d["raw_dir"], base["split"], "val")
    rows = []

    def trial(tag, overrides):
        cfg = apply_overrides(base, overrides)
        run_dir = out / tag
        if len(list((run_dir / "posteriors").glob("*.json"))) < len(train):
            run(cfg, "train", run_dir, workers=args.workers, quiet=True)
        res = evaluate(cfg, run_dir / "posteriors", train, val)
        row = {"tag": tag, "overrides": overrides, "dir": str(run_dir),
               **{f"lot_{k}": v for k, v in res["lot"].items()},
               "kin_prototype_top1": res["kin_prototype"]["top1"], "kin_1nn_top1": res["kin_1nn"]["top1"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        write(out, rows)

    for lam, beta in itertools.product(base["selection"]["lambda_size"], base["selection"]["beta"]):
        trial(f"lambda{lam:g}_beta{beta:g}", [f"inference.lambda_size={lam}", f"inference.beta={beta}"])
    stage1 = max(rows, key=_key)
    for name, extra in base["selection"].get("variants", {}).items():
        trial(f"{stage1['tag']}_{name}", stage1["overrides"] + list(extra))
    print("\n".join(write(out, rows)))


if __name__ == "__main__":
    main()
