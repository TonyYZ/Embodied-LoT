"""Choose lambda (prior strength) and beta (likelihood sharpness) on the validation actors.

    python -m grace_lot.inference.select [--out results/grace_v2]

Stage 1: every (lambda, beta) in ``[selection]``.
Stage 2 (skipped with --no-variants): at the best (lambda, beta), each entry of ``[selection].variants``
(search bounds, MCMC length, proposal mixture, ...), given as config overrides.

For each setting:
  1. infer each manner's posterior from the 15 training actors;
  2. recognise the 5 validation actors' videos (``analysis.heldout``).
The chosen setting has the best top-1 accuracy (ties: mean reciprocal rank). With
``[selection].max_r_hat``, only settings whose MCMC converges (median R-hat of the
log-posterior over manners at most that) are eligible.
Neither human similarity data nor the test actors are read. Results go to
``<out>/select/`` and are rewritten after every setting. Settings whose
posteriors already exist are reused, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from grace_lot.analysis.heldout import evaluate
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.inference.run import apply_overrides, run


def _key(r):
    return r["lot_top1"], r["lot_mrr"]


def _best(rows, max_r_hat=None):
    """The best setting, among those that converge when ``max_r_hat`` is given (all, if none does)."""
    ok = [r for r in rows if max_r_hat is None or r.get("r_hat_median", np.inf) <= max_r_hat]
    return max(ok or rows, key=_key)


def write(out: Path, rows, max_r_hat=None) -> list:
    best = _best(rows, max_r_hat)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps({"rows": rows, "best": best, "chosen_dir": best["dir"],
                                                    "max_r_hat": max_r_hat}, indent=1))
    lines = ["| setting | top-1 | top-5 | MRR | R-hat median | R-hat max | kin prototype top-1 | kin 1-NN top-1 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['tag']} | {r['lot_top1']:.3f} | {r['lot_top5']:.3f} | {r['lot_mrr']:.3f} | "
              f"{r.get('r_hat_median', float('nan')):.2f} | {r.get('r_hat_max', float('nan')):.2f} | "
              f"{r['kin_prototype_top1']:.3f} | {r['kin_1nn_top1']:.3f} |" for r in rows]
    rule = f"best top-1 among settings with median R-hat <= {max_r_hat}" if max_r_hat else "best top-1"
    lines += ["", f"Selected: {best['tag']} ({rule}; validation actors only; chance top-1 = 0.038)"]
    (out / "selection.md").write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default="results/grace_v2")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-variants", action="store_true", help="stop after the (lambda, beta) grid")
    args = ap.parse_args()
    base = load_config(args.config)
    out = Path(args.out) / "select"
    d = base["data"]
    train = split_videos(d["raw_dir"], base["split"], "train")
    val = split_videos(d["raw_dir"], base["split"], "val")
    rows = []
    max_r_hat = base["selection"].get("max_r_hat")

    def trial(tag, overrides):
        cfg = apply_overrides(base, overrides)
        run_dir = out / tag
        if len(list((run_dir / "posteriors").glob("*.json"))) < len(train):
            run(cfg, "train", run_dir, workers=args.workers, quiet=True)
        res = evaluate(cfg, run_dir / "posteriors", train, val)
        rh = [json.loads(f.read_text())["r_hat_log_post"] for f in (run_dir / "posteriors").glob("*.json")]
        row = {"tag": tag, "overrides": overrides, "dir": str(run_dir),
               "r_hat_median": float(np.median(rh)), "r_hat_max": float(np.max(rh)),
               **{f"lot_{k}": v for k, v in res["lot"].items()},
               "kin_prototype_top1": res["kin_prototype"]["top1"], "kin_1nn_top1": res["kin_1nn"]["top1"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        write(out, rows, max_r_hat)

    weights = base["selection"].get("temporal_weight")          # optional third axis
    for lam, beta, w in itertools.product(base["selection"]["lambda_size"], base["selection"]["beta"],
                                          weights if weights else [None]):
        tag = f"lambda{lam:g}_beta{beta:g}" + (f"_w{w:g}" if w is not None else "")
        extra = [f"inference.temporal_weight={w}"] if w is not None else []
        trial(tag, [f"inference.lambda_size={lam}", f"inference.beta={beta}"] + extra)
    stage1 = _best(rows, max_r_hat)
    for name, extra in ({} if args.no_variants else base["selection"].get("variants", {})).items():
        trial(f"{stage1['tag']}_{name}", stage1["overrides"] + list(extra))
    print("\n".join(write(out, rows, max_r_hat)))


if __name__ == "__main__":
    main()
