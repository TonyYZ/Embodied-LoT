"""Compare fits of the same manners on different boards (validation actors only; densities differ by board, so
they are not compared).

    python -m grace_lot.analysis.board_compare control=results/x/all26_current fine=results/x/fine24_5 ...

The first run's manners that every run has are used. Per run:
  recognition   top-1 among those manners, from ``predictive.json`` (restricted to them and their videos)
  templates     the best kinematic template's top-1 on the same videos and features (``kinematic_baselines``)
  moving        share of posterior draws whose outputs change over the frames, per manner
Writes ``board_compare.json`` next to the last run.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from grace_lot.config import load_config
from grace_lot.lot_adapter.adapter import motion_combinations, rollout
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot


def recognition(pred: dict, manners: list) -> float:
    idx = [pred["manners"].index(m) for m in manners]
    cols = [i for i, v in enumerate(pred["videos"]) if v.split("_", 1)[1] in manners]
    L = np.array(pred["lpd"])[np.ix_(idx, cols)]
    truth = np.array([manners.index(pred["videos"][i].split("_", 1)[1]) for i in cols])
    return float(np.mean(L.argmax(0) == truth))


def moving_share(run: Path, manners: list, draws: int = 300, seed: int = 0) -> dict:
    rc = json.loads((run / "config.json").read_text())
    apply_semantics(saved_lot(rc["lot"]))
    pre, rng, out = rc["preprocess"], random.Random(seed), {}
    for m in manners:
        post = json.loads((run / "posteriors" / f"{m}.json").read_text())["posterior"]
        progs, w = zip(*post)
        sample = rng.choices(progs, w, k=draws)
        moves = {}
        for s in set(sample):
            p = parse_program(s)
            v = [rollout(p, pre["frames"], pre["field_size"], st) for st, _ in motion_combinations(p, pre["field_size"])]
            moves[s] = any(np.abs(np.diff(x, axis=0)).max() > 0 for x in v)
        out[m] = float(np.mean([moves[s] for s in sample]))
    return out


def templates(run: Path, manners: list) -> float:
    """Best kinematic template top-1 on these manners (runs ``kinematic_baselines`` on a restricted copy)."""
    import shutil
    import subprocess
    import tempfile
    pred = json.loads((run / "predictive.json").read_text())
    if sorted(pred["manners"]) != sorted(manners):
        idx = [pred["manners"].index(m) for m in manners]
        cols = [i for i, v in enumerate(pred["videos"]) if v.split("_", 1)[1] in manners]
        pred = dict(pred, manners=manners, videos=[pred["videos"][i] for i in cols],
                    lpd=np.array(pred["lpd"])[np.ix_(idx, cols)].tolist())
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(run / "config.json", Path(d) / "config.json")
        (Path(d) / "predictive.json").write_text(json.dumps(pred))
        subprocess.run([sys.executable, "-m", "grace_lot.analysis.kinematic_baselines", "--run", d], check=True,
                       capture_output=True)
        k = json.loads((Path(d) / "kinematic_baselines.json").read_text())
    return max(v["top1"] for b, v in k.items() if b.startswith(("prototype", "1nn")))


def main() -> None:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    runs = {k: Path(v) for k, v in runs.items()}
    have = [set(json.loads((r / "predictive.json").read_text())["manners"]) for r in runs.values()]
    first = json.loads((next(iter(runs.values())) / "predictive.json").read_text())["manners"]
    manners = [m for m in first if all(m in h for h in have)]
    res = {"manners": manners}
    for name, r in runs.items():
        pred = json.loads((r / "predictive.json").read_text())
        res[name] = {"recognition": recognition(pred, manners), "templates": templates(r, manners),
                     "moving": moving_share(r, manners)}
        x = res[name]
        print(f"{name:10s} recognition {x['recognition']:.2f}  templates {x['templates']:.2f}  moving "
              + " ".join(f"{m} {v:.2f}" for m, v in x["moving"].items()), flush=True)
    (list(runs.values())[-1] / "board_compare.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
