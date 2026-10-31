"""Where each program's score comes from, on the first run's descriptors.

    python -m grace_lot.analysis.compare_scoring results/.../full_run results/.../motion_run

Per video and part, the matched cost split into motion (rhythm amplitudes), posture and limb pairs, against the empty
program, for each run's MAP; also its nodes and outputs, and its share of the way from the empty program to an ideal one.
Writes ``compare_scoring.json`` in the first run (read by ``results_page``). Training actors only."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.inference.descriptor_likelihood import POSTURE, PAIRS, _coupling
from grace_lot.inference.run import build, observations_for
from grace_lot.lot_adapter.adapter import motion_combinations, rollout
from grace_lot.lot_adapter.program import OUT, apply_semantics, nodes, parse_program, saved_lot, size


def split(lik, p) -> np.ndarray:
    """Summed over videos: motion, posture and pair NLL under the program's best combination per video."""
    L, matched = lik._scores(p)
    combos = list(motion_combinations(p, lik.size))
    out, np_ = np.zeros(3), len(POSTURE)
    for n, b in enumerate(L.argmax(0)):
        v = rollout(p, lik.frames + lik.warmup, lik.size, combos[b][0], claims=lik.expected)
        v, cl = v if lik.expected else (v, None)
        v, cl = v[lik.warmup:], None if cl is None else cl[lik.warmup:]
        v, cl = (v, cl) if v.ndim == 4 else (v[:, None], None if cl is None else cl[:, None])
        K = v.shape[1]
        rows, cols, _ = matched[b][n]
        per = {}
        for k, q in zip(rows, cols):
            if k < K:
                a, z = lik._channel(np.ascontiguousarray(v[:, k]), k,
                                    None if cl is None else np.ascontiguousarray(cl[:, k]))
                mu, var = a.mean(0), a.var(0)
                per[q] = z
            else:
                mu, var = lik.bg["amu"], lik.bg["avar"]
            V = var + lik.s2[q]
            c = 0.5 * (((lik.amps[n, q] - mu) ** 2) / V + np.log(2 * np.pi * V))
            out[0] += c[:-np_].sum()
            out[1] += c[-np_:].sum()
        for j, (pa, pb) in enumerate(PAIRS):
            if pa in per and pb in per:
                x = _coupling(per[pa], per[pb])
                m, sv = x.mean(0), x.var(0)
            else:
                m, sv = lik.bg["cmu"][j], lik.bg["cvar"][j]
            sv = sv + lik.s2d[j]
            out[2] += (0.5 * (((lik.coup[n, j] - m) ** 2) / sv + np.log(2 * np.pi * sv))).sum()
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    full, motion = Path(sys.argv[1]), Path(sys.argv[2])
    rc = json.loads((full / "config.json").read_text())
    cfg = load_config("configs/grace_v7_parts.toml")
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc[k]
    cfg["lot"] = saved_lot(rc["lot"])
    cfg["inference"].pop("library", None)                  # the fits are compared as programs
    apply_semantics(cfg["lot"])
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    report = {}
    for m in sorted(f.stem for f in (full / "posteriors").glob("*.json")):
        lik = build(cfg, observations_for(cfg, train[m]))[3]
        e = split(lik, parse_program("out"))
        mu, cm, ideal = lik.amps.mean(0), lik.coup.mean(0), 0.0
        for a, c in zip(lik.amps, lik.coup):
            C = 0.5 * (((a[None] - mu[:, None]) ** 2) / lik.s2[None] + np.log(2 * np.pi * lik.s2[None])).sum(-1)
            r, k = linear_sum_assignment(C)
            ideal -= C[r, k].sum() + 0.5 * (((c - cm) ** 2) / lik.s2d + np.log(2 * np.pi * lik.s2d)).sum()
        empty = -e.sum()
        report[m] = dict(empty=empty, ideal=ideal)
        for name, run in (("full fit", full), ("motion fit", motion)):
            p = parse_program(json.loads((run / "posteriors" / f"{m}.json").read_text())["map"])
            g, ll = e - split(lik, p), lik(p)
            report[m][name] = dict(loglik=ll, share=(ll - empty) / (ideal - empty), nodes=size(p.tree),
                                   outputs=sum(n.kind == OUT for _, n in nodes(p.tree)), moving=str(p).count("*"),
                                   motion=g[0], posture=g[1], pairs=g[2])
            print(m, name, {k: round(float(v), 2) for k, v in report[m][name].items()}, flush=True)
    (full / "compare_scoring.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
