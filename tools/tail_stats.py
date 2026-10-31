"""The rollout cost distribution over a corpus of programs, and how each stage scales with the number of units.

    PYTHONPATH=. python tools/tail_stats.py dist --out d.json corpus.json     per-program seconds and tail shares
    PYTHONPATH=. python tools/tail_stats.py ticks --out t.json --from d.json  per-tick trace of programs at chosen ranks

``dist`` times every program as the likelihood would (every motion state, 32 ticks, claims). ``ticks`` steps the slowest
programs and a sample of normal ones and records, per tick, the units, how many are off the board and how far, orders,
claims painted and the wall time in each stage, by wrapping the stages here (nothing is added to the engine)."""

from __future__ import annotations

import argparse
import functools
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np

FRAMES, SIZE = 32, 12


def dist(args) -> None:
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    from grace_lot.lot_adapter.program import parse_program
    rows, rng = [], random.Random(1)
    for f in args.corpus:
        srcs = json.loads(Path(f).read_text())
        rng.shuffle(srcs)
        for src in srcs[:args.limit or None]:
            p = parse_program(src)
            t = time.perf_counter()
            combos = motion_combinations(p, SIZE)
            for states, _ in combos:
                rollout(p, FRAMES, SIZE, states, claims=True)
            rows.append(dict(src=src, seconds=time.perf_counter() - t, states=len(combos)))
    Path(args.out).write_text(json.dumps(rows))
    print(summary([r["seconds"] for r in rows]))


def summary(t) -> str:
    t = np.sort(np.asarray(t))[::-1]
    c, med = np.cumsum(t) / t.sum(), np.median(t)
    p95, p99 = np.percentile(t, 95), np.percentile(t, 99)
    return (f"n {len(t)} total {t.sum():.1f}s median {med * 1e3:.1f}ms p95 {p95 * 1e3:.0f}ms p99 {p99 * 1e3:.0f}ms "
            f"max {t[0] * 1e3:.0f}ms | top1% {c[max(0, int(.01 * len(t)) - 1)]:.0%} top5% {c[max(0, int(.05 * len(t)) - 1)]:.0%} "
            f"| p95/med {p95 / med:.1f} p99/med {p99 / med:.1f}")


def ticks(args) -> None:
    import embodied_lot as E
    from grace_lot.lot_adapter.adapter import motion_combinations
    from grace_lot.lot_adapter.program import compile_program, parse_program
    totals = {}

    def timed(owner, name):
        fn = getattr(owner, name)
        depth = [0]

        @functools.wraps(fn)
        def inner(*a, **k):
            depth[0] += 1
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                depth[0] -= 1
                if not depth[0]:
                    totals[name] = totals.get(name, 0.0) + time.perf_counter() - t
                    totals[name + "#"] = totals.get(name + "#", 0) + 1
        setattr(owner, name, inner)

    for name in ("_paint", "_render_all", "_issue_orders", "_motion_field", "_move_bodies_gameboard", "_unit_figures",
                 "_chain_move", "_spawn_bodies", "_reset_fields", "_flow_begin", "_flow_end", "node_views",
                 "_perceive", "_scan"):
        timed(E.ConceptGraph, name)
    timed(E, "_claim_values")
    rows = sorted(json.loads(Path(args.dist).read_text()), key=lambda r: -r["seconds"])
    n = len(rows)
    ranks = dict(top=range(args.top), p99=range(int(.01 * n), int(.01 * n) + args.each),
                 p95=range(int(.05 * n), int(.05 * n) + args.each), median=range(n // 2, n // 2 + args.each))
    out = []
    for group, rs in ranks.items():
        for r in rs:
            src = rows[r]["src"]
            p = parse_program(src)
            for states, _ in motion_combinations(p, SIZE):
                g = compile_program(p)
                g.grid = (SIZE, SIZE)
                g.reset_gates()
                g.assign_roles()
                if states:
                    g.motion_states = dict(states)
                g.windows()
                for t in range(FRAMES):
                    totals.clear()
                    t0 = time.perf_counter()
                    g.step()
                    wall = time.perf_counter() - t0
                    toks = list(g._tokens().values())
                    placed = [b for b in toks if getattr(b, "position", None) is not None and getattr(b, "extent", None) is not None]
                    off, far = 0, 0
                    for b in placed:
                        r0, r1, c0, c1 = b.window(SIZE, SIZE)
                        d = max(-r1, r0 - SIZE, -c1, c0 - SIZE)
                        if d >= 0:
                            off += 1
                        far = max(far, d)
                    out.append(dict(group=group, rank=r, tick=t, wall=wall, units=len(toks), off=off, far=far,
                                    orders=sum(len(getattr(b, "orders", ())) for b in toks),
                                    sub_orders=sum(len(S["orders"]) for S in g.__dict__.get("_sub", {}).values()),
                                    overhang=0, stages=dict(totals)))
                    if time.perf_counter() - t0 > args.budget:
                        break
    Path(args.out).write_text(json.dumps(out))
    print(len(out), "ticks")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("dist", "ticks"))
    ap.add_argument("corpus", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--from", dest="dist", default=None)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--each", type=int, default=4)
    ap.add_argument("--budget", type=float, default=30.0, help="stop a program once one tick takes this long")
    args = ap.parse_args()
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics({"subcell": 1})
    dist(args) if args.mode == "dist" else ticks(args)


if __name__ == "__main__":
    main()
