"""Expressivity suite: can the language say what it is meant to say?

    python -m grace_lot.analysis.expressivity [--cases expressivity/cases.toml] [--only containment]

Each case names a concept (containment, carrying, support, rhythm, ...), the program intended to express it, and
the boards it should produce, sketched in characters. The program is run on the engine (``adapter.rollout``,
semantics from ``--config``) and every checked frame is compared with its sketch. A case file is TOML:

    [[case]]
    concept = "containment"
    name    = "walls around a body"
    program = "(serial_h universal/V (body center/H) universal/V)"   # wrapped in (parallel out ...) if no out
    frames  = [0, 6]                 # frames checked (the rollout runs to the last one); default [0]
    state   = 0                      # which joint state of the unknown-motion units (motion_combinations)
    channel = 0                      # which output window
    note    = "what the case is about"
    [case.boards]
    all = '''                        # "all": every checked frame; or one entry per frame: "0" = ..., "6" = ...
    #..#
    #..#
    '''

A sketch is any grid of rows of equal length, top row first. Each character covers one block of the 12 x 12
board (a 4 x 4 sketch covers 3 x 3 cells per character) and is checked against the block's mean value:

    #  figure (yang): mean >= 0.75        .  ground (yin): mean <= 0.25
    ?  in between: unknown, or a block that is part figure    *  anything

Writes a markdown report (``--report``, default ``expressivity/report.md``): per case PASS / FAIL, and per
checked frame the sketch, the board at the sketch's resolution with mismatches marked ``X``, and the full board.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Dict, List

import numpy as np

SIZE = 12
FIGURE, GROUND = 0.75, 0.25


def char(v: float) -> str:
    return "#" if v >= FIGURE else "." if v <= GROUND else "?"


def parse_sketch(text: str) -> List[str]:
    rows = [r.strip() for r in text.strip("\n").splitlines() if r.strip()]
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError(f"a sketch needs rows of equal length:\n{text}")
    if bad := set("".join(rows)) - set("#.?*"):
        raise ValueError(f"unknown sketch characters {sorted(bad)}")
    return rows


def coarse(board: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Block means of a board (H, W) at a sketch's resolution."""
    return np.array([[block.mean() for block in np.array_split(band, cols, axis=1)]
                     for band in np.array_split(board, rows, axis=0)])


def mismatches(sketch: List[str], board: np.ndarray) -> np.ndarray:
    """(rows, cols) True where the board's block disagrees with the sketch."""
    c = coarse(board, len(sketch), len(sketch[0]))
    want = np.array([list(r) for r in sketch])
    got = np.vectorize(char)(c)
    return (want != "*") & (want != got)


def render(program: str, frames: int, state: int = 0) -> np.ndarray:
    """(frames, channels, 12, 12) boards of a program under one joint state of its unknown-motion units."""
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    from grace_lot.lot_adapter.program import parse_program
    if "out" not in program.replace("(", " ").replace(")", " ").split():
        program = f"(parallel out {program})"
    p = parse_program(program)
    combos = motion_combinations(p, SIZE)
    v = rollout(p, frames, SIZE, combos[min(state, len(combos) - 1)][0])
    return v if v.ndim == 4 else v[:, None]


def run_case(case: dict) -> dict:
    frames = case.get("frames", [0])
    boards = case.get("boards", {})
    sketches = {t: parse_sketch(boards.get(str(t), boards.get("all", ""))) for t in frames
                if str(t) in boards or "all" in boards}
    try:
        v = render(case["program"], max(frames) + 1, case.get("state", 0))[:, case.get("channel", 0)]
    except Exception as e:                                       # a program the language cannot even run
        return {**case, "ok": False, "error": f"{type(e).__name__}: {e}", "checks": []}
    checks = [(t, s, v[t], mismatches(s, v[t])) for t, s in sorted(sketches.items())]
    return {**case, "ok": all(not m.any() for *_, m in checks), "error": None, "checks": checks}


def report(results: List[dict]) -> str:
    lines = ["# Expressivity suite", "",
             f"{sum(r['ok'] for r in results)} of {len(results)} cases pass. "
             "Sketch characters: `#` figure (≥ 0.75), `.` ground (≤ 0.25), `?` in between, `*` anything. "
             "In the middle column, `X` marks a block that disagrees with the sketch.", ""]
    for r in results:
        lines += [f"## {'PASS' if r['ok'] else 'FAIL'}: {r['concept']}, {r['name']}", "",
                  f"`{r['program']}`" + (f"  \n{r['note']}" if r.get("note") else ""), ""]
        if r["error"]:
            lines += [f"Engine error: `{r['error']}`", ""]
            continue
        for t, sketch, board, bad in r["checks"]:
            got = np.vectorize(char)(coarse(board, len(sketch), len(sketch[0])))
            marked = ["".join("X" if bad[i, j] else got[i, j] for j in range(len(sketch[0])))
                      for i in range(len(sketch))]
            full = ["".join(char(x) for x in row) for row in board]
            h = max(len(sketch), SIZE)
            pad = lambda rows, w: rows + [" " * w] * (h - len(rows))
            cols = zip(pad(sketch, len(sketch[0])), pad(marked, len(sketch[0])), pad(full, SIZE))
            lines += [f"frame {t}: sketch | engine at sketch resolution | engine 12 x 12", "```"]
            lines += [f"{a}   {b}   {c}" for a, b, c in cols] + ["```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="expressivity/cases.toml")
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--report", default="expressivity/report.md")
    ap.add_argument("--only", nargs="*", help="concepts to run")
    args = ap.parse_args()
    from grace_lot.config import load_config
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(load_config(args.config)["lot"])
    cases = tomllib.loads(Path(args.cases).read_text(encoding="utf-8"))["case"]
    results = [run_case(c) for c in cases if not args.only or c["concept"] in args.only]
    Path(args.report).write_text(report(results), encoding="utf-8")
    for r in results:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['concept']:12s} {r['name']}" + (f"  ({r['error']})" if r["error"] else ""))
    print(f"{sum(r['ok'] for r in results)}/{len(results)} pass; report: {args.report}")


if __name__ == "__main__":
    main()
