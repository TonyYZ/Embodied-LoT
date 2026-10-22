"""Expressivity suite: can the language say what it is meant to say?

    python -m grace_lot.analysis.expressivity [--cases expressivity/cases.toml] [--only containment]
                                              [--lot pure_claims=true solid_figure=true] [--report ...]

Each case names a concept (containment, carrying, support, rhythm, ...), the program intended to express it, and
the boards it should produce, sketched in characters. A concept is pinned down by a prototype program, variants
that should still count (``expect = "match"``, the default) and contrasts that should not (``expect = "differ"``),
all checked against the same sketch. The program is run on the engine (``adapter.rollout``, semantics from
``--config``) under every joint state of its unknown-motion units, and every checked frame is compared with its
sketch. A case file is TOML:

    [[case]]
    concept = "containment"
    name    = "walls around a body"
    program = "(serial_h universal/V (body center/H) universal/V)"   # wrapped in (parallel out ...) if no out
    frames  = [0, 6]                 # frames checked (the rollout runs to the last one); default [0]
    expect  = "match"                # "differ": a contrast, which passes when some check does NOT match
    states  = "all"                  # or one index into motion_combinations; default every joint state
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
    _  not figure: ground or unknown (mean <= 0.5)
    o  content with room around it: the cells of all o blocks together hold some figure (>= 0.75) and some
       room, i.e. cells that are not figure (<= 0.5: ground or unknown)

The report also says when a program lies outside the search grammar (``playground.language_issues``): the engine
still runs it, but inference could never propose it.

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
    if bad := set("".join(rows)) - set("#.?*o_"):
        raise ValueError(f"unknown sketch characters {sorted(bad)}")
    return rows


def coarse(board: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Block means of a board (H, W) at a sketch's resolution."""
    return np.array([[block.mean() for block in np.array_split(band, cols, axis=1)]
                     for band in np.array_split(board, rows, axis=0)])


def blocks(board: np.ndarray, rows: int, cols: int) -> list:
    return [np.array_split(band, cols, axis=1) for band in np.array_split(board, rows, axis=0)]


def mismatches(sketch: List[str], board: np.ndarray) -> np.ndarray:
    """(rows, cols) True where the board's block disagrees with the sketch. The o blocks are judged together:
    all of them are marked when their cells hold no figure (>= 0.75) or no room (<= 0.5)."""
    c = coarse(board, len(sketch), len(sketch[0]))
    want = np.array([list(r) for r in sketch])
    got = np.vectorize(char)(c)
    bad = (want != "*") & (want != "o") & (want != "_") & (want != got)
    bad |= (want == "_") & (c > 0.5 + 1e-9)
    if (want == "o").any():
        b = blocks(board, len(sketch), len(sketch[0]))
        cells = np.concatenate([b[i][j].ravel() for i, j in zip(*np.nonzero(want == "o"))])
        if not ((cells >= FIGURE).any() and (cells <= 0.5).any()):
            bad |= want == "o"
    return bad


def program_of(src: str):
    from grace_lot.lot_adapter.program import parse_program
    if "out" not in src.replace("(", " ").replace(")", " ").split():
        src = f"(parallel out {src})"
    return parse_program(src)


def render(program: str, frames: int, states="all") -> list:
    """[(joint state, (frames, channels, 12, 12) boards)] for every joint state of the unknown-motion units, or
    the one at index ``states``."""
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    p = program_of(program)
    combos = [c for c, _ in motion_combinations(p, SIZE)]
    if states != "all":
        combos = [combos[min(int(states), len(combos) - 1)]]
    out = []
    for c in combos:
        v = rollout(p, frames, SIZE, c)
        out.append((c, v if v.ndim == 4 else v[:, None]))
    return out


def run_case(case: dict, grammar=None) -> dict:
    frames = case.get("frames", [0])
    boards = case.get("boards", {})
    sketches = {t: parse_sketch(boards.get(str(t), boards.get("all", ""))) for t in frames
                if str(t) in boards or "all" in boards}
    differ = case.get("expect", "match") == "differ"
    issues = []
    try:
        if grammar is not None:
            from grace_lot.analysis.playground import language_issues
            issues = language_issues(program_of(case["program"]), grammar)
        rendered = render(case["program"], max(frames) + 1, case.get("states", "all"))
    except Exception as e:                                       # a program the language cannot even run
        return {**case, "ok": False, "matched": False, "error": f"{type(e).__name__}: {e}", "checks": [],
                "issues": issues}
    ch = case.get("channel", 0)
    checks = [(state, t, s, v[t, ch], mismatches(s, v[t, ch])) for state, v in rendered
              for t, s in sorted(sketches.items())]
    matched = all(not m.any() for *_, m in checks)
    return {**case, "ok": matched != differ, "matched": matched, "error": None, "checks": checks,
            "issues": issues}


def report(results: List[dict], semantics: str = "the config's") -> str:
    lines = ["# Expressivity suite", "", f"Semantics: {semantics}", "",
             f"{sum(r['ok'] for r in results)} of {len(results)} cases pass. "
             "Sketch characters: `#` figure (≥ 0.75), `.` ground (≤ 0.25), `?` in between, `*` anything. "
             "In the middle column, `X` marks a block that disagrees with the sketch.", ""]
    for r in results:
        kind = "contrast (should not match)" if r.get("expect") == "differ" else "should match"
        lines += [f"## {'PASS' if r['ok'] else 'FAIL'}: {r['concept']}, {r['name']}", "",
                  f"`{r['program']}`  \n{kind}; {'matches' if r['matched'] else 'does not match'} the sketch"
                  + (f"  \n{r['note']}" if r.get("note") else ""), ""]
        if r["issues"]:
            lines += ["Outside the search grammar: " + "; ".join(r["issues"]), ""]
        if r["error"]:
            lines += [f"Engine error: `{r['error']}`", ""]
            continue
        shown = 0
        for state, t, sketch, board, bad in r["checks"]:
            if not bad.any() and shown >= 2:
                continue                                            # every mismatching frame, and a couple of others
            shown += 1
            got = np.vectorize(char)(coarse(board, len(sketch), len(sketch[0])))
            marked = ["".join("X" if bad[i, j] else got[i, j] for j in range(len(sketch[0])))
                      for i in range(len(sketch))]
            full = ["".join(char(x) for x in row) for row in board]
            h = max(len(sketch), SIZE)
            pad = lambda rows, w: rows + [" " * w] * (h - len(rows))
            cols = zip(pad(sketch, len(sketch[0])), pad(marked, len(sketch[0])), pad(full, SIZE))
            label = f"state {state}, " if state else ""
            lines += [f"{label}frame {t}: sketch | engine at sketch resolution | engine 12 x 12", "```"]
            lines += [f"{a}   {b}   {c}" for a, b, c in cols] + ["```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="expressivity/cases.toml")
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--report", default="expressivity/report.md")
    ap.add_argument("--only", nargs="*", help="concepts to run")
    ap.add_argument("--lot", nargs="*", default=[], help="semantics overrides, e.g. pure_claims=true")
    args = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # case names may use symbols (⊏, Π)
    from grace_lot.config import load_config
    from grace_lot.inference.grammar import Grammar
    from grace_lot.lot_adapter.program import apply_semantics
    lot = load_config(args.config)["lot"]
    for item in args.lot:
        key, value = item.split("=", 1)
        lot[key] = {"true": True, "false": False}.get(value.lower(), value)
    apply_semantics(lot)
    grammar = Grammar.from_config(lot)
    cases = tomllib.loads(Path(args.cases).read_text(encoding="utf-8"))["case"]
    results = [run_case(c, grammar) for c in cases if not args.only or c["concept"] in args.only]
    Path(args.report).write_text(report(results, ", ".join(args.lot) or "the config's"), encoding="utf-8")
    for r in results:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['concept']:12s} {r['name']}" + (f"  ({r['error']})" if r["error"] else "")
              + ("  [outside the search grammar]" if r["issues"] else ""))
    print(f"{sum(r['ok'] for r in results)}/{len(results)} pass; report: {args.report}")


if __name__ == "__main__":
    main()
