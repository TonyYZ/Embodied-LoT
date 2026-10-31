"""Are two versions of the code the same? Fingerprints every rollout and likelihood of a corpus of programs.

    PYTHONPATH=<old copy> python tools/check_rollouts.py --out old.json corpus*.json
    PYTHONPATH=.          python tools/check_rollouts.py --out new.json corpus*.json --against old.json

A corpus is a JSON list of program texts (``tools/bench_search.py --corpus``), to which ``--random`` adds programs drawn
from the grammar: static and periodic ones, moving bodies, dashed edges, several outputs, templates, fields. For each
program and each of its motion states, the fingerprint is the SHA-256 of the exact bytes of the rollout's values and
claim strengths; equal fingerprints are ``np.array_equal`` arrays. Each likelihood is compared as an exact float."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--against", default=None)
    ap.add_argument("--random", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="at most this many programs from each corpus file")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--set", nargs="*", default=[], help="further config settings, as bench_search's")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    sys.path.insert(1, str(Path(__file__).parent))
    from bench_search import ROUND0
    from grace_lot.config import load_config
    from grace_lot.grace.dataset import split_videos
    from grace_lot.inference.grammar import Grammar
    from grace_lot.inference.run import apply_overrides, build, observations_for
    from grace_lot.lot_adapter.adapter import _component_cache, motion_combinations, rollout
    from grace_lot.lot_adapter.program import Program, apply_semantics, parse_program

    cfg = apply_overrides(load_config("configs/grace_v7_parts.toml"), ROUND0 + args.set)
    apply_semantics(cfg["lot"])
    grammar = Grammar.from_config(cfg["lot"])
    rng = random.Random(args.seed)
    programs = []
    for f in args.corpus:
        srcs = json.loads(Path(f).read_text())
        rng.shuffle(srcs)
        programs += srcs[:args.limit] if args.limit else srcs
    for _ in range(args.random):
        programs.append(str(grammar.sample(rng)))
    programs = list(dict.fromkeys(programs))
    observations = observations_for(cfg, split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")["bowing"])
    lik = build(cfg, observations)[3]
    size, frames = lik.size, args.frames
    out = {}

    def digest(*arrays) -> str:
        h = hashlib.sha256()
        for a in arrays:
            a = np.ascontiguousarray(a)
            h.update(repr(a.shape).encode())
            h.update(a.astype(np.float64).tobytes())
        return h.hexdigest()[:24]

    for i, src in enumerate(programs):
        try:
            p = parse_program(src)
        except Exception as e:
            out[src] = dict(error=f"parse {type(e).__name__}")
            continue
        try:
            rec = {}
            for states, w in motion_combinations(p, size):
                _component_cache.clear()
                v, c = rollout(p, frames, size, states, claims=True)
                rec[repr(sorted(states.items()))] = (w, digest(v, c))
            rec["ll"] = lik(p)
            out[src] = rec
        except Exception as e:                  # a crash is behaviour too: it must stay the same crash
            out[src] = dict(error=f"{type(e).__name__}: {e}"[:200])
        if i % 2000 == 0:
            print(i, len(programs), flush=True)
    Path(args.out).write_text(json.dumps(out))
    errs = sum("error" in r for r in out.values())
    print(f"{len(out)} programs, {errs} raised")
    if args.against:
        old = json.loads(Path(args.against).read_text())
        common = [s for s in out if s in old]
        bad = [s for s in common if out[s] != old[s]]
        print(f"compared {len(common)}: {len(bad)} differ")
        for s in bad[:5]:
            print(" ", s[:160], "\n   old", str(old[s])[:200], "\n   new", str(out[s])[:200])
        sys.exit(1 if bad or len(common) < len(out) else 0)


if __name__ == "__main__":
    main()
