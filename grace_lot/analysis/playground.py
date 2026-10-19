"""A playground for the v7 language: type a program, step it on the gameboard, look for bugs.

    python -m grace_lot.analysis.playground [--config configs/grace_v7_parts.toml] [--port 8765]

then open http://localhost:8765 for the graph visualizer (``visualizer/page.html``: edit a graph, load or
export v7 program text, run it on the engine and replay the ticks), or /gameboard for the board-first page.
The pages send programs to this server, which runs the real engine
(``embodied_lot``) under the config's semantics and records every tick phase by phase, as
``ConceptGraph._step_gameboard`` runs them. Each run also checks itself: the recorded boards must equal a
twin graph stepped with ``step()``, and the component rollouts (``adapter.rollout``) must equal whole-program
rollouts; any difference is reported on the page as a bug.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np

import embodied_lot
from embodied_lot import BodyNode, GroupNode, ObjectNode, PlanNode
from grace_lot.analysis.gameboard_traces import board_cells, node_label, tree_layout
from grace_lot.config import load_config
from grace_lot.inference.grammar import Grammar
from grace_lot.lot_adapter import adapter
from grace_lot.lot_adapter.program import (apply_semantics, as_program, compile_program, depth, edges_valid,
                                           is_valid, motion_units, nodes, parse_program, size, windows_as_content)

GRID = 12
_lock = Lock()                       # the engine's semantics are module flags: one run at a time

EXAMPLES = [
    ("One body glides", "(parallel (serial_h (body (parallel center/V bottom/H*)) neutral/H neutral/H) out)"),
    ("Two bodies in file",
     "(parallel (serial_h (body (parallel center/V bottom/H*)) (body (parallel center/V bottom/H*)) neutral/H) out)"),
    ("A field does not drive itself", "(parallel (serial_v (field (parallel center/V bottom/V*)) out) out)"),
    ("A body drives itself", "(parallel (serial_v (body (parallel center/V bottom/V*)) out) out)"),
    ("Template copies, each its own instance",
     "(parallel (serial_v (template periphery/H (field center/V)) out) (serial_h bottom/V* out out))"),
    ("Unknown direction (universal)", "(parallel center/V universal/V* out)"),
    ("Unknown whether it moves (neutral)", "(parallel center/H neutral/V* out)"),
    ("Known still (void)", "(parallel center/V void/V* out)"),
    ("A tree gating another tree", "(parallel center/H bottom/H*) (parallel center/V out) | .0>.1.0"),
    ("Measure tree", "(serial_h universal/H neutral/H neutral/H) (parallel center/H out) | .0>.1.0"),
    ("Self-observing oscillator",
     "(parallel (serial_h (parallel bottom/H* center/H) (parallel top/H* center/H (body center/H))) out) "
     "| .0.0.1>.0.0.0 .0.1.1>.0.1.0"),
]


def language_issues(prog, grammar: Grammar) -> list:
    """Why a program lies outside the configured language (the conditions ``SizePrior`` checks)."""
    g, t, out = grammar, prog.tree, []
    forest = 1 if g.output_leaves and t.kind == "outputs" else 0
    if size(t) - forest > g.max_nodes:
        out.append(f"{size(t) - forest} nodes, more than max_nodes = {g.max_nodes}")
    if depth(t) - forest > g.max_depth:
        out.append(f"depth {depth(t) - forest}, more than max_depth = {g.max_depth}")
    if not is_valid(t):
        out.append("not a well-formed tree (child counts or node kinds)")
    if len(prog.dashed) > g.max_dashed:
        out.append(f"{len(prog.dashed)} dashed edges, more than max_dashed = {g.max_dashed}")
    if not edges_valid(prog, g.dashed_targets):
        out.append(f"a dashed edge the search could not add (targets: {g.dashed_targets})")
    if g.output_leaves and not 1 <= sum(n.kind == "out" for _, n in nodes(t)) <= g.max_outputs:
        out.append(f"needs 1 to {g.max_outputs} output windows (out)")
    if g.content_windows and not windows_as_content(t, g.grouping):
        out.append("an out/in window used as a template (windows may only be content)")
    if g.max_motion_units is not None and motion_units(prog, g.grouping) > g.max_motion_units:
        out.append(f"more than {g.max_motion_units} unknown-motion units")
    if not out and g.log_prob(t) == -math.inf:
        out += grammar_issues(t, g) or ["the grammar gives it probability 0"]
    return out


def grammar_issues(t, g: Grammar) -> list:
    """The nodes ``Grammar.log_prob`` rejects, named by path."""
    out = []
    first = 1 if t.kind == "outputs" else 0

    def visit(p, path, d):
        where = "." + ".".join(map(str, path)) if path else "the root"
        if p.kind == "obj":
            if p.params[0] not in g.primitives:
                out.append(f"{where}: primitive {p.params[0]} is not in the language")
            elif p.params[2] and not g.allow_dynamic:
                out.append(f"{where}: dynamic leaves are off")
            return
        if p.kind in ("out", "in"):
            return
        if g.leaf_forced(d):
            out.append(f"{where}: a {p.kind} at depth {d}; at depth {g.max_depth - 1} and below only leaves "
                       f"(max_depth = {g.max_depth})")
        elif p.kind not in g.compositions:
            out.append(f"{where}: {p.kind} is not one of the compositions {list(g.compositions)}")
        elif len(p.children) not in g.child_counts(p.kind):
            r = g.child_counts(p.kind)
            out.append(f"{where}: {p.kind} has {len(p.children)} children; it may have {r.start} to {r.stop - 1}")
        for i, c in enumerate(p.children):
            visit(c, path + (i,), d + 1)
    if t.kind == "outputs":
        for k, c in enumerate(t.children):
            visit(c, (k,), first)
    else:
        visit(t, (), 1 if g.max_outputs and g.output_leaves else 0)
    return out


def forest_layout(g) -> tuple:
    """Each tree laid out on its own, side by side: node positions and the tree each node belongs to."""
    pos, tree_of, x0 = {}, {}, 0
    for r in g.roots:
        root = r.children[1]
        lay = tree_layout(g, root)
        for nid, (x, y) in lay.items():
            pos[nid], tree_of[nid] = (x + x0, y), r.node_id
        x0 += max(x for x, _ in lay.values()) + 1.6
    return pos, tree_of


def _cells(mask) -> list:
    rows, cols = np.nonzero(mask)
    return [[int(r), int(c)] for r, c in zip(rows, cols)]


def _order(o) -> dict:
    return {"axis": "vertical" if o[0] == 0 else "horizontal", "direction": int(np.sign(o[1])),
            "cells": round(float(o[2]), 2), "rate": round(float(o[3]), 3), "credit": round(float(o[4]), 3),
            "leaf": o[5]}


def prepare(src: str, cfg_path: str) -> tuple:
    """The program under the config's semantics, why it is outside the language (if it is), a fresh graph."""
    cfg = load_config(cfg_path)
    apply_semantics(cfg["lot"])
    tree_part = src.split("|")[0]
    if tree_part.count("(") != tree_part.count(")"):            # parse_program fails obscurely on these
        raise ValueError(f"unbalanced parentheses: {tree_part.count('(')} '(' and {tree_part.count(')')} ')'")
    prog = as_program(parse_program(src))
    return prog, language_issues(prog, Grammar.from_config(cfg["lot"])), fresh_graph(prog)


def fresh_graph(prog):
    g = compile_program(prog)
    g.grid = (GRID, GRID)
    g.reset_gates()
    g.assign_roles()
    return g


def viewer_snapshot(g, label: str, masks: dict) -> dict:
    """The graph in ``visualize_lot``'s snapshot format; each node's field is its tree's space on its own cells."""
    from embodied_lot import ReceptiveField
    from visualize_lot import _edge_payload, _node_payload
    tree, spaces, nodes = g.tree_index(), g._spaces or {}, []
    for n in g.nodes:
        space = spaces.get(tree.get(n.node_id))
        mask = masks.get(n.node_id)
        field = None if space is None else ReceptiveField(space if mask is None else np.where(mask, space, 0.5))
        nodes.append({**_node_payload(n, field), "role": getattr(n, "role", "imagined")})
    return {"label": label, "graph_name": "program", "engine": True, "nodes": nodes, "result": 0.0,
            "edges": [_edge_payload(e) for e in g.solid_edges] + [_edge_payload(e) for e in g.dashed_edges],
            "roots": [r.node_id for r in g.roots]}


def program_graph(src: str, cfg_path: str) -> dict:
    """A program as a visualizer payload: its compiled graph, drawn once (tick 0, before any step)."""
    prog, issues, g = prepare(src, cfg_path)
    g._spaces, g._imagery = g._render_all()
    base = viewer_snapshot(g, "initial drawing", g.view_masks((GRID, GRID)))
    return {"version": 1, "name": str(prog), "base": base, "frames": [base], "icon_sprites": {},
            "program": str(prog), "issues": issues}


def record(src: str, ticks: int, states: dict, cfg_path: str) -> dict:
    prog, issues, g = prepare(src, cfg_path)
    units = {k: kind for k, (kind, _, _) in sorted(g.unknown_motion_units().items())}
    options = {"universal": ["a", "b"], "neutral": ["still", "a", "b"]}
    states = {k: states.get(k, options[kind][0]) for k, kind in units.items()}
    states = {k: v if v in options[units[k]] else options[units[k]][0] for k, v in states.items()}
    g.motion_states = dict(states)
    twin = compile_program(prog)
    twin.grid = (GRID, GRID)
    twin.reset_gates()
    twin.assign_roles()
    twin.motion_states = dict(states)

    pos, tree_of = forest_layout(g)
    parents = g._parents()
    nodes_out = [{"id": n.node_id, "type": n.node_type, "x": pos[n.node_id][0],
                  "label": n.node_type if isinstance(n, GroupNode) else node_label(n),
                  "y": pos[n.node_id][1], "tree": tree_of[n.node_id],
                  "dynamic": bool(isinstance(n, ObjectNode) and n.is_dynamic),
                  "parent": next((p.node_id for p in parents.get(n.node_id, []) if p.node_id in pos), None)}
                 for n in g.nodes if n.node_id in pos]
    dashed = [{"source": e.source.node_id, "target": e.target.node_id} for e in g.dashed_edges]
    movers = [n for n in g.nodes if isinstance(n, BodyNode) and not isinstance(n, PlanNode)]
    windows = [w.node_id for w in g.windows()]
    bugs, frames, snapshots = [], [], []
    for t in range(ticks):
        seen = g._imagery if g._imagery is not None else embodied_lot.blank_board((GRID, GRID))
        # the phases of ConceptGraph._step_gameboard, in its order
        g.assign_roles()
        g._scan()
        views = g.node_views(g.grid)
        g._perceive(views)
        perception = {r.node_id: round(float(r.activation), 3) for r in g._observer_roots}
        measures = {m.node_id: round(float(m.activation), 3) for m in getattr(g, "_measures", [])}
        masks = g.view_masks((GRID, GRID))
        prev_act = dict(g._prev_act)
        tokens = g._group_tokens() if embodied_lot.GROUPING_NODES else []
        before = {u.node_id: len(u.orders) for u in movers + tokens}
        g._issue_orders(views)
        if embodied_lot.GROUPING_NODES:
            g._drive_bodies(views)
        issued = [{"unit": u.node_id, **_order(o)} for u in movers + tokens
                  for o in u.orders[before.get(u.node_id, 0):]]
        drives = {u.node_id: [round(float(x), 3) for x in u.drive] for u in tokens}         # spent by the move
        onsets = [nid for nid, a in g._prev_act.items()
                  if prev_act.get(nid, 0.0) <= embodied_lot.ONSET_THRESHOLD < a]
        gates_before = {n.node_id: n.gate for n in g.nodes}
        g._propagate_delayed()
        gate_changes = [{"node": n.node_id, "from": round(gates_before[n.node_id], 3), "to": round(n.gate, 3)}
                        for n in g.nodes if n.node_id in pos and abs(n.gate - gates_before[n.node_id]) > 1e-6]
        offsets = {k: st["off"].copy() for k, st in g._mass.items()}
        g._move_bodies_gameboard(views)
        if embodied_lot.MOTION_KINDS:
            g._move_mass()
        mass = [{"tree": g.roots.index(g[k]) if k in g._nodes else k,
                 "cells": int(np.any(st["off"] != offsets.get(k, 0), axis=-1).sum())} for k, st in g._mass.items()]
        g._spaces, g._imagery = g._render_all()
        g._tick += 1
        twin.step()
        if not (np.array_equal(g._imagery, twin._imagery)
                and all(np.array_equal(g._spaces[k], twin._spaces[k]) for k in g._spaces)):
            bugs.append(f"tick {t}: the recorded board differs from ConceptGraph.step()")
        moved = g.view_masks((GRID, GRID))
        snapshots.append(viewer_snapshot(g, f"tick {t}", moved))
        shifts = []
        for n in movers:
            if n.node_id in masks and n.node_id in moved and masks[n.node_id].any() and moved[n.node_id].any():
                (r0, c0), (r1, c1) = (np.argwhere(m).min(axis=0) for m in (masks[n.node_id], moved[n.node_id]))
                if (r0, c0) != (r1, c1):
                    shifts.append({"unit": n.node_id, "rows": int(r1 - r0), "cols": int(c1 - c0)})
        frames.append({
            "t": t,
            "seen": np.round(seen, 3).tolist(),
            "drawn": np.round(g._imagery, 3).tolist(),
            "spaces": {k: np.round(v, 3).tolist() for k, v in g._spaces.items()},
            "activation": {n.node_id: round(float(n.activation), 3) for n in g.nodes if n.node_id in pos},
            "gate": {n.node_id: round(float(n.gate), 3) for n in g.nodes if n.node_id in pos},
            "role": {n.node_id: getattr(n, "role", "imagined") for n in g.nodes if n.node_id in pos},
            "observers": [{"id": nid, "value": v, **board_cells(nid, masks)} for nid, v in perception.items()],
            "measures": [{"id": nid, "value": v, **board_cells(nid, masks)} for nid, v in measures.items()],
            "units": [{"id": n.node_id, "kind": n.node_type, "cells": _cells(moved[n.node_id])}
                      for n in movers if n.node_id in moved],
            "tokens": [{"id": u.node_id, "kind": u.node.node_type, "drive": drives[u.node_id],
                        "orders": [_order(o) for o in u.orders]} for u in tokens],
            "shifts": shifts, "mass": [m for m in mass if m["cells"]],
            "issued": issued, "onsets": onsets, "gate_changes": gate_changes,
        })

    whole = adapter.rollout_whole(prog, ticks + 1, GRID, states)
    whole = whole if whole.ndim == 4 else whole[:, None]
    comp = adapter.rollout(prog, ticks + 1, GRID, states)
    comp = comp if comp.ndim == 4 else comp[:, None]
    if not np.array_equal(whole, comp):
        bugs.append("component rollout (adapter.rollout) differs from the whole-program rollout")
    combos = adapter.motion_combinations(prog, GRID)
    return {"program": str(prog), "issues": issues, "bugs": bugs, "grid": GRID, "nodes": nodes_out,
            "dashed": dashed, "roots": [r.node_id for r in g.roots], "windows": windows,
            "units": [{"key": k, "kind": kind, "options": options[kind], "state": states[k]} for k, kind in units.items()],
            "combinations": len(combos), "channels": np.round(whole, 3).tolist(), "frames": frames,
            "config": cfg_path, "prior_ok": not issues, "snapshots": snapshots}


def visualizer_page(cfg_path: str) -> str:
    """``visualizer/page.html`` (via ``visualize_lot``) on the first example, told that the engine is here."""
    from visualize_lot import _html_template
    with _lock:
        payload = program_graph(EXAMPLES[0][1], cfg_path)
    payload["engine"] = {"examples": [{"title": t, "program": p} for t, p in EXAMPLES], "config": cfg_path}
    return _html_template("Embodied-LoT v7 visualizer", payload)


def page(cfg_path: str) -> str:
    html = Path(__file__).with_name("playground_page.html").read_text(encoding="utf-8")
    data = json.dumps({"examples": [{"title": t, "program": p} for t, p in EXAMPLES], "config": cfg_path})
    return html.replace("__BOOT__", data.replace("</", r"<\/"))


def make_handler(cfg_path: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, visualizer_page(cfg_path).encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/gameboard":
                self._send(200, page(cfg_path).encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/api/run", "/api/graph"):
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                ticks = max(1, min(int(req.get("ticks", 24)), 80))
                with _lock:
                    out = (program_graph(str(req.get("program", "")), cfg_path) if self.path == "/api/graph" else
                           record(str(req.get("program", "")), ticks, dict(req.get("states") or {}), cfg_path))
                self._send(200, json.dumps(out, separators=(",", ":")).encode("utf-8"), "application/json")
            except Exception as e:                       # a parse error or an engine error: show it on the page
                import traceback
                body = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=6)}
                self._send(200, json.dumps(body).encode("utf-8"), "application/json")

        def log_message(self, *args):
            pass
    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.config))
    print(f"v7 playground on http://localhost:{args.port}  (semantics from {args.config})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
