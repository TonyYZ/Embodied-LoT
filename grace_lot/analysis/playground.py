"""A playground for the v7 language: type a program, step it on the gameboard, look for bugs.

    python -m grace_lot.analysis.playground [--config configs/grace_v7_parts.toml] [--port 8765]
                                            [--lot pure_claims=true solid_figure=true soft_edges=true static_fields=true]
    python -m grace_lot.analysis.playground --offline playground_offline.html     # a self-contained copy (no server)

then open http://localhost:8765 for the graph visualizer (``visualizer/page.html``: edit a graph, load or
export v7 program text, run it on the engine and replay the ticks), or /gameboard for the board-first page.
The pages send programs to this server, which runs the real engine
(``embodied_lot``) under the config's semantics (with the language flags the page or ``--lot`` switches on:
``SEMANTICS``) and records every tick phase by phase, as
``ConceptGraph._step_gameboard`` runs them. Each run also checks itself: the recorded boards must equal a
twin graph stepped with ``step()``, and the component rollouts (``adapter.rollout``) must equal whole-program
rollouts; any difference is reported on the page as a bug.
"""

from __future__ import annotations

import argparse
import itertools
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
# The language flags a page can switch (lot config keys): see embodied_lot.PURE_CLAIMS, SOLID_FIGURE, SOFT_EDGES,
# STATIC_FIELDS, STANDING_MOTION. Static fields need pure claims; standing motion needs static fields.
SEMANTICS = ("pure_claims", "solid_figure", "soft_edges", "static_fields", "standing_motion")
CURRENT = {k: True for k in SEMANTICS}        # the language the examples and a fresh page run in: keep it current

EXAMPLES = [
    ("One body glides, clearing its path through unexplored stuff", "(parallel (serial_h (body (parallel center/V bottom/H*)) neutral/H neutral/H) out)"),
    ("Two bodies in file",
     "(parallel (serial_h (body (parallel center/V bottom/H*)) (body (parallel center/V bottom/H*)) neutral/H) out)"),
    ("A field does not drive itself (its motion carries its stuff out)",
     "(parallel (serial_v (field (parallel center/V bottom/V*)) out) out)"),
    ("A body drives itself", "(parallel (serial_v (body (parallel center/V bottom/V*)) out) out)"),
    ("Template copies, each its own instance (each copy's stuff carried up)",
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
# More examples: (title, program[, flags]); without flags, CURRENT.
EXAMPLES += [
    ("a basket by superposition (surrounded, with a bottom)", "(parallel periphery/H bottom/V out)"),
    ("a basket by hollowing a block", "(parallel universal/H (template center/H upper/V void/H) out)"),
    ("a ball in a sand basket (packed to the board's edge, the sand holds)",
     "(parallel periphery/H bottom/V (template center/H center/V (body (parallel universal/H universal/H* universal/V*))) out)"),
    ("a ball beside a field basket (the walls hold)",
     "(parallel (field (parallel periphery/H bottom/V)) (template center/H center/V (body (parallel universal/H universal/H* universal/V*))) out)"),
    ("a plant (a source field under upward motion)",
     "(parallel (template bottom/V center/H (field universal/H)) bottom/V* out)"),
    ("a spawner (a field holding a rising body)",
     "(parallel (template bottom/V center/H (field (body (parallel center/H bottom/V*)))) out)"),
    ("a drain (a void field swallows a rising body)",
     "(parallel (template top/V center/H (field void/H)) (template bottom/V center/H (body (parallel universal/H bottom/V*))) out)"),
    ("a fountain (a field with motion inside)",
     "(parallel (template bottom/V center/H (field (parallel universal/H bottom/V*))) void/V out)"),
    ("a rigid box (a body carrying a field), pushed by the ball inside",
     "(parallel (serial_h void/V (body (parallel (field (parallel periphery/H bottom/V)) (template center/H center/V "
     "(body (parallel universal/H universal/H*))))) void/V) out)"),
    ("contraction cannot squeeze a solid block",
     "(parallel (template center/H center/V universal/H) periphery/V* out)"),
    ("a body carrying a field with motion drives itself",
     "(parallel (serial_h void/V (body universal/H (field bottom/H*)) void/V) out)"),
    ("a bubble machine pushes out the bodies it holds",
     "(parallel (serial_h void/V (serial_v void/H (field (parallel top/H* (body universal/H))) void/H) void/V) out)"),
    ("a conveyor carries a cart all the way",
     "(parallel (field bottom/H*) (serial_h void/V (body universal/H) void/V void/V) out)"),
    ("a ball moves a hollow box only by pushing its walls",
     "(parallel (serial_h void/V (body (parallel (field (parallel periphery/H bottom/V)) (template center/H center/V "
     "(body (parallel universal/H bottom/H*))))) void/V) out)"),
    ("a body riding in a moving body is driven as well as carried",
     "(parallel (serial_h void/V (body (parallel bottom/V bottom/H* (template center/V center/H (body universal/H)))) "
     "void/V void/V) out)"),
    ("a pusher drives a row of blocks against the edge, where they pack",
     "(parallel (serial_h (body (parallel universal/H bottom/H*)) (body universal/H) neutral/V (body universal/H) "
     "neutral/V neutral/V) out)"),
    ("a ball pushes a sand pile across open ground",
     "(parallel (serial_h void/V (body (parallel universal/H bottom/H*)) (serial_v void/H universal/H void/H) void/V void/V "
     "void/V) out)"),
    ("conveyed stuff pushes resting stuff; none is lost",
     "(parallel (serial_h (field bottom/H*) void/V) (serial_h universal/V void/V universal/V) out)"),
    ("through fog (a field) the path closes behind",
     "(parallel (serial_h (body (parallel center/V bottom/H*)) (field neutral/H) neutral/H) out)"),
    ("a ball in fixed walls moves as small as it must be",
     "(parallel (field (parallel bottom/V periphery/H)) (template center/H (body (parallel center/V universal/H* "
     "universal/V*))) out)"),
    ("periphery motion gathers two strips at the centre",
     "(parallel (serial_v universal/H void/H void/H void/H void/H universal/H) periphery/V* out)"),
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


def outline(g, node, masks: dict):
    """A body's outline on the board: its region, but for an instance whose moves narrowed its possible extent
    (``GroupToken.placements``), only the cells those placements cover."""
    m = masks.get(node.node_id)
    narrowed = [t for t in getattr(node, "instances", {}).values() if getattr(t, "placements", None) is not None]
    if m is None or not narrowed:
        return m
    at, out = g._moving_bodies(g.node_views(g.grid)), m.copy()
    for t in narrowed:
        cells = at.get(t.node_id, (None, None, None))[2]
        if cells is None or not cells.any():
            continue
        out &= ~cells
        rows, cols = np.nonzero(cells)
        r0, c0 = rows.min(), cols.min()
        places = embodied_lot._placements(rows.max() + 1 - r0, cols.max() + 1 - c0)
        for k in t.placements:
            hs, ws, dy, dx, _ = places[k]
            out[r0 + dy:r0 + dy + hs, c0 + dx:c0 + dx + ws] = True
    return out


def _order(o) -> dict:
    return {"axis": "vertical" if o[0] == 0 else "horizontal", "direction": int(np.sign(o[1])),
            "cells": round(float(o[2]), 2), "rate": round(float(o[3]), 3), "credit": round(float(o[4]), 3),
            "leaf": o[5]}


def prepare(src: str, cfg_path: str, lot: dict = None) -> tuple:
    """The program under the config's semantics (``lot``: the language flags to switch, ``SEMANTICS``), why it is
    outside the language (if it is), a fresh graph."""
    cfg = load_config(cfg_path)
    cfg["lot"].update({k: bool(v) for k, v in (lot or {}).items() if k in SEMANTICS})
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


def program_graph(src: str, cfg_path: str, lot: dict = None) -> dict:
    """A program as a visualizer payload: its compiled graph, drawn once (tick 0, before any step)."""
    prog, issues, g = prepare(src, cfg_path, lot)
    g._spaces, g._imagery = g._render_all()
    base = viewer_snapshot(g, "initial drawing", g.view_masks((GRID, GRID)))
    return {"version": 1, "name": str(prog), "base": base, "frames": [base], "icon_sprites": {},
            "program": str(prog), "issues": issues}


def substance_board(S: dict) -> np.ndarray:
    """STATIC_FIELDS: a substance grid as a board: each cell's claimed value, by its strength (unknown where none)."""
    return 0.5 + (S["v"] - 0.5) * S["s"]


def record(src: str, ticks: int, states: dict, cfg_path: str, lot: dict = None) -> dict:
    prog, issues, g = prepare(src, cfg_path, lot)
    static = embodied_lot.STATIC_FIELDS and embodied_lot.PURE_CLAIMS
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
    g.assign_roles()                             # tick 0: the board as the program draws it, before any tick runs
    spaces, drawn = g._render_all()              # (as the window rollout reads its first frame)
    masks = g.view_masks((GRID, GRID))
    snapshots.append(viewer_snapshot(g, "start", masks))
    frames.append({
        "t": 0, "start": True,
        "seen": np.round(embodied_lot.blank_board((GRID, GRID)), 3).tolist(),
        "drawn": np.round(drawn, 3).tolist(),
        "spaces": {k: np.round(v, 3).tolist() for k, v in spaces.items()},
        "activation": {n.node_id: round(float(n.activation), 3) for n in g.nodes if n.node_id in pos},
        "gate": {n.node_id: round(float(n.gate), 3) for n in g.nodes if n.node_id in pos},
        "role": {n.node_id: getattr(n, "role", "imagined") for n in g.nodes if n.node_id in pos},
        "observers": [], "measures": [],
        "units": [{"id": n.node_id, "kind": n.node_type, "cells": _cells(masks[n.node_id])} for n in movers if n.node_id in masks],
        "tokens": [{"id": u.node_id, "kind": u.node.node_type, "drive": [0.0, 0.0], "orders": []}
                   for u in (g._group_tokens() if embodied_lot.GROUPING_NODES else [])],
        "shifts": [], "mass": [], "issued": [], "onsets": [], "gate_changes": [], "acts": [], "stuff": [],
        "substance": [], "resets": [], "spawned": [],
        "grids": {k: np.round(substance_board(S), 3).tolist() for k, S in getattr(g, "_sub", {}).items()} if static else {},
    })
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
        sub_before = {k: len(S["orders"]) for k, S in getattr(g, "_sub", {}).items()}
        g._issue_orders(views)
        acts = []                                    # standing motion: what each switched-on leaf acts on this tick
        if embodied_lot._standing():
            for key, bodies in sorted(g._inside.items()):
                if "#" in key and not key.endswith("m") and g._prev_act.get(key.split("#")[0], 0.0) > embodied_lot.ONSET_THRESHOLD:
                    acts.append({"leaf": key.split("#")[0], "bodies": sorted(bodies)})
        stuff = [{"tree": g.roots.index(g[k]), "cells": int(sum(o[0].sum() for o in S["orders"][sub_before.get(k, 0):]))}
                 for k, S in getattr(g, "_sub", {}).items()]
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
        substance, spawned, resets = [], [], []
        if static:                                   # move substance, re-assert fields, spawn held bodies
            before_sub = {k: substance_board(S) for k, S in getattr(g, "_sub", {}).items()}
            for rid in list(getattr(g, "_sub", {})):
                g._move_substance(rid)
            moved_sub = {k: substance_board(S) for k, S in g._sub.items()}
            for rid in list(g._sub):
                g._reset_fields(rid)
                resets.append({"tree": g.roots.index(g[rid]), "cells": int(g._sub[rid]["F"].sum())})
            count = lambda: {n.node_id: len(n.instances) for n in g._spawners()}
            was = count()
            g._spawn_bodies()
            spawned = [{"unit": k, "instances": v, "new": v - was.get(k, 0)} for k, v in count().items()]
            substance = [{"tree": g.roots.index(g[k]), "cells": int((np.abs(moved_sub[k] - b) > 1e-9).sum())}
                         for k, b in before_sub.items()]
        elif embodied_lot.MOTION_KINDS:
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
        snapshots.append(viewer_snapshot(g, f"tick {t + 1}", moved))
        shifts = []
        for n in movers:
            if n.node_id in masks and n.node_id in moved and masks[n.node_id].any() and moved[n.node_id].any():
                (r0, c0), (r1, c1) = (np.argwhere(m).min(axis=0) for m in (masks[n.node_id], moved[n.node_id]))
                if (r0, c0) != (r1, c1):
                    shifts.append({"unit": n.node_id, "rows": int(r1 - r0), "cols": int(c1 - c0)})
        frames.append({
            "t": t + 1,
            "seen": np.round(seen, 3).tolist(),
            "drawn": np.round(g._imagery, 3).tolist(),
            "spaces": {k: np.round(v, 3).tolist() for k, v in g._spaces.items()},
            "activation": {n.node_id: round(float(n.activation), 3) for n in g.nodes if n.node_id in pos},
            "gate": {n.node_id: round(float(n.gate), 3) for n in g.nodes if n.node_id in pos},
            "role": {n.node_id: getattr(n, "role", "imagined") for n in g.nodes if n.node_id in pos},
            "observers": [{"id": nid, "value": v, **board_cells(nid, masks)} for nid, v in perception.items()],
            "measures": [{"id": nid, "value": v, **board_cells(nid, masks)} for nid, v in measures.items()],
            "units": [{"id": n.node_id, "kind": n.node_type, "cells": _cells(outline(g, n, moved))}
                      for n in movers if n.node_id in moved],
            "tokens": [{"id": u.node_id, "kind": u.node.node_type, "drive": drives[u.node_id],
                        "orders": [_order(o) for o in u.orders]} for u in tokens],
            "shifts": shifts, "mass": [m for m in mass if m["cells"]],
            "issued": issued, "onsets": onsets, "gate_changes": gate_changes,
            "acts": acts, "stuff": [m for m in stuff if m["cells"]],
            "substance": [m for m in substance if m["cells"]], "resets": [m for m in resets if m["cells"]],
            "spawned": spawned,
            "grids": {k: np.round(substance_board(S), 3).tolist() for k, S in getattr(g, "_sub", {}).items()}
            if static else {},
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
            "config": cfg_path, "prior_ok": not issues, "snapshots": snapshots,
            "semantics": {k: bool(getattr(embodied_lot, k.upper())) for k in SEMANTICS}}


def _examples() -> list:
    return [{"title": e[0], "program": e[1], "lot": e[2] if len(e) > 2 else CURRENT} for e in EXAMPLES]


def offline_page(cfg_path: str, ticks: int = 24) -> str:
    """The board-first page with every example's runs built in (each joint state of its unknown motion, ``ticks``
    ticks, under its own language switches), so it plays without the server: a phone, an e-mail attachment. Programs
    and switches other than the examples' cannot run offline; the page says so."""
    runs = []
    for e in _examples():
        lot = {k: bool(e["lot"].get(k, False)) for k in SEMANTICS}
        first = record(e["program"], ticks, {}, cfg_path, lot)
        options = [dict(zip([u["key"] for u in first["units"]], c))
                   for c in itertools.product(*[u["options"] for u in first["units"]])]
        for states in options:
            d = first if states == {u["key"]: u["state"] for u in first["units"]} else record(e["program"], ticks, states, cfg_path, lot)
            d.pop("snapshots", None)                     # the graph viewer's, not shown on this page
            runs.append({"program": e["program"], "lot": lot, "states": states, "run": d})
    html = page(cfg_path).replace(
        "Type a program and step it on the 12×12 gameboard. The server runs the real engine",
        "Pick an example and step it on the 12×12 gameboard. This offline copy holds runs of the real engine")
    data = json.dumps({"ticks": ticks, "runs": runs}, separators=(",", ":"))
    return html.replace("const OFFLINE = null;", "const OFFLINE = " + data.replace("</", r"<\/") + ";")


def visualizer_page(cfg_path: str, lot: dict = None) -> str:
    """``visualizer/page.html`` (via ``visualize_lot``) on the first example, told that the engine is here."""
    from visualize_lot import _html_template
    with _lock:
        payload = program_graph(EXAMPLES[0][1], cfg_path, lot)
    payload["engine"] = {"examples": [{"title": t, "program": p} for t, p, *_ in EXAMPLES], "config": cfg_path}
    return _html_template("Embodied-LoT v7 visualizer", payload)


def page(cfg_path: str, lot: dict = None) -> str:
    html = Path(__file__).with_name("playground_page.html").read_text(encoding="utf-8")
    data = json.dumps({"examples": _examples(), "config": cfg_path, "semantics": list(SEMANTICS),
                       "lot": {k: bool((lot if lot else CURRENT).get(k, False)) for k in SEMANTICS}})
    return html.replace("__BOOT__", data.replace("</", r"<\/"))


def make_handler(cfg_path: str, lot: dict = None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, visualizer_page(cfg_path, lot).encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/gameboard":
                self._send(200, page(cfg_path, lot).encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path not in ("/api/run", "/api/graph"):
                return self._send(404, b"not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                ticks = max(1, min(int(req.get("ticks", 24)), 80))
                flags = {**(lot or {}), **dict(req.get("lot") or {})}           # the page's switches win
                with _lock:
                    out = (program_graph(str(req.get("program", "")), cfg_path, flags) if self.path == "/api/graph"
                           else record(str(req.get("program", "")), ticks, dict(req.get("states") or {}), cfg_path,
                                       flags))
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
    ap.add_argument("--lot", nargs="*", default=[], help="language flags on by default, e.g. static_fields=true")
    ap.add_argument("--offline", help="write a self-contained page with the examples built in to this file, and exit")
    args = ap.parse_args()
    if args.offline:
        Path(args.offline).write_text(offline_page(args.config), encoding="utf-8")
        print(f"wrote {args.offline}", flush=True)
        return
    lot = {k: v.lower() == "true" for k, v in (item.split("=", 1) for item in args.lot)}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.config, lot))
    print(f"v7 playground on http://localhost:{args.port}  (semantics from {args.config}"
          + (f", with {', '.join(k for k, v in lot.items() if v)}" if any(lot.values()) else "") + ")", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
