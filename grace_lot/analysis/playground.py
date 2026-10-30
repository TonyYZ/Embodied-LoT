"""A playground for the v7 language: type a program, step it on the gameboard, look for bugs.

    python -m grace_lot.analysis.playground [--config configs/grace_v7_parts.toml] [--port 8765] [--lot solid_figure=false]
    python -m grace_lot.analysis.playground --offline playground_offline.html     # a self-contained copy (no server)

then open http://localhost:8765 for the board-first page (also at /gameboard), or /graph for the graph visualizer
(``visualizer/page.html``: edit a graph, load or export v7 program text, run it on the engine and replay the ticks).
The pages send programs to this server, which runs the real engine
(``embodied_lot``), with the one switch the page or ``--lot`` sets (``SEMANTICS``: built-in solidity), and records
every tick phase by phase, as
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
from embodied_lot import BodyNode, GroupNode, ObjectNode, PlanNode, SerialNode
from grace_lot.config import load_config
from grace_lot.inference.grammar import Grammar
from grace_lot.lot_adapter import adapter
from grace_lot.lot_adapter.program import (apply_semantics, as_program, compile_program, depth, edges_valid,
                                           is_valid, motion_units, nodes, parse_program, size, windows_as_content)

GRID = 12
_lock = Lock()                       # the engine's semantics are module flags: one run at a time
# The switch a page can set (a lot config key): embodied_lot.SOLID_FIGURE. The language is otherwise built in.
SEMANTICS = ("solid_figure",)
CURRENT = {"solid_figure": True}     # what the examples and a fresh page run with
NUMERIC = {"subcell": 1}             # numeric switches a request may set: embodied_lot.SUBCELL (1: whole cells)

EXAMPLES = [
    ("One body glides, clearing its path through unexplored stuff",
     "(parallel (serial_h (body center/V bottom/H*) neutral/H neutral/H) out)"),
    ("Two bodies in file",
     "(parallel (serial_h (body center/V bottom/H*) (body center/V bottom/H*) neutral/H) out)"),
    ("A field does not drive itself (its motion carries its stuff out)",
     "(parallel (serial_v (field center/V bottom/V*) neutral/H) out)"),
    ("A body drives itself",
     "(parallel (serial_v (body center/V bottom/V*) neutral/H) out)"),
    ("Template copies, each its own instance (each copy's stuff carried up)",
     "(parallel (serial_v (template periphery/H (field center/V)) neutral/H) (serial_h bottom/V* neutral/V neutral/V) out)"),
    ("Unknown direction (universal)",
     "(parallel center/V universal/V* out)"),
    ("Unknown whether it moves (neutral)",
     "(parallel center/H neutral/V* out)"),
    ("Known still (void)",
     "(parallel center/V void/V* out)"),
    ("A tree gating another tree",
     "(parallel center/H bottom/H*) (parallel center/V out) | .0>.1.0"),
    ("Measure tree",
     "(serial_h universal/H neutral/H neutral/H) (parallel center/H out) | .0>.1.0"),
    ("Self-observing oscillator: each half sends the body it sees beyond itself, into the other half",
     "(parallel (serial_h (parallel bottom/H* center/H) (parallel top/H* center/H (body center/H))) out) | "
     ".0.0.1>.0.0.0 .0.1.1>.0.1.0"),
    ("a basket by superposition (surrounded, with a bottom)",
     "(parallel periphery/H bottom/V out)"),
    ("a basket by hollowing a block",
     "(parallel universal/H (template center/H upper/V void/H) out)"),
    ("a ball in a sand basket pinned in place (void*): the sand holds, the ball stays inside",
     "(parallel periphery/H bottom/V void/V* (template center/H center/V (serial_v (serial_h (body universal/H "
     "universal/H* universal/V*) void/V) void/H)) out)"),
    ("a ball in a loose sand basket pushes the sand off the board (nothing pins it; the edge is open)",
     "(parallel periphery/H bottom/V (template center/H center/V (serial_v (serial_h (body universal/H "
     "universal/H* universal/V*) void/V) void/H)) out)"),
    ("a ball in a basket of pinned fields (each wall a field with void*) stays inside",
     "(parallel (serial_h (field universal/H void/H*) void/V void/V void/V void/V (field universal/H void/H*)) "
     "(serial_v (field universal/H void/H*) void/H void/H void/H void/H void/H) (template center/H center/V "
     "(serial_v (serial_h (body universal/H universal/H* universal/V*) void/V) void/H)) out)"),
    ("a ball beside a field basket pushes through a wall and leaves; the wall heals behind (nothing pins it)",
     "(parallel (field periphery/H bottom/V) (template center/H center/V (serial_v (serial_h (body universal/H"
     " universal/H* universal/V*) void/V) void/H)) out)"),
    ("a ball pushes into a free-standing wall (a field): the mass piles up ahead, the wall closes behind",
     "(parallel (serial_h (serial_v void/H (body universal/H bottom/H*) void/H) void/V (field universal/H) void/V "
     "void/V void/V) out)"),
    ("a pinned rock (a field claiming no motion, void*) stops the ball",
     "(parallel (serial_h (serial_v void/H (body universal/H bottom/H*) void/H) void/V (field universal/H void/H*) "
     "void/V void/V void/V) out)"),
    ("neutral* un-knows the pin in a band: the ball pushes in, then stops there (the field's unknown motion outranks its own); the rest of the rock holds",
     "(parallel (serial_h (serial_v void/H (body universal/H bottom/H*) void/H) void/V (field universal/H void/H* "
     "(template center/V neutral/H*)) void/V void/V void/V) out)"),
    ("a tar pit (an empty void* region): the ball rolls in and is held",
     "(parallel (serial_h (serial_v void/H (body universal/H bottom/H*) void/H) void/V (field void/H*) void/V void/V "
     "void/V) out)"),
    ("a plant (a source field under upward motion)",
     "(parallel (template bottom/V center/H (field universal/H)) bottom/V* out)"),
    ("a spawner (a field holding a rising body)",
     "(parallel (template bottom/V center/H (field (body center/H bottom/V*))) out)"),
    ("a drain (a void field swallows a rising body)",
     "(parallel (template top/V center/H (field void/H)) (template bottom/V center/H (body universal/H "
     "bottom/V*)) out)"),
    ("a fountain (a field with motion inside)",
     "(parallel (template bottom/V center/H (field universal/H bottom/V*)) void/V out)"),
    ("a rigid box (a body carrying a field), pushed by the ball inside, off the board",
     "(parallel (serial_h void/V (body (field periphery/H bottom/V) (template center/H center/V (body "
     "universal/H universal/H*))) void/V) out)"),
    ("contraction cannot squeeze a solid block",
     "(parallel (template center/H center/V universal/H) periphery/V* out)"),
    ("a body carrying a field with motion drives itself",
     "(parallel (serial_h void/V (body universal/H (field bottom/H*)) void/V) out)"),
    ("a bubble machine pushes out the bodies it holds",
     "(parallel (serial_h void/V (serial_v void/H (field top/H* (body universal/H)) void/H) void/V) out)"),
    ("a conveyor carries a cart all the way, and off the board",
     "(parallel (field bottom/H*) (serial_h void/V (body universal/H) void/V void/V) out)"),
    ("a ball crosses a roomy box, then moves it only by pushing its wall, all the way off the board",
     "(parallel (serial_h (body (field periphery/H bottom/V) (template center/H center/V (body universal/H "
     "bottom/H*))) void/V) out)"),
    ("a body riding in a moving body is carried, not driven twice",
     "(parallel (serial_h void/V (body bottom/V bottom/H* (template center/V center/H (body universal/H))) "
     "void/V void/V) out)"),
    ("a pusher drives a row of blocks against a pinned wall, where they stop side by side",
     "(parallel (serial_h (body universal/H bottom/H*) (body universal/H) neutral/V (body universal/H) "
     "neutral/V (field universal/V void/V*)) out)"),
    ("a ball pushes a sand pile across open ground and off the board",
     "(parallel (serial_h void/V (body universal/H bottom/H*) (serial_v void/H universal/H void/H) void/V "
     "void/V void/V) out)"),
    ("conveyed stuff pushes resting stuff; none is squeezed, and what reaches the edge leaves the board",
     "(parallel (serial_h (field bottom/H*) void/V) (serial_h universal/V void/V universal/V) out)"),
    ("through fog (a field) the path closes behind",
     "(parallel (serial_h (body center/V bottom/H*) (field neutral/H) neutral/H) out)"),
    ("a ball in a closed box (walls drawn as outlines) pushes through a wall and leaves; the wall heals behind",
     "(parallel (field periphery/H bottom/V top/V) (template center/H center/V (body universal/H universal/H* "
     "universal/V*)) out)"),
    ("solidity stated by the program (built-in solidity off): once no gap is left, a push apart cancels the push in "
     "exactly; they stop touching",
     "(parallel (serial_h (body universal/H bottom/H*) void/V void/V (body universal/H top/H*)) "
     "(template center/H universal/H) (template center/H center/H*) out) | .1.1>.2.1",
     {**CURRENT, "solid_figure": False}),
    ("no solidity at all (built-in solidity off, no rule): the bodies run into each other",
     "(parallel (serial_h (body universal/H bottom/H*) void/V void/V (body universal/H top/H*)) out)",
     {**CURRENT, "solid_figure": False}),
    ("periphery motion gathers two strips at the centre",
     "(parallel (serial_v universal/H void/H void/H void/H void/H universal/H) periphery/V* out)"),
    ("stated solidity, and parting again: the right body turns back once they touch; the push apart never moved "
     "anyone, so it lapses, and the left body follows",
     "(parallel (serial_h (body universal/H bottom/H*) void/V void/V (body universal/H top/H* bottom/H*)) "
     "(template center/H universal/H) (template center/H center/H*) (template center/H void/H) out) | "
     ".1.1>.2.1 .1.1>.0.3.2 .3.1>.0.3.1",
     {**CURRENT, "solid_figure": False}),
]



def node_label(n) -> str:
    if isinstance(n, ObjectNode):
        return f"{n.name}/{'V' if n.orientation.value == 1 else 'H'}{'*' if n.is_dynamic else ''}"
    if isinstance(n, SerialNode):
        return "serial_" + ("V" if n.split_orientation.value == 1 else "H")
    return n.node_type


def tree_layout(g, root):
    """Tidy layered layout: leaves in order along x, parents centred over their children."""
    pos, xs = {}, [0]

    def visit(n, d):
        kids = [c for c in n.children if c.node_type not in ("return", "input")]
        if not kids:
            pos[n.node_id] = (xs[0], d)
            xs[0] += 1
        else:
            for c in kids:
                visit(c, d + 1)
            pos[n.node_id] = (sum(pos[c.node_id][0] for c in kids) / len(kids), d)
    visit(root, 0)
    return pos


def board_cells(nid: str, masks: dict) -> dict:
    """A node's board cells (they may be turned) and their bounding box."""
    mask = masks.get(nid, np.zeros((GRID, GRID), dtype=bool))
    rows, cols = np.nonzero(mask)
    box = [int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1] if rows.size else [0, 0, 0, 0]
    return {"region": box, "cells": [[int(r), int(c)] for r, c in zip(rows, cols)]}


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
    if not windows_as_content(t):
        out.append("an out/in window used as a template (windows may only be content)")
    if g.max_motion_units is not None and motion_units(prog) > g.max_motion_units:
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


def prepare(src: str, cfg_path: str, lot: dict = None) -> tuple:
    """The program under the config (``lot``: the switches to set, ``SEMANTICS``), why it is outside the search
    grammar (if it is), a fresh graph."""
    cfg = load_config(cfg_path)
    cfg["lot"].update({k: bool(v) for k, v in (lot or {}).items() if k in SEMANTICS})
    cfg["lot"].update({k: int(v) for k, v in (lot or {}).items() if k in NUMERIC})
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
    """A substance grid as a board: each cell's claimed value, by its strength (unknown where none)."""
    return 0.5 + (S["v"] - 0.5) * S["s"]


def record(src: str, ticks: int, states: dict, cfg_path: str, lot: dict = None) -> dict:
    prog, issues, g = prepare(src, cfg_path, lot)
    units = {k: kind for k, (kind, _, _) in sorted(g.unknown_motion_units().items())}
    options = {"universal": ["a", "b"]}
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
    motion_leaves = [n for n in g.nodes if isinstance(n, ObjectNode) and n.is_dynamic and n.node_id in pos]
    tree_index = g.tree_index()

    def motion_regions(masks, act):              # where each moving leaf acts on the board, and whether it is on
        return [{"id": n.node_id, "label": node_label(n), "tree": tree_index.get(n.node_id),
                 "on": act.get(n.node_id, 0.0) > embodied_lot.ONSET_THRESHOLD, "cells": _cells(masks[n.node_id])}
                for n in motion_leaves if n.node_id in masks and masks[n.node_id].any()]
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
        "motion": motion_regions(masks, {}),
        "tokens": [{"id": u.node_id, "kind": u.node.node_type, "orders": []} for u in g._group_tokens()],
        "shifts": [], "onsets": [], "gate_changes": [], "acts": [], "stuff": [],
        "substance": [], "resets": [], "spawned": [],
        "grids": {k: np.round(substance_board(S), 3).tolist() for k, S in getattr(g, "_sub", {}).items()},
    })
    for t in range(ticks):
        seen = g._imagery if g._imagery is not None else embodied_lot.blank_board((GRID, GRID))
        # the phases of ConceptGraph._step_gameboard, in its order
        g.assign_roles()
        g._scan()
        views = g.node_views(g.grid)
        g._perceive(views)
        g._gate_motion_now()
        perception = {r.node_id: round(float(r.activation), 3) for r in g._observer_roots}
        measures = {m.node_id: round(float(m.activation), 3) for m in getattr(g, "_measures", [])}
        masks = g.view_masks((GRID, GRID))
        prev_act = dict(g._prev_act)
        tokens = g._group_tokens()
        sub_before = {k: len(S["orders"]) for k, S in getattr(g, "_sub", {}).items()}
        g._issue_orders(views)
        acts = [{"leaf": key.split("#")[0], "bodies": sorted(bodies)}        # what each leaf on acts on this tick
                for key, bodies in sorted(g._inside.items())
                if "#" in key and g._prev_act.get(key.split("#")[0], 0.0) > embodied_lot.ONSET_THRESHOLD]
        stuff = [{"tree": g.roots.index(g[k]), "cells": int(sum(o[0].sum() for o in S["orders"][sub_before.get(k, 0):]))}
                 for k, S in getattr(g, "_sub", {}).items()]
        onsets = [nid for nid, a in g._prev_act.items()
                  if prev_act.get(nid, 0.0) <= embodied_lot.ONSET_THRESHOLD < a]
        gates_before = {n.node_id: n.gate for n in g.nodes}
        g._propagate_delayed()
        gate_changes = [{"node": n.node_id, "from": round(gates_before[n.node_id], 3), "to": round(n.gate, 3)}
                        for n in g.nodes if n.node_id in pos and abs(n.gate - gates_before[n.node_id]) > 1e-6]
        flow_from = g._flow_begin()
        g._move_bodies_gameboard(views)
        resets = []                                  # move substance, re-assert fields, spawn held bodies
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
        g._flow_end(flow_from)
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
            "units": [{"id": n.node_id, "kind": n.node_type, "cells": _cells(moved[n.node_id])}
                      for n in movers if n.node_id in moved],
            "motion": motion_regions(masks, g._prev_act),
            "tokens": [{"id": u.node_id, "kind": u.node.node_type, "orders": [_order(o) for o in u.orders]}
                       for u in tokens],
            "shifts": shifts, "onsets": onsets, "gate_changes": gate_changes,
            "acts": acts, "stuff": [m for m in stuff if m["cells"]],
            "substance": [m for m in substance if m["cells"]], "resets": [m for m in resets if m["cells"]],
            "spawned": spawned,
            "grids": {k: np.round(substance_board(S), 3).tolist() for k, S in getattr(g, "_sub", {}).items()},
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
            "semantics": {k: bool(getattr(embodied_lot, k.upper())) for k in SEMANTICS},
            "numeric": {k: int(getattr(embodied_lot, k.upper())) for k in NUMERIC}}


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
            for f in d["frames"]:
                f.pop("grids", None)                     # the mass grids: not shown on the page
                if len(d["roots"]) == 1:                 # one tree: its space is the drawn board (shown only for several)
                    f["spaces"] = {}
            runs.append({"program": e["program"], "lot": lot, "states": states, "run": d})
    html = page(cfg_path).replace(
        "Type a program and step it on the 12×12 gameboard. The server runs the real engine",
        "Pick an example and step it on the 12×12 gameboard. This offline copy holds runs of the real engine")
    def coarse(x):                               # two decimals are plenty to look at, and halve the page
        if isinstance(x, float):
            return round(x, 2)
        if isinstance(x, list):
            return [coarse(v) for v in x]
        if isinstance(x, dict):
            return {k: coarse(v) for k, v in x.items()}
        return x
    data = json.dumps(coarse({"ticks": ticks, "runs": runs}), separators=(",", ":"))
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
            if self.path in ("/", "/index.html", "/gameboard"):
                self._send(200, page(cfg_path, lot).encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/graph":
                self._send(200, visualizer_page(cfg_path, lot).encode("utf-8"), "text/html; charset=utf-8")
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
    ap.add_argument("--lot", nargs="*", default=[], help="switches, e.g. solid_figure=false subcell=3")
    ap.add_argument("--offline", help="write a self-contained page with the examples built in to this file, and exit")
    args = ap.parse_args()
    if args.offline:
        Path(args.offline).write_text(offline_page(args.config), encoding="utf-8")
        print(f"wrote {args.offline}", flush=True)
        return
    lot = {k: int(v) if k in NUMERIC else v.lower() == "true" for k, v in (item.split("=", 1) for item in args.lot)}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.config, lot))
    print(f"v7 playground on http://localhost:{args.port}  (graph editor: /graph; semantics from {args.config}"
          + (f", with {', '.join(f'{k}={v}' for k, v in lot.items())}" if lot else "") + ")", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
