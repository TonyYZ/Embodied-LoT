"""Record ludo-dynamics traces from the real engine, for the interactive explainer page.

    python -m grace_lot.analysis.ludo_traces --out results/ludo_viz

writes ``traces.json`` and ``ludo.html`` (the traces embedded in ``ludo_page.html``).

Each example program is compiled and stepped with ``dynamics = "ludo"``. Every tick runs
the same phases as ``ConceptGraph._step_ludo``, so the page can show what each phase did:
perceive -> issue orders -> propagate (for next tick) -> move -> draw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import embodied_lot
from embodied_lot import BodyNode, ObjectNode, PlanNode, SerialNode
from grace_lot.lot_adapter.program import compile_program, parse_program

GRID = 12

EXAMPLES = [
    {"key": "one-order",
     "title": "One order",
     "program": "(parallel (body[1,0.34,0.06] center/H bottom/H*))",
     "note": "The body's region in the tree is the whole board, so it starts in the middle. Its unconditioned "
             "motion leaf switches on at the first tick and sends it one order: one body width (4 cells) to the "
             "right. The token walks there one cell at a time and stops."},
    {"key": "oscillator",
     "title": "Self-observing oscillator",
     "program": "(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H "
                "(body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0",
     "note": "The body is a child of the serial's right half, so it starts in the right strip. Two observers "
             "watch the board. When the right observer sees the body, it switches on the leftward motion leaf; the "
             "body walks left until the left observer sees it, which switches on the rightward leaf. A motion "
             "leaf's direction comes from its trigram: motion runs from the figure's side across, so horizontal "
             "'bottom' (figure on the left) moves right and 'top' moves left. It moves the bodies in its own strip."},
    {"key": "arrival",
     "title": "Walking into a lit leaf",
     "program": "(serial_h top/V* (parallel top/H* (body[0.5,0.5,0.06] center/H)))",
     "note": "The left strip's downward leaf (top/V*) is on from the start, but no body is there. The right "
             "strip's leftward leaf sends the body left. As soon as the body enters the left strip, the lit "
             "downward leaf gives it one order of its own, so it ends up in the lower left. Staying inside earns "
             "no more; leaving and coming back while the leaf is on would earn another."},
    {"key": "slow",
     "title": "Same oscillator, half speed",
     "program": "(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H "
                "(body[1,0.5,0.03] center/H))) | .0.1>.0.0 .1.1>.1.0",
     "note": "Halving the body's step halves its rate: it gains half a cell of credit per tick, so it moves "
             "every other tick. Positions stay on whole cells."},
    {"key": "narrow",
     "title": "Narrow body stalls",
     "program": "(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H "
                "(body[1,0.34,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0",
     "note": "An order moves the body one body width. A 4-cell body walks 4 cells left and stops between "
             "the two observers' views, so nothing switches it back on."},
    {"key": "parent-observers",
     "title": "Parent nodes as observers",
     "program": "(serial_h (parallel bottom/H* (parallel center/H)) (parallel top/H* (parallel center/H) "
                "(body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0",
     "note": "The observers are now parallel nodes. A parent observer perceives the pattern its children "
             "form, so here it behaves exactly like the leaf version."},
    {"key": "parent-target",
     "title": "Gating a whole group",
     "program": "(serial_h neutral/H (parallel center/H (parallel top/H*) "
                "(body[1,0.5,0.06] center/H))) | .1.0>.1.1",
     "note": "The left half is neutral (it draws nothing). The right half holds the observer, a group with the "
             "leftward motion leaf, and the body, so the body starts on the right, inside the group's space. The "
             "dashed edge targets the group, a parent. The group starts unknown (gate 0.5, too weak to fire). Once the right observer sees the body the group opens, its "
             "motion leaf fires once, and the body leaves. Briefly the observer sees the body's known-yin edge (gate 0); "
             "once the body is gone it sees only unknown cells, and the group settles back at 0.5, too weak to fire."},
    {"key": "orient",
     "title": "Templates turn their content",
     "program": "(serial_h (plan (serial_v top/V* bottom/V*) bottom/V) (plan (parallel bottom/V* top/H*) bottom/V))",
     "note": "A dynamic template does not move anything: it turns its content. Both halves draw the same content, "
             "bottom/V (a figure at its bottom). On the left, a vertical serial template gives two slots: the "
             "lower one faces down (top/V*), so its content is upside down, and the upper one faces up (bottom/V*). "
             "On the right, a parallel template sums up (bottom/V*) and left (top/H*): the content faces up-left, "
             "so its figure lies towards the lower right, in stair-stepped cell bands."},
    {"key": "nested-turns",
     "title": "Turns add up",
     "program": "(serial_h (plan top/H* (plan top/H* bottom/V)) (plan (parallel bottom/V* top/H*) (plan top/H* bottom/V)))",
     "note": "A dynamic template inside turned content turns again, and the turns add. Left half: facing left "
             "inside facing left is upside down, so bottom/V's figure is at the top. Right half: facing up-left, "
             "then left inside it, is down-left, so the figure lies towards the upper right."},
    {"key": "turned-body",
     "title": "A body in a turned frame",
     "program": "(parallel (plan top/H* (serial_h neutral/H (parallel top/H* (body[1,0.5,0.06] center/V)))))",
     "note": "The template top/H* turns the whole content to face left. Inside that frame, the body sits in "
             "the right strip and a leaf pushes it left. Turned onto the board, the frame's right strip is the "
             "upper half and its left is down, so the body starts at the top and walks down. Its stripe "
             "(center/V) is turned upright with it."},
]


def node_label(n) -> str:
    if isinstance(n, ObjectNode):
        return f"{n.name}/{'V' if n.orientation.value == 1 else 'H'}{'*' if n.is_dynamic else ''}"
    if isinstance(n, BodyNode) and not isinstance(n, PlanNode):
        y, x = n.position
        h, w = n.extent
        return f"body w={w:g} step={n.step_size:g}"
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


def towards(homes: dict, body_id: str, axis: int, direction: int) -> str:
    """Which way an order moves its body on the board (orders run along the body's own frame)."""
    by, bx = embodied_lot._view_turn(homes[body_id][0], (direction, 0) if axis == 0 else (0, direction))
    parts = [name for value, name in ((by, "down" if by > 0 else "up"), (bx, "right" if bx > 0 else "left"))
             if abs(value) > 1e-9]
    return "-".join(parts)


def board_cells(nid: str, masks: dict) -> dict:
    """A node's board cells (they may be turned) and their bounding box."""
    mask = masks.get(nid, np.zeros((GRID, GRID), dtype=bool))
    rows, cols = np.nonzero(mask)
    box = [int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1] if rows.size else [0, 0, 0, 0]
    return {"region": box, "cells": [[int(r), int(c)] for r, c in zip(rows, cols)]}


def observer_view(nid: str, masks: dict, perception: dict) -> dict:
    """An observer's board cells and what it sees."""
    return {"id": nid, **board_cells(nid, masks), "value": perception[nid]}


def trace(example: dict, ticks: int = 36) -> dict:
    embodied_lot.EVALUATE_IMAGINED_SUBTREES = True
    embodied_lot.GATE_STRUCTURES = True
    prog = parse_program(example["program"])
    g = compile_program(prog)
    g.dynamics, g.grid = "ludo", (GRID, GRID)
    g.reset_gates()
    g.assign_roles()
    root = g["par_O"].children[1]
    layout = tree_layout(g, root)
    shown = set(layout)
    nodes = [{"id": n.node_id, "type": n.node_type, "label": node_label(n),
              "x": layout[n.node_id][0], "y": layout[n.node_id][1],
              "dynamic": bool(isinstance(n, ObjectNode) and n.is_dynamic),
              "parent": next((p.node_id for p in g._parents().get(n.node_id, [])), None)}
             for n in g.nodes if n.node_id in shown]
    dashed = [{"source": e.source.node_id, "target": e.target.node_id} for e in g.dashed_edges]
    bodies = [n for n in g.nodes if isinstance(n, BodyNode) and not isinstance(n, PlanNode)]
    frames = []
    for t in range(ticks):
        g.assign_roles()
        seen = (g._imagery if g._imagery is not None else embodied_lot.blank_board((GRID, GRID))).round(2).tolist()
        g._scan()
        g._perceive()
        masks = g.view_masks((GRID, GRID))
        perception = {r.node_id: round(float(r.activation), 3) for r in g._observer_roots}
        before = {b.node_id: len(b.orders) for b in bodies}
        prev = dict(g._prev_act)
        g._issue_orders()
        homes = g._moving_bodies(g.node_views((GRID, GRID)))
        issued = []
        for b in bodies:
            for o in b.orders[before[b.node_id]:]:
                issued.append({"body": b.node_id, "leaf": o[5], "towards": towards(homes, b.node_id, o[0], o[1]),
                               "cells": o[2], "rate": round(o[3], 3)})
        onsets = [nid for nid, a in g._prev_act.items() if prev.get(nid, 0.0) <= embodied_lot.ONSET_THRESHOLD < a]
        gates_before = {n.node_id: n.gate for n in g.nodes}
        g._propagate_delayed()
        gate_changes = [{"node": n.node_id, "from": round(gates_before[n.node_id], 3), "to": round(n.gate, 3)}
                        for n in g.nodes if n.node_id in shown and abs(n.gate - gates_before[n.node_id]) > 1e-6]
        g._move_bodies_ludo()
        g._imagery = g._render_imagery()
        moved = g.view_masks((GRID, GRID))          # where each body is after this tick's move
        frames.append({
            "t": t,
            "seen": seen,
            "drawn": g._imagery.round(2).tolist(),
            "activation": {n.node_id: round(float(n.activation), 3) for n in g.nodes if n.node_id in shown},
            "gate": {n.node_id: round(float(n.gate), 3) for n in g.nodes if n.node_id in shown},
            "role": {n.node_id: getattr(n, "role", "imagined") for n in g.nodes if n.node_id in shown},
            "observers": [observer_view(r.node_id, masks, perception) for r in g._observer_roots],
            "bodies": [{"id": b.node_id, "window": board_cells(b.node_id, moved)["region"],
                        "cells": board_cells(b.node_id, moved)["cells"],
                        "orders": [{"towards": towards(homes, b.node_id, o[0], o[1]), "cells": o[2],
                                    "rate": round(o[3], 3), "credit": round(o[4], 3)} for o in b.orders]}
                       for b in bodies],
            "gate_changes": gate_changes,
            "onsets": onsets,
            "issued": issued,
        })
    return {**example, "nodes": nodes, "dashed": dashed, "frames": frames, "grid": GRID}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ludo_viz")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.dumps([trace(e) for e in EXAMPLES], separators=(",", ":"))
    (out / "traces.json").write_text(data)
    page = (Path(__file__).with_name("ludo_page.html")).read_text(encoding="utf-8")
    (out / "ludo.html").write_text(page.replace("__DATA__", data.replace("</", r"<\/")), encoding="utf-8")
    data = json.loads(data)
    for d in data:
        cols = "".join(f"{f['bodies'][0]['window'][2]:x}" for f in d["frames"]) if d["frames"][0]["bodies"] else "(no body)"
        print(f"{d['key']:18s} body column per tick: {cols}")


if __name__ == "__main__":
    main()
