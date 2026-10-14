#iloveguineapig
"""
Interactive visualization for Embodied-LoT concept graphs.

The module exports any ConceptGraph to a standalone HTML file. The page lets
you inspect gates/activations, drag nodes, toggle edge classes, make
client-side topology edits, and replay snapshots recorded from the real Python
model.

Typical use:

    from visualize_lot import visualize

    graph.set_field("I", field)
    graph.evaluate()
    path = visualize(graph, "network.html")

To replay actual model dynamics, pass fields or a precomputed trace:

    path = visualize(graph, "run.html", fields=[field0, field1, field2], mode="step")

Browser edits are exploratory only. They change the embedded visualization
state, not the original Python ConceptGraph.

The page itself lives in ``visualizer/``: ``page.html`` holds the layout,
``viewer.css`` the styles and ``viewer.js`` the browser code (rendering,
editing, and the JavaScript copy of the evaluator, which must stay in step
with ``embodied_lot.py``). They are plain files, so edit them directly; this
module only fills in the title and the graph data.
"""

from __future__ import annotations

import base64
import copy
import html
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from embodied_lot import (
    BodyNode,
    ConceptGraph,
    DashedEdge,
    InputNode,
    ObjectNode,
    OutputNode,
    ParallelNode,
    PlanNode,
    ReceptiveField,
    SerialNode,
    SolidEdge,
)


def _clip01(value: Any) -> float:
    """Return a JSON-friendly float clipped to [0, 1]."""
    try:
        return float(np.clip(value, 0.0, 1.0))
    except Exception:
        return 0.0


def _field_summary(field: Optional[ReceptiveField]) -> Optional[Dict[str, Any]]:
    if field is None:
        return None
    data = field.activation
    result: Dict[str, Any] = {
        "shape": list(data.shape),
        "mean": float(data.mean()),
        "min": float(data.min()),
        "max": float(data.max()),
        "bounds": list(getattr(field, "bounds", (0.0, 1.0, 0.0, 1.0))),
    }
    if data.size <= 10_000:
        result["values"] = data.tolist()
    if field.motion is not None:
        motion = field.motion
        result["motion_mean_abs"] = float(np.abs(motion).mean())
        result["motion_shape"] = list(motion.shape)
    return result


def _node_payload(node: Any, field: Optional[ReceptiveField] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": node.node_id,
        "type": node.node_type,
        "class": type(node).__name__,
        "activation": _clip01(getattr(node, "activation", 0.0)),
        "gate": _clip01(getattr(node, "gate", 1.0)),
        "children": [child.node_id for child in getattr(node, "children", [])],
    }

    if isinstance(node, InputNode):
        payload.update({
            "modality": node.modality,
            "field": _field_summary(field or node.field),
        })
    elif isinstance(node, OutputNode):
        payload["role"] = "output"
    elif isinstance(node, ObjectNode):
        payload.update({
            "pattern": list(node.pattern),
            "pattern_name": node.name,
            "orientation": node.orientation.name.lower(),
            "dynamic": bool(node.is_dynamic),
        })
    elif isinstance(node, ParallelNode):
        payload["inherited_weight"] = float(node.inherited_weight)
    elif isinstance(node, SerialNode):
        payload["split_orientation"] = node.split_orientation.name.lower()
    elif isinstance(node, BodyNode):
        payload.update({
            "position": None if node.position is None else list(node.position),
            "extent": None if node.extent is None else list(node.extent),
            "velocity": list(node.velocity),
            "step_size": node.step_size,
        })
        if isinstance(node, PlanNode):
            payload["subfield_shape"] = node.subfield_shape

    if field is not None and not isinstance(node, InputNode):
        payload["field"] = _field_summary(field)
    return payload


def _edge_payload(edge: SolidEdge | DashedEdge) -> Dict[str, str]:
    return {
        "source": edge.source.node_id,
        "target": edge.target.node_id,
        "type": edge.edge_type.value,
    }


_ICON_EXTENSIONS = {".png", ".gif", ".jpg", ".jpeg", ".webp", ".svg"}


def _image_path_to_data_url(path: str | Path) -> str:
    src = Path(path)
    data = src.read_bytes()
    mime, _ = mimetypes.guess_type(src.name)
    if mime is None:
        mime = "image/svg+xml" if src.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _icon_sprites_payload(
    icon_sprites: Optional[Mapping[str, str | Path]],
) -> Dict[str, str]:
    if not icon_sprites:
        return {}
    return {
        str(name): _image_path_to_data_url(path)
        for name, path in icon_sprites.items()
        if path is not None
    }


def icon_sprites_from_folder(
    folder: str | Path,
    *,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """
    Return an icon-sprite mapping from files in a folder.

    File stems become icon names, so `sprites/eye.png` overrides the built-in
    `eye` icon, `sprites/hand.png` overrides `hand`, and so on.  Pass `names`
    to restrict which stems are accepted.
    """
    root = Path(folder)
    allowed = None if names is None else {str(name) for name in names}
    sprites: Dict[str, Path] = {}
    for item in sorted(root.iterdir()):
        if not item.is_file() or item.suffix.lower() not in _ICON_EXTENSIONS:
            continue
        if allowed is not None and item.stem not in allowed:
            continue
        sprites[item.stem] = item
    return sprites


def snapshot(graph: ConceptGraph, label: Optional[str] = None) -> Dict[str, Any]:
    """
    Capture the current graph topology and node state.

    This does not call evaluate(). It records whatever gate, activation, body,
    and input states are already present in the graph.
    """
    fields = graph.derived_fields(shape=(9, 9))
    nodes = [_node_payload(node, fields.get(node.node_id)) for node in graph.nodes]

    return {
        "label": label or graph.name or "snapshot",
        "graph_name": graph.name,
        "nodes": nodes,
        "edges": (
            [_edge_payload(edge) for edge in graph.solid_edges]
            + [_edge_payload(edge) for edge in graph.dashed_edges]
        ),
        "roots": [node.node_id for node in graph.roots],
        "result": _graph_result(graph),
        "body_states": graph.body_states(),
        "activation_summary": graph.activation_summary(),
    }


def trace(
    graph: ConceptGraph,
    fields: Optional[Sequence[ReceptiveField]] = None,
    *,
    steps: int = 0,
    mode: str = "step",
    input_id: str = "I",
    copy_graph: bool = True,
) -> List[Dict[str, Any]]:
    """
    Record a sequence of real Python model states.

    Parameters
    ----------
    graph:
        Source graph.
    fields:
        Optional input fields. Each field is set on `input_id` before the
        frame is evaluated.
    steps:
        Number of frames to record when `fields` is omitted. If both are
        omitted, a single current-state frame is returned.
    mode:
        "step" preserves temporal graph state and moves bodies. "evaluate"
        performs stateless evaluation for each frame.
    input_id:
        InputNode id or modality to update when fields are provided.
    copy_graph:
        Deep-copy the graph before replaying so trace generation does not
        mutate the caller's object. Set False if you explicitly want mutation.
    """
    if mode not in {"step", "evaluate"}:
        raise ValueError("mode must be either 'step' or 'evaluate'.")

    runner = copy.deepcopy(graph) if copy_graph else graph
    frames: List[Dict[str, Any]] = []
    supplied_fields = list(fields) if fields is not None else None

    if supplied_fields is None:
        if steps <= 0:
            return [snapshot(runner, label="current")]
        total = int(steps)
        for i in range(total):
            if mode == "step":
                runner.step()
            else:
                runner.evaluate()
            frames.append(snapshot(runner, label=f"t={i}"))
        return frames

    for i, field in enumerate(supplied_fields):
        runner.set_field(input_id, field)
        if mode == "step":
            runner.step()
        else:
            runner.evaluate()
        frames.append(snapshot(runner, label=f"t={i}"))
    return frames


def graph_data(
    graph: ConceptGraph,
    *,
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    icon_sprites: Optional[Mapping[str, str | Path]] = None,
) -> Dict[str, Any]:
    """
    Return the JSON-serializable payload consumed by the HTML visualizer.
    """
    base = snapshot(graph, label="current")
    frame_list = list(frames) if frames is not None else [base]
    return {
        "version": 1,
        "name": graph.name or "Embodied-LoT graph",
        "base": base,
        "frames": frame_list,
        "icon_sprites": _icon_sprites_payload(icon_sprites),
    }


def visualize(
    graph: ConceptGraph,
    path: str | Path = "lot_visualization.html",
    *,
    fields: Optional[Sequence[ReceptiveField]] = None,
    steps: int = 0,
    mode: str = "step",
    input_id: str = "I",
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    title: Optional[str] = None,
    icon_sprites: Optional[Mapping[str, str | Path]] = None,
) -> Path:
    """
    Write an interactive HTML visualization and return its path.

    If `fields` or `steps` are provided, a trace is recorded from a deep copy of
    the graph using the real ConceptGraph methods. If `frames` are supplied,
    they are used directly.
    """
    if frames is None and (fields is not None or steps > 0):
        frames = trace(graph, fields, steps=steps, mode=mode, input_id=input_id)
    return write_html(
        graph,
        path,
        frames=frames,
        title=title,
        icon_sprites=icon_sprites,
    )


def write_html(
    graph: ConceptGraph,
    path: str | Path = "lot_visualization.html",
    *,
    frames: Optional[Sequence[Dict[str, Any]]] = None,
    title: Optional[str] = None,
    icon_sprites: Optional[Mapping[str, str | Path]] = None,
) -> Path:
    """
    Write a standalone HTML graph viewer.
    """
    out = Path(path)
    payload = graph_data(graph, frames=frames, icon_sprites=icon_sprites)
    page_title = title or payload["name"]
    html = _html_template(page_title, payload)
    out.write_text(html, encoding="utf-8")
    return out.resolve()


def _graph_result(graph: ConceptGraph) -> float:
    returns = [n for n in graph.nodes if isinstance(n, OutputNode)]
    if returns:
        return _clip01(returns[-1].activation)
    roots = graph.roots
    if roots:
        return _clip01(roots[-1].activation)
    return 0.0


_ASSET_DIR = Path(__file__).with_name("visualizer")


def _html_template(title: str, payload: Dict[str, Any]) -> str:
    """Assemble the standalone page from ``visualizer/page.html``, ``viewer.css`` and ``viewer.js``."""
    asset = lambda name: (_ASSET_DIR / name).read_text(encoding="utf-8")
    return (asset("page.html")
            .replace("@@TITLE@@", html.escape(title))
            .replace("@@STYLE@@", asset("viewer.css"))
            .replace("@@SCRIPT@@", asset("viewer.js"))
            .replace("@@DATA@@", json.dumps(payload, ensure_ascii=False)))


__all__ = [
    "graph_data",
    "icon_sprites_from_folder",
    "snapshot",
    "trace",
    "visualize",
    "write_html",
]
