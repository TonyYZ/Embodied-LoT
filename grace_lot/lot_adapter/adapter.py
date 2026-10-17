"""lot_trajectory(r): the motion a program generates, in the same format as human_trajectory(v).

    graph = compile_program(r)          ParallelNode(O, r), with no input window
    graph.reset_gates()
    for t in 0..T-1:
        frame_t = graph.render_output((H, W), place_bodies=True)
        graph.step()                    moves the BodyNodes

Each frame is the LoT's own generative read-out (``render_output``), not a
pixel rendering. Frames change when bodies move or dashed edges change gates. Repeated
states (static or periodic) are detected and copied rather than re-rendered. Results are memoised per program.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from grace_lot.lot_adapter.program import P, compile_program, output_roots


def _state_signature(graph) -> tuple:
    """Everything that can change a later frame: body positions, gates and activations."""
    orders = tuple(tuple(tuple(o) for o in n.orders) for n in graph.nodes if hasattr(n, "orders"))
    return (tuple((nid, tuple(s["position"])) for nid, s in sorted(graph.body_states().items())),
            tuple((n.gate, n.activation) for n in graph.nodes), orders,
            tuple(sorted(graph._prev_act.items())),
            tuple(sorted((k, tuple(sorted(v))) for k, v in getattr(graph, "_inside", {}).items())),
            graph.motion_state() if hasattr(graph, "motion_state") else ())


def rollout(program: P, frames: int, size: int) -> np.ndarray:
    """Rendered output fields: (T, H, W) for one output, or (T, K, H, W) for a K-channel program.

    The rollout is deterministic. If the state (body positions, gates,
    activations) at frame t equals the state at an earlier frame j, every
    later frame repeats with period t - j. That covers static programs
    (period 1) and dashed-edge oscillations (period 2), which are then
    copied instead of re-rendered.
    """
    graph = compile_program(program)
    graph.grid = (size, size)
    graph.reset_gates()
    if graph.dynamics == "ludo":
        graph.assign_roles()             # observers are perceivers and are not drawn
    windows = graph.windows()
    if windows:                       # output leaves: channel k = the board inside window k's region
        return _window_rollout(graph, [w.node_id for w in windows], frames, size)
    roots = output_roots(program)
    multi = getattr(program, "tree", program).kind == "outputs"
    out = np.empty((frames, len(roots), size, size) if multi else (frames, size, size))
    seen = {}
    for t in range(frames):
        sig = _state_signature(graph)
        if sig in seen:
            period = t - seen[sig]
            for u in range(t, frames):
                out[u] = out[u - period]
            break
        seen[sig] = t
        frame = [graph.render_output((size, size), output_root_id=r, place_bodies=True).activation for r in roots]
        out[t] = np.stack(frame) if multi else frame[0]
        graph.step()
    return out


def _window_rollout(graph, ids, frames: int, size: int) -> np.ndarray:
    """(T, K, size, size): each output window's own region, in its own frame, stretched to the full grid.

    An output is only the region its window occupies: its tree's space inside the window (carried into
    the window's frame if it sits in turned content), wherever the window is and however big. A window
    in a body moves with the body, so a pattern carried along stays the same output; an output changes
    only when something moves relative to its window. For output, the window's cells are stretched to
    the full grid (each cell repeated). A window in several template slots uses its heaviest view.
    """
    import embodied_lot
    from embodied_lot import ReceptiveField, _view_field, blank_board
    tree = graph.tree_index()
    out = np.empty((frames, len(ids), size, size))
    seen = {}
    for t in range(frames):
        sig = _state_signature(graph)
        if sig in seen:
            period = t - seen[sig]
            for u in range(t, frames):
                out[u] = out[u - period]
            break
        seen[sig] = t
        spaces, shared = graph._render_all()          # reused when the step just drew this same state
        views = graph.node_views((size, size))        # memoised: recomputed only if something moved
        for k, i in enumerate(ids):
            space = spaces[tree[i]] if embodied_lot.SEPARATE_TREE_SPACES else shared
            vs = views.get(i)
            if not vs:                                  # a window with no region shows nothing known
                out[t, k] = blank_board((size, size))
                continue
            _, chain, rect = max(vs, key=lambda v: v[0])
            local = _view_field(ReceptiveField(space), chain, rect).activation
            h, w = local.shape
            out[t, k] = local[np.arange(size) * h // size][:, np.arange(size) * w // size]
        graph.step()
    return out


class LotRenderer:
    """``lot_trajectory`` with an LRU cache keyed by the program."""

    def __init__(self, frames: int, size: int, cache_size: int = 5_000) -> None:
        self.frames, self.size = frames, size
        self._cache: "OrderedDict[P, np.ndarray]" = OrderedDict()
        self._cache_size = cache_size
        self.hits = self.misses = 0

    def __call__(self, program: P) -> np.ndarray:
        if program in self._cache:
            self.hits += 1
            self._cache.move_to_end(program)
            return self._cache[program]
        self.misses += 1
        result = rollout(program, self.frames, self.size).astype(np.float32)
        result.setflags(write=False)
        self._cache[program] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result
