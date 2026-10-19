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

from grace_lot.lot_adapter.program import OUT, OUTPUTS, P, Program, as_program, compile_program, nodes, output_roots


def _state_signature(graph) -> tuple:
    """Everything that can change a later frame: body positions, gates and activations."""
    orders = tuple(tuple(tuple(o) for o in n.orders) for n in graph.nodes if hasattr(n, "orders"))
    return (tuple((nid, tuple(s["position"])) for nid, s in sorted(graph.body_states().items())),
            tuple((n.gate, n.activation) for n in graph.nodes), orders,
            tuple(sorted(graph._prev_act.items())),
            tuple(sorted((k, tuple(sorted(v))) for k, v in getattr(graph, "_inside", {}).items())),
            graph.motion_state() if hasattr(graph, "motion_state") else ())


def motion_combinations(program: P, size: int) -> list:
    """Every joint state of the program's unknown-motion units, with its prior weight.

    universal*: "a" / "b" (one way or the other), 1/2 each; neutral*: "still" 1/2, "a" 1/4, "b" 1/4.
    A program without unknown motion has one combination, ``({}, 1.0)``.
    """
    import itertools
    graph = compile_program(program)
    graph.grid = (size, size)
    graph.reset_gates()
    if graph.dynamics == "gameboard":
        graph.assign_roles()
    units = sorted(graph.unknown_motion_units().items())
    opts = {"universal": [("a", 0.5), ("b", 0.5)], "neutral": [("still", 0.5), ("a", 0.25), ("b", 0.25)]}
    out = []
    for choice in itertools.product(*[opts[kind] for _, (kind, _, _) in units]):
        out.append(({key: c[0] for (key, _), c in zip(units, choice)}, float(np.prod([c[1] for c in choice]))))
    return out


COMPONENT_CACHE_SIZE = 5_000                    # about 30 KB each: ~150 MB per worker
_component_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def _components_independent() -> bool:
    """Whether trees joined by no dashed edge evolve independently, with outputs that do not depend on node
    numbering: separate tree spaces, grouping nodes, and unknown motion as a latent (no id-seeded pushes)."""
    import embodied_lot as E
    return (E.SEPARATE_TREE_SPACES and E.GROUPING_NODES and E.PROBABILISTIC_MOTION
            and E.DEFAULT_DYNAMICS == "gameboard")


def _id_paths(prog: Program) -> dict:
    """Node id -> tree path, numbered as ``compile_program`` numbers them."""
    trees = prog.tree.children if prog.tree.kind == OUTPUTS else (prog.tree,)
    walks = [nodes(t, (k,)) for k, t in enumerate(trees)] if prog.tree.kind == OUTPUTS else [nodes(prog.tree)]
    return {f"n{i}": path for i, (path, _) in enumerate((x for w in walks for x in w), start=1)}


def _components(prog: Program) -> list:
    """The forest's components (trees joined by dashed edges): (root indices, standalone forest)."""
    trees = prog.tree.children
    parent = list(range(len(trees)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a, b in prog.dashed:
        parent[find(a[0])] = find(b[0])
    groups = {}
    for i in range(len(trees)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        new = {k: j for j, k in enumerate(idx)}
        edges = tuple(sorted(((new[a[0]],) + a[1:], (new[b[0]],) + b[1:]) for a, b in prog.dashed if a[0] in new))
        out.append((idx, Program(P(OUTPUTS, (), tuple(trees[k] for k in idx)), edges)))
    return out


def _semantics() -> tuple:
    import embodied_lot as E
    return tuple((k, v) for k, v in vars(E).items() if k.isupper() and isinstance(v, (bool, int, float, str)))


def rollout(program: P, frames: int, size: int, motion_states: dict = None) -> np.ndarray:
    """``rollout_whole``, assembled from independently rolled-out and cached components when that is exact.

    Under the v7 semantics (``_components_independent``), trees joined by no dashed edge never affect each
    other, so each component's output windows are rolled out on their own, memoised, and put back in program
    order; components without output windows are skipped. Bit-for-bit the same as ``rollout_whole``.
    """
    prog = as_program(program)
    if prog.tree.kind != OUTPUTS or len(prog.tree.children) < 2 or not _components_independent():
        return rollout_whole(program, frames, size, motion_states)
    comps = _components(prog)
    channels = [p for p, q in nodes(prog.tree) if q.kind == OUT]
    if len(comps) < 2 or not channels:
        return rollout_whole(program, frames, size, motion_states)
    ids = _id_paths(prog)
    states = {k: v for k, v in (motion_states or {}).items()}
    sem = _semantics()
    out = np.empty((frames, len(channels), size, size))
    where = {}                                            # root index -> (component, its index there)
    for c, (idx, _) in enumerate(comps):
        for j, k in enumerate(idx):
            where[k] = (c, j)
    rolled = {}
    for c, (idx, sub) in enumerate(comps):
        sub_channels = [p for p, q in nodes(sub.tree) if q.kind == OUT]
        if not sub_channels:
            continue
        sub_ids = {path: nid for nid, path in _id_paths(sub).items()}
        sub_states = {}
        for key, value in states.items():
            unit, kind = key.rsplit(":", 1)
            path = ids[unit]
            if path[0] in idx:
                sub_states[f"{sub_ids[(idx.index(path[0]),) + path[1:]]}:{kind}"] = value
        ckey = (sub, tuple(sorted(sub_states.items())), frames, size, sem)
        v = _component_cache.get(ckey)
        if v is None:
            v = rollout_whole(sub, frames, size, sub_states)
            v = v if v.ndim == 4 else v[:, None]
            _component_cache[ckey] = v
            if len(_component_cache) > COMPONENT_CACHE_SIZE:
                _component_cache.popitem(last=False)
        else:
            _component_cache.move_to_end(ckey)
        rolled[c] = (v, {p: i for i, p in enumerate(sub_channels)})
    for i, p in enumerate(channels):
        c, j = where[p[0]]
        v, index = rolled[c]
        out[:, i] = v[:, index[(j,) + p[1:]]]
    return out


def rollout_whole(program: P, frames: int, size: int, motion_states: dict = None) -> np.ndarray:
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
    if graph.dynamics == "gameboard":
        graph.assign_roles()             # observers are perceivers and are not drawn
    if motion_states:
        graph.motion_states = dict(motion_states)   # this realization of the unknown motion
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
