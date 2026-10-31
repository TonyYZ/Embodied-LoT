"""Hashable program trees for search, compiled into ordinary ``ConceptGraph``s.

The search space is the *generative* fragment of the LoT: an imagined
subtree under ``ParallelNode(O, ...)`` with no input window and no dashed
edges. A program is an immutable ``P`` node:

    P("obj",      ("top", "V", False))                  ObjectNode(TOP, VERTICAL, static)
    P("parallel", (), children)                         ParallelNode
    P("serial_h", (), children) / P("serial_v", ...)    SerialNode(H / V)
    P("template", (), children)                         TemplateNode(templates..., material): copies, no grouping
    P("field",    (), children)                         FieldNode: a rigid unit moved only from outside
    P("body",     (), children)                         AgentBodyNode: a rigid unit its own leaves propel

``plan`` is read as ``template``. Copies of a group made by a template are independent instances. (Bodies with
numeric parameters, ``body[h,w,speed]``, still parse, for reading earlier runs; they no longer compile.)

A full program is ``Program(tree, dashed)``. ``dashed`` holds dashed gate
edges ``(source_path, target_path)``, where a path is the tuple of child
indices from the root. Targets are leaves (ObjectNodes); sources can be any
node. With ``step()``, a dashed loop through a ``plan(VOID ..)`` template
alternates its content between frames. This is how the existing LoT
expresses rhythm.

``P`` and ``Program`` are tuples, so programs can be dict keys (for caching
and posterior counts). ``str(p)`` is an s-expression with the dashed edges
after a ``|``, and ``parse`` reverses it. Compiling goes through the public
``embodied_lot`` API and does not change the LoT's semantics.
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Tuple

from embodied_lot import (
    ALL_PATTERNS, TRIGRAM_PATTERNS, AgentBodyNode, ConceptGraph, FieldNode, Node, ObjectNode, Orientation,
    OutputNode, ParallelNode, WindowNode, InWindowNode, SerialNode, TemplateNode,
)

LEAF = "obj"
COMPOSITIONS = ("parallel", "serial_h", "serial_v", "plan", "body")
MIN_CHILDREN = {"parallel": 1, "serial_h": 1, "serial_v": 1, "plan": 2, "body": 1, "outputs": 1, "out": 0, "in": 0,
                "template": 2, "field": 1}
GROUPS = ("field", "body")                 # rigid units
# An output window leaf, written ``out``: it reads the board inside the region its place in the tree
# gives it (``embodied_lot.WindowNode``). A program's output channels are its ``out`` leaves.
OUT = "out"
# An input window leaf, written ``in``: it shows an external field in the region its place in the tree
# gives it (``embodied_lot.InWindowNode``; data via ``ConceptGraph.set_window_input``). The search grammar
# does not generate it; programs that perceive external data use it.
IN = "in"

OUTPUT_ROOT = "par_O"
# A program whose root is ``P("outputs", (), (c_0, .., c_K-1))`` has K anonymous output
# channels. Channel k compiles to its own imagined root ``ParallelNode(par_O<k>)[OutputNode(O<k>), c_k]``.
# Several roots and output nodes are ordinary LoT graphs; the language is unchanged. ``outputs``
# is only a container, so it may appear only at the root and can never be a dashed-edge endpoint.
OUTPUTS = "outputs"


# The language is built into the engine (``embodied_lot``: legacy/all_languages and the git tag ``all-languages`` have
# every earlier one). A config may state it, but not ask for another. ``LANGUAGE``: the semantics keys runs recorded,
# with the value the language has; ``EARLIER``: what a saved run that lacks a key meant (it predates the key).
LANGUAGE = {"imagined_evaluation": "full", "dynamics": "gameboard", "gate_structures": True,
            "unknown_background": True, "tree_placed_bodies": True, "motion_away_from_figure": True,
            "orienting_templates": True, "tree_sized_bodies": True, "motion_kinds": True, "output_leaves": True,
            "content_windows": True, "grouping_nodes": True, "probabilistic_motion": True, "measure_trees": True,
            "pure_claims": True, "soft_edges": False, "static_fields": True, "standing_motion": True,
            "flow_perception": True, "multi_part_units": True, "contour_claims": True, "motion_claims": True,
            "open_edge": True, "see_start": True, "missions": True}
EARLIER = {**{k: False for k in LANGUAGE}, "imagined_evaluation": "legacy", "dynamics": "legacy", "gate_structures": True}
SWITCHES = {"solid_figure": True, "subcell": 1, "window_crop": False, "fine_motion": 1}          # what a config may still choose (default: the language's)


def apply_semantics(lot_cfg: dict) -> None:
    """Set the engine's switches from a config's ``lot`` section (``solid_figure``, ``subcell``, and ``onset_threshold``). A key
    that asks for an earlier language is an error: run that config at the tag ``all-languages``."""
    import embodied_lot
    other = {k: v for k, v in lot_cfg.items() if k in LANGUAGE and v != LANGUAGE[k]
             and not (k == "dynamics" and v == "ludo")}            # (the gameboard's earlier name)
    if other:
        raise ValueError(f"lot settings of an earlier language: {other} (run them at git tag all-languages)")
    embodied_lot.ONSET_THRESHOLD = float(lot_cfg.get("onset_threshold", 0.5))
    embodied_lot.SOLID_FIGURE = bool(lot_cfg.get("solid_figure", SWITCHES["solid_figure"]))
    embodied_lot.SUBCELL = int(lot_cfg.get("subcell", SWITCHES["subcell"]))
    embodied_lot.WINDOW_CROP = bool(lot_cfg.get("window_crop", SWITCHES["window_crop"]))
    embodied_lot.FINE_MOTION = int(lot_cfg.get("fine_motion", SWITCHES["fine_motion"]))


def resolved_lot(lot_cfg: dict) -> dict:
    """``lot_cfg`` with every semantics key explicit, for saving with a run."""
    return {**LANGUAGE, **SWITCHES, **lot_cfg}


def saved_lot(lot_cfg: dict) -> dict:
    """A saved run's ``lot`` section, a missing semantics key read as what it meant then (``apply_semantics`` refuses
    a run of an earlier language)."""
    return {**EARLIER, "solid_figure": False, **lot_cfg}


def output_roots(prog) -> list:
    """The output-root node ids of a program, one per channel (a single-output program has one)."""
    tree = prog.tree if isinstance(prog, Program) else prog
    return [f"{OUTPUT_ROOT}{k}" for k in range(len(tree.children))] if tree.kind == OUTPUTS else [OUTPUT_ROOT]


class P(NamedTuple):
    kind: str
    params: tuple = ()
    children: Tuple["P", ...] = ()

    def __str__(self) -> str:
        if self.kind in (OUT, IN):
            return self.kind
        if self.kind == OUTPUTS and len(self.children) >= 2:   # a forest: its trees side by side
            return " ".join(map(str, self.children))
        if self.kind == LEAF:
            name, ori, dyn = self.params
            return f"{name}/{ori}{'*' if dyn else ''}"
        head = self.kind
        if self.kind == "body" and self.params:             # parameter-free bodies print as plain (body ...)
            head += "[" + ",".join(f"{v:g}" for v in self.params) + "]"
        return f"({head} " + " ".join(map(str, self.children)) + ")"


Path = Tuple[int, ...]
Edge = Tuple[Path, Path]


class Program(NamedTuple):
    tree: P
    dashed: Tuple[Edge, ...] = ()       # sorted, without duplicates

    def __str__(self) -> str:
        if not self.dashed:
            return str(self.tree)
        return str(self.tree) + " | " + " ".join(f"{_fmt(a)}>{_fmt(b)}" for a, b in self.dashed)


def _fmt(path: Path) -> str:
    return "." + ".".join(map(str, path))


def _unfmt(text: str) -> Path:
    return tuple(int(t) for t in text[1:].split(".") if t)


def as_program(x) -> Program:
    return x if isinstance(x, Program) else Program(x)


def with_edges(tree: P, edges) -> Program:
    return Program(tree, tuple(sorted(set(edges))))


def parse_program(text: str) -> Program:
    tree_text, _, edge_text = text.partition(" | ")
    edges = [tuple(_unfmt(x) for x in e.split(">")) for e in edge_text.split()]
    return with_edges(parse(tree_text), edges)


def leaf(name: str, ori: str = "V", dynamic: bool = False) -> P:
    return P(LEAF, (name, ori, bool(dynamic)))


def parse(text: str) -> P:
    """Inverse of ``str(P)``."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def read() -> P:
        nonlocal pos
        tok = tokens[pos]; pos += 1
        if tok in (OUT, IN):
            return P(tok)
        if tok != "(":
            name, _, rest = tok.partition("/")
            if not rest or rest.rstrip("*") not in ("H", "V"):
                raise ValueError(f"leaf {tok!r}: write it as name/orientation, H or V, with * if it moves "
                                 f"(e.g. upper/V, bottom/H*), or out / in for a window")
            return leaf(name, rest[0], rest.endswith("*"))
        head = tokens[pos]; pos += 1
        kind, params = head, ()
        if head.startswith("body["):                # earlier runs: body[...] with parameters
            kind, params = "body", tuple(float(v) for v in head[5:-1].split(",") if v)   # body[] = no parameters
        elif head not in MIN_CHILDREN or head in (OUT, IN):                 # (outputs: a one-tree forest)
            raise ValueError(f"unknown composition {head!r}: use one of "
                             f"{', '.join(k for k in MIN_CHILDREN if k not in (OUT, IN, OUTPUTS))}")
        children = []
        while tokens[pos] != ")":
            children.append(read())
        pos += 1
        return P(kind, params, tuple(children))

    trees = [read()]
    while pos < len(tokens):                       # several top-level trees: a forest
        trees.append(read())
    return trees[0] if len(trees) == 1 else P(OUTPUTS, (), tuple(trees))


# ---- tree utilities -------------------------------------------------------

def nodes(p: P, path: tuple = ()) -> Iterator[Tuple[tuple, P]]:
    """Pre-order (path, node) pairs. A path is the tuple of child indices from the root."""
    yield path, p
    for i, c in enumerate(p.children):
        yield from nodes(c, path + (i,))


def size(p: P) -> int:
    return 1 + sum(size(c) for c in p.children)


def depth(p: P) -> int:
    return 1 + max((depth(c) for c in p.children), default=0)


def get(p: P, path: tuple) -> P:
    for i in path:
        p = p.children[i]
    return p


def replace(p: P, path: tuple, new: P) -> P:
    if not path:
        return new
    kids = list(p.children)
    kids[path[0]] = replace(kids[path[0]], path[1:], new)
    return p._replace(children=tuple(kids))


def leaves(p: P) -> List[P]:
    return [n for _, n in nodes(p) if n.kind == LEAF]


def is_valid(p: P) -> bool:
    return all(n.kind == LEAF or len(n.children) >= MIN_CHILDREN[n.kind] for _, n in nodes(p))


def motion_units(prog) -> int:
    """How many unknown-motion units a program has (what the prior bounds, ``max_motion_units``).

    A dynamic universal / neutral leaf outside observer subtrees belongs to its innermost field or body, or else is
    its own unit; universal and neutral leaves of one unit count separately.
    """
    prog = as_program(prog)
    srcs, tgts = {a for a, _ in prog.dashed}, {b for _, b in prog.dashed}
    observer_roots = srcs - tgts
    units = set()

    def walk(q, path, group):
        if any(path[:len(r)] == r for r in observer_roots):
            return
        if q.kind == LEAF:
            name, _, dyn = q.params
            if dyn and name in ("universal", "neutral"):
                units.add((group if group is not None else path, name))
            return
        if q.kind in GROUPS:
            group = path
        for i, c in enumerate(q.children):
            walk(c, path + (i,), group)
    walk(prog.tree, (), None)
    return len(units)


def windows_as_content(p: P) -> bool:
    """No ``out``/``in`` window is a template: in a template, a window may only be the last child (the material)."""
    return all(c.kind not in (OUT, IN) for _, n in nodes(p) if n.kind in ("template", "plan") for c in n.children[:-1])


# ---- compilation ----------------------------------------------------------

def candidate_edges(tree: P, targets: str = "leaves") -> list:
    """Every dashed edge the search may add: (source path, target path), source != target.

    Sources are all nodes; targets are leaves only (``"leaves"``) or all nodes (``"any"``).
    The ``outputs`` container is neither, since it is not a node of the LoT graph.
    """
    if targets not in ("leaves", "any"):
        raise ValueError(f"unknown dashed_targets {targets!r}")
    real = [(p, n) for p, n in nodes(tree) if not (p == () and tree.kind == OUTPUTS) and n.kind not in (OUT, IN)]   # windows: no gates
    tgts = [p for p, n in real if targets == "any" or n.kind == LEAF]
    return [(a, b) for a, _ in real for b in tgts if a != b]


def edges_valid(prog: Program, targets: str = "leaves") -> bool:
    """Every dashed edge is one the search could have added (see ``candidate_edges``)."""
    allowed = set(candidate_edges(prog.tree, targets))
    return all(e in allowed for e in prog.dashed)


def compile_program(prog, name: str = "program") -> ConceptGraph:
    """``ParallelNode(par_O)[OutputNode(O), <tree>]`` plus the program's dashed edges, as a ConceptGraph."""
    prog = as_program(prog)
    g = ConceptGraph(name)
    counter = 0
    by_path = {}

    def build(q: P, path: Path = ()) -> Node:
        if q.kind == OUTPUTS:
            raise ValueError("'outputs' may only appear at the root")
        nonlocal counter
        counter += 1
        nid = f"n{counter}"
        by_path[path] = node = _make(q, nid)
        if q.children:
            g.connect(node, *[build(c, path + (i,)) for i, c in enumerate(q.children)])
        else:
            g.add_node(node)
        return node

    if prog.tree.kind == OUTPUTS:
        for k, child in enumerate(prog.tree.children):
            par = g.connect(ParallelNode(f"{OUTPUT_ROOT}{k}"), OutputNode(f"O{k}"), build(child, (k,)))
            g.root = g.root or par
    else:
        body = build(prog.tree)
        g.root = g.connect(ParallelNode(OUTPUT_ROOT), OutputNode("O"), body)
    for a, b in prog.dashed:
        g.add_dashed_edge(by_path[a], by_path[b])
    return g


def _make(q: P, nid: str) -> Node:
    if q.kind == LEAF:
        pname, ori, dyn = q.params
        o = Orientation.VERTICAL if ori == "V" else Orientation.HORIZONTAL
        return ObjectNode(nid, ALL_PATTERNS[pname], o, is_dynamic=dyn, name=pname)
    if q.kind == OUT:
        return WindowNode(nid)
    if q.kind == IN:
        return InWindowNode(nid)
    if q.kind == "parallel":
        return ParallelNode(nid)
    if q.kind in ("serial_h", "serial_v"):
        return SerialNode(nid, Orientation.HORIZONTAL if q.kind == "serial_h" else Orientation.VERTICAL)
    if q.kind in ("plan", "template"):
        return TemplateNode(nid)
    if q.kind == "field":
        return FieldNode(nid)
    if q.params:
        raise ValueError(f"a body with parameters ({q}) is an earlier language's (run it at git tag all-languages)")
    return AgentBodyNode(nid)
