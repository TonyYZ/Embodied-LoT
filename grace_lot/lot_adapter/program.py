"""Hashable program trees for search, compiled into ordinary ``ConceptGraph``s.

The search space is the *generative* fragment of the LoT: an imagined
subtree under ``ParallelNode(O, ...)`` with no input window and no dashed
edges. A program is an immutable ``P`` node:

    P("obj",      ("top", "V", False))                  ObjectNode(TOP, VERTICAL, static)
    P("parallel", (), children)                         ParallelNode
    P("serial_h", (), children) / P("serial_v", ...)    SerialNode(H / V)
    P("plan",     (), children)                         PlanNode(templates..., content)
    P("body",     (), children)                         BodyNode: sized and placed by the tree
    P("body",     (h, w, speed), children)              BodyNode(extent, step_size): earlier runs only
    P("body",     (y, x, h, w, speed), children)        BodyNode(position, ...): earlier runs only

With ``lot.grouping_nodes`` (the default), copying, grouping and agency are separate operators:

    P("template", (), children)                         TemplateNode(templates..., material): copies, no grouping
    P("field",    (), (child,))                         FieldNode: a rigid unit moved only from outside
    P("body",     (), (child,))                         AgentBodyNode: a rigid unit its own leaves propel

``plan`` is read as ``template``. Fields and bodies take one child; copies of a group made by a template
are independent instances.

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
    TRIGRAM_PATTERNS, AgentBodyNode, BodyNode, ConceptGraph, FieldNode, Node, ObjectNode, Orientation,
    OutputNode, ParallelNode, WindowNode, InWindowNode, PlanNode, SerialNode, TemplateNode,
)

LEAF = "obj"
COMPOSITIONS = ("parallel", "serial_h", "serial_v", "plan", "body")
MIN_CHILDREN = {"parallel": 1, "serial_h": 1, "serial_v": 1, "plan": 2, "body": 1, "outputs": 1, "out": 0, "in": 0,
                "template": 2, "field": 1}
GROUPS = ("field", "body")                 # unary rigid units under grouping_nodes
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


def apply_semantics(lot_cfg: dict) -> None:
    """Set the core's imagined-evaluation semantics from ``lot.imagined_evaluation``.

    "full" (default): every imagined node is evaluated on every step.
    "legacy": the earlier shortcut, used by runs made before this was fixed.
    Also sets ``lot.dynamics`` ("gameboard" by default | "legacy" for earlier runs), ``lot.onset_threshold`` and
    ``lot.gate_structures`` (a structure's gate scales its composite; default true) and
    ``lot.unknown_background`` (undrawn cells render as unknown, 0.5; default true) and
    ``lot.tree_placed_bodies`` (bodies start centred in their tree region; default true),
    ``lot.motion_away_from_figure`` (horizontal motion matches vertical; default true) and
    ``lot.orienting_templates`` (dynamic templates orient their content; default true) and
    ``lot.tree_sized_bodies`` (bodies have no parameters: size from the tree; default true) and
    ``lot.motion_kinds`` (each motion trigram has its own kind of motion, moving mass too; default true),
    ``lot.pure_claims`` (centre-axis patterns claim the substance, others only their figure; nested claims win;
    default false) and ``lot.solid_figure`` (moving bodies cannot enter the rest of their tree's figure; default false).
    """
    import embodied_lot
    mode = lot_cfg.get("imagined_evaluation", "full")
    if mode not in ("full", "legacy"):
        raise ValueError(f"unknown imagined_evaluation {mode!r}")
    embodied_lot.EVALUATE_IMAGINED_SUBTREES = mode == "full"
    dynamics = embodied_lot.DYNAMICS_ALIASES.get(lot_cfg.get("dynamics", "gameboard"), lot_cfg.get("dynamics", "gameboard"))
    if dynamics not in ("legacy", "gameboard"):
        raise ValueError(f"unknown dynamics {dynamics!r}")
    embodied_lot.DEFAULT_DYNAMICS = dynamics
    embodied_lot.ONSET_THRESHOLD = float(lot_cfg.get("onset_threshold", 0.5))
    embodied_lot.GATE_STRUCTURES = bool(lot_cfg.get("gate_structures", True))
    embodied_lot.UNKNOWN_BACKGROUND = bool(lot_cfg.get("unknown_background", True))
    embodied_lot.TREE_PLACED_BODIES = bool(lot_cfg.get("tree_placed_bodies", True))
    embodied_lot.MOTION_AWAY_FROM_FIGURE = bool(lot_cfg.get("motion_away_from_figure", True))
    embodied_lot.ORIENTING_TEMPLATES = bool(lot_cfg.get("orienting_templates", True))
    embodied_lot.TREE_SIZED_BODIES = bool(lot_cfg.get("tree_sized_bodies", True))
    embodied_lot.MOTION_KINDS = bool(lot_cfg.get("motion_kinds", True))
    embodied_lot.SEPARATE_TREE_SPACES = bool(lot_cfg.get("output_leaves", True))
    embodied_lot.GROUPING_NODES = bool(lot_cfg.get("grouping_nodes", True))
    embodied_lot.PROBABILISTIC_MOTION = bool(lot_cfg.get("probabilistic_motion", True))
    embodied_lot.MEASURE_TREES = bool(lot_cfg.get("measure_trees", True))
    embodied_lot.PURE_CLAIMS = bool(lot_cfg.get("pure_claims", False))
    embodied_lot.SOLID_FIGURE = bool(lot_cfg.get("solid_figure", False))
    embodied_lot.SOFT_EDGES = bool(lot_cfg.get("soft_edges", False))
    embodied_lot.STATIC_FIELDS = bool(lot_cfg.get("static_fields", False))
    embodied_lot.STANDING_MOTION = bool(lot_cfg.get("standing_motion", False))
    embodied_lot.FLOW_PERCEPTION = bool(lot_cfg.get("flow_perception", False))
    embodied_lot.MULTI_PART_UNITS = bool(lot_cfg.get("multi_part_units", False))
    embodied_lot.CONTOUR_CLAIMS = bool(lot_cfg.get("contour_claims", False))


# Semantics keys whose defaults have changed. A saved run config that lacks one was made
# before the key existed, i.e. under the legacy behaviour.
SEMANTICS_KEYS = {"imagined_evaluation": ("full", "legacy"), "dynamics": ("gameboard", "legacy"),
                  "unknown_background": (True, False), "tree_placed_bodies": (True, False),
                  "motion_away_from_figure": (True, False), "orienting_templates": (True, False),
                  "tree_sized_bodies": (True, False), "motion_kinds": (True, False),
                  "output_leaves": (True, False), "content_windows": (True, False), "grouping_nodes": (True, False),
                  "probabilistic_motion": (True, False), "measure_trees": (True, False),
                  "pure_claims": (False, False), "solid_figure": (False, False), "soft_edges": (False, False),
                  "static_fields": (False, False), "standing_motion": (False, False),
                  "flow_perception": (False, False), "multi_part_units": (False, False),
                  "contour_claims": (False, False)}


def resolved_lot(lot_cfg: dict) -> dict:
    """``lot_cfg`` with every semantics key explicit (current defaults), for saving with a run."""
    return {**{k: new for k, (new, _) in SEMANTICS_KEYS.items()}, **lot_cfg}


def saved_lot(lot_cfg: dict) -> dict:
    """A saved run's ``lot`` section, with missing semantics keys read as the legacy behaviour."""
    return {**{k: old for k, (_, old) in SEMANTICS_KEYS.items()}, **lot_cfg}


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
            name, rest = tok.split("/")
            return leaf(name, rest[0], rest.endswith("*"))
        head = tokens[pos]; pos += 1
        kind, params = head, ()
        if head.startswith("body["):                # earlier runs: body[...] with parameters
            kind, params = "body", tuple(float(v) for v in head[5:-1].split(",") if v)   # body[] = no parameters
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


def motion_units(prog, grouping: bool = True) -> int:
    """How many unknown-motion units a program has (as ``ConceptGraph.unknown_motion_units``).

    A dynamic universal / neutral leaf outside observer subtrees belongs to its innermost field or body
    (with grouping) or else is its own unit; universal and neutral leaves of one unit count separately.
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
        if grouping and q.kind in GROUPS:
            group = path
        for i, c in enumerate(q.children):
            walk(c, path + (i,), group)
    walk(prog.tree, (), None)
    return len(units)


def windows_as_content(p: P, grouping: bool = False) -> bool:
    """No ``out``/``in`` window is a template: in a template (or an earlier body / plan), a window may only be
    the last child (the material)."""
    kinds = ("template", "plan") if grouping else ("body", "plan", "template")
    return all(c.kind not in (OUT, IN) for _, n in nodes(p) if n.kind in kinds for c in n.children[:-1])


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
        return ObjectNode(nid, TRIGRAM_PATTERNS[pname], o, is_dynamic=dyn, name=pname)
    if q.kind == OUT:
        return WindowNode(nid)
    if q.kind == IN:
        return InWindowNode(nid)
    if q.kind == "parallel":
        return ParallelNode(nid)
    if q.kind in ("serial_h", "serial_v"):
        return SerialNode(nid, Orientation.HORIZONTAL if q.kind == "serial_h" else Orientation.VERTICAL)
    import embodied_lot
    if embodied_lot.GROUPING_NODES:
        if q.kind in ("plan", "template"):
            return TemplateNode(nid)
        if q.kind == "field":
            return FieldNode(nid)
        if q.kind == "body" and not q.params:
            return AgentBodyNode(nid)
    if q.kind in ("plan", "template"):
        return PlanNode(nid)
    if q.kind == "field":
        raise ValueError("'field' needs lot.grouping_nodes")
    if not q.params:
        return BodyNode(nid)                                          # sized and placed by the tree
    if len(q.params) == 3:
        h, w, speed = q.params
        return BodyNode(nid, extent=(h, w), step_size=speed)       # placed by place_bodies()
    y, x, h, w, speed = q.params
    return BodyNode(nid, position=(y, x), extent=(h, w), step_size=speed)
