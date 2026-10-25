"""Core graph model for Embodied Language of Thought.

A concept is a ``ConceptGraph`` of perceptual/attentional operators:

* ``ObjectNode``    trigram scanner (an image schema over three bands)
* ``ParallelNode``  attention hub: pools ``InputNode`` fields, averages scanners
* ``SerialNode``    splits the field into strips, averages children on them
* ``PlanNode``      template/content replacement rule
* ``BodyNode``      a plan with position, extent and velocity that moves
* ``InputNode`` / ``OutputNode``  sensory and output windows

Solid edges compose nodes. Dashed edges are positive conditional gates:
``target.gate = mean(source.activation)``. They never invert their source.

Observed object nodes (below a ``ParallelNode`` that owns an ``InputNode``)
scan a field: ``activation = gate * scan(field)``. Field-free object nodes
are imagined: ``activation = gate``.

Every activation ``a`` also has a yin reading ``1 - a``. Only structures
that ask for it, such as a ``VOID`` plan template, use that complement.

``evaluate()`` is stateless (reset, scan, propagate, rescan).
``step()`` keeps gates and body state across frames.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict, deque
from functools import lru_cache
from enum import Enum
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np


# ============================================================
# 1. Pattern vocabulary
# ============================================================

class Orientation(Enum):
    """Axis along which bands or strips are read."""
    HORIZONTAL = 0   # left -> right
    VERTICAL   = 1   # bottom -> top


# Index 0 is the bottom band (vertical) or the left band (horizontal),
# following the Yijing bottom-to-top reading. 0 = yin, 1 = yang.
TRIGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "void":      (0,   0,   0  ),   # Kun   (earth)    all yin; pure ground
    "top":       (0,   0,   1  ),   # Gen   (mountain) figure at top
    "center":    (0,   1,   0  ),   # Kan   (water)    figure at center
    "upper":     (0,   1,   1  ),   # Xun   (wind)     majority upper
    "bottom":    (1,   0,   0  ),   # Zhen  (thunder)  figure at bottom
    "periphery": (1,   0,   1  ),   # Li    (fire)     figure at periphery
    "lower":     (1,   1,   0  ),   # Dui   (lake)     majority lower
    "universal": (1,   1,   1  ),   # Qian  (heaven)   all yang
    "neutral":   (0.5, 0.5, 0.5),   # Zhong            unknown
}

BIGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "tai_yin":   (0, 0),
    "shao_yin":  (0, 1),
    "shao_yang": (1, 0),
    "tai_yang":  (1, 1),
}

MONOGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "yin":  (0,),
    "yang": (1,),
}

ALL_PATTERNS = {**TRIGRAM_PATTERNS, **BIGRAM_PATTERNS, **MONOGRAM_PATTERNS}
PATTERN_TO_NAME: Dict[Tuple, str] = {v: k for k, v in ALL_PATTERNS.items()}
_TRIGRAM_NAMES: Dict[Tuple, str] = {v: k for k, v in TRIGRAM_PATTERNS.items()}

# Motion of dynamic trigrams: (directions, magnitudes), each given as
# (upper_half, lower_half). Directions are in {-1, 0, 1}.
TRIGRAM_MOTION: Dict[str, Tuple[Tuple, Tuple]] = {
    "void":      (( 0,  0), (0,   0  )),
    "top":       (( 1,  1), (1,   1  )),
    "center":    ((-1,  1), (1,   1  )),
    "upper":     (( 0,  1), (0,   1  )),
    "bottom":    ((-1, -1), (1,   1  )),
    "periphery": (( 1, -1), (1,   1  )),
    "lower":     ((-1,  0), (1,   0  )),
    "universal": (( 0,  0), (1,   1  )),
    "neutral":   (( 0,  0), (0.5, 0.5)),
}


# Imagined (field-free) evaluation reaches every node of a subtree on every step, so dashed
# loops and dynamic trigrams anywhere inside bodies, plans and serial nodes are live. Set to
# False to reproduce the earlier shortcut, where a plan/body copied its last child's gate
# without evaluating it, and a SerialNode without a field returned 0 without evaluating its
# children. Nothing changes when a field is present.
EVALUATE_IMAGINED_SUBTREES = True

# Temporal dynamics used by ``ConceptGraph.step()``:
#   "legacy"  the original step: bodies drift along their first dynamic trigram, and
#             imagined dashed targets update within the same step.
#   "gameboard"    a board-game grid world (see ``ConceptGraph._step_gameboard``):
#             * observers perceive the previous tick's imagery in their own grid region and are
#               not drawn. An observer is any node that is a dashed-edge source but not a dashed
#               target (or a leaf with an input window). A parent observer perceives the pattern
#               its children form, read the way its type composes them (serial strips, parallel
#               pooling, plan template, body window); its whole subtree is its perceptual pattern;
#             * every dashed effect takes one tick: gates set at tick t act from tick t+1,
#               whether the target is a leaf or a whole group;
#             * when an imagined dynamic trigram switches on (activation crosses
#               ONSET_THRESHOLD, strictly), it issues one movement order to every body whose
#               window overlaps its own region (strip or body window), wherever the body
#               sits in the tree. While it stays on, a body that enters its region (again) gets
#               one order too; a body that stays inside gets no more. An order is one body extent in whole cells,
#               in its trigram direction, carried out at step_size x activation cells per tick.
#               Orders add up; bodies move one cell at a time and stay on the grid.
DEFAULT_DYNAMICS = "gameboard"
DYNAMICS_ALIASES = {"ludo": "gameboard"}      # the gameboard's earlier name, as saved with older runs
ONSET_THRESHOLD = 0.5

# A structure node's gate scales its whole composite: its own activation, the weight its
# subtree is drawn with, and (gameboard) the effective activation of motion leaves beneath it.
# A plan template's gate keeps its separate role as the template probability p. With False,
# structure gates are ignored as before (they only mattered as plan templates).
GATE_STRUCTURES = True

# What an undrawn cell means in rendered imagery. With True, a cell nobody draws is unknown
# (0.5). Imagined yang bands pull it towards 1 and imagined yin bands towards 0 (known yin);
# a neutral bit (0.5) and a dynamic (motion) trigram draw nothing. Yang and yin paint are pooled separately by noisy-OR
# (Y, N) and the cell is 0.5 + 0.5 * (Y - N). Observers are unaffected: they score only their
# yang bands, so their yin bands act as masks. With False (the earlier renderer), undrawn
# cells are 0 and yin bands paint half strength into the same noisy-OR as yang.
UNKNOWN_BACKGROUND = True

# Where a body starts. With True, nobody specifies body positions: ``reset_gates()`` centres
# every body in the region its place in the tree gives it (its serial strip, its parent body's
# window, or the whole field), and any position passed to ``BodyNode`` is overwritten. Extent
# and step size stay parameters. With False, bodies keep the position they were built with.
TREE_PLACED_BODIES = True

# Motion runs from a dynamic trigram's figure side across to the background, on both axes:
# top/V* moves down, bottom/V* up, top/H* left, bottom/H* right. With False, horizontal motion
# keeps the earlier mirrored sign (top/H* moved right, towards its figure).
MOTION_AWAY_FROM_FIGURE = True

# A dynamic template orients its content instead of selecting part of the field. Each dynamic
# leaf in the template gives a direction (its motion), and the content's "up" is turned to
# face it: a serial template gives one oriented slot per strip; any other template (a leaf,
# parallel, plan, body) gives one slot over its region, facing the sum of its dynamic leaves'
# directions. Diagonals become stair-stepped cell bands. As for static templates, the content
# is oriented with probability p (the template's gate) and used unchanged with 1 - p. Dynamic
# leaves inside templates give orientation only: in gameboard they issue no movement orders.
# With False, a dynamic template selects the half of the field in its motion direction.
ORIENTING_TEMPLATES = True

# A program's meaning comes purely from its tree: no numbers inside it. A body fills the region
# its place in the tree gives it (its serial strip, its parent body's window, the whole field,
# or its turned frame), set by ``place_bodies``; any extent or step passed to ``BodyNode`` is
# overwritten. In gameboard a body moves at the motion leaf's gate (times its groups' gates) times
# its trigram's motion strength, in cells per tick. With False, bodies keep their extent and
# step parameters (earlier runs).
TREE_SIZED_BODIES = True

# Each motion trigram has its own kind of motion (gameboard), and motion moves mass as well as bodies.
# Mass: cells drawn by imagined leaves outside any body are particles with an accumulated
# whole-cell offset; the tree keeps drawing its pattern at the displaced positions, and vacated
# cells read as unknown. Bodies stay whole tokens. Kinds:
#   top / bottom      bulk: everything in the leaf's space moves together (a body by its own
#                     extent, mass by the space's extent)
#   upper / lower     gradient: layers leave one per tick (at full gate) from the leading edge, so
#                     mass spreads like diffusion and bodies leave in a queue
#   center            the two halves of the space (along the leaf's axis) move apart, each by
#                     half its own size (centrifugal)
#   periphery         the two halves move together, each by half its own size (centripetal)
#   universal         each particle or body gets a random push along the leaf's axis every tick
#                     while the leaf is on (seeded by leaf and tick: a program's video is fixed)
# Everything moves at the gate: one cell (or layer) per tick when it is fully open. With False,
# every motion trigram moves bodies in bulk at its motion strength, and mass does not move.
MOTION_KINDS = True

# Separate trees are separate spaces (gameboard). Each root draws in its own space; its observers
# perceive only that space, and its motion leaves move only its own bodies and mass. Things share
# a space only by being composed under a common node, which makes them one tree. Trees can still
# be linked by dashed edges. With False, all roots are drawn on one shared board (earlier runs).
SEPARATE_TREE_SPACES = True

# Copying, grouping and agency are separate operators (gameboard). ``template`` copies its material (last
# child) into its template's slots and groups nothing; it is the earlier ``plan``. ``field`` makes its one
# child a rigid unit that outside motion moves as a whole. ``body`` is a field with agency: its own dynamic
# leaves propel it every tick (gate cells per tick along the leaf's turned direction; universal pushes it at
# random; centre/periphery have no net direction). A dynamic leaf inside a field or body still moves every
# other substance overlapping its region, never its own groups. Copies of a group made by a template are
# independent instances, each with its own place and motion. With False, ``body`` is the earlier movable
# plan (templates first, material last) and ``plan`` the fixed one.
GROUPING_NODES = True

# Unknown motion is a latent, not a random path (docs/probabilistic_semantics.md). ``universal*`` says
# "there is motion here, I don't know which way", ``neutral*`` "I don't know whether there is motion here",
# ``void*`` "I know there is none". Each ``universal*`` / ``neutral*`` leaf belongs to a unit (its innermost
# field or body node, shared by the node's template copies; else the leaf itself), and a rollout fixes every
# unit's state in ``ConceptGraph.motion_states``: "a" or "b" (a bulk push one way or the other along the
# leaf's axis, as top* / bottom*) or "still" (no orders). The likelihood mixes the states: universal a / b
# with 1/2 each, neutral still 1/2, a 1/4, b 1/4. Unset units default to "a" (universal) and "still"
# (neutral). Cells that moving mass or a moving body leaves behind read as known empty (yin), so motion
# cannot be used to withdraw a claim. With False, universal* gives random pushes, neutral* nothing, and vacated cells read
# as unknown (earlier runs).
PROBABILISTIC_MOTION = True

# A tree whose top node is a dashed source would be all observer: nothing in it is drawn, so its own space is
# empty and it would read a constant. Instead it is a *measure*: it is drawn like any imagined tree, and its
# activation is the mean figure claim of its own space as drawn last tick, mean over the board of
# max(0, 2v - 1) (unknown and yin add nothing; 0.5 before the first drawing). A dashed edge from it sets its
# target's gate to that amount, an alpha the tree itself defines (a tree that is one third figure fades its
# target to 1/3). Trees with an input window keep perceiving. Every other dashed source stays an observer.
# With False, such a tree is an observer of its empty space (earlier runs).
MEASURE_TREES = True

# Pure claims. The centre-axis patterns (void, neutral, universal: every band alike) claim the substance itself in
# every band: absent, unknown, present. The other patterns contrast figure with ground and claim only their figure
# bands; their ground bands say nothing. Overlapping claims are painted general first: a claim whose cells (where a
# moved body started) lie strictly inside another claim's cells wins there; claims over the same cells, or cells
# that only partly overlap, combine (disagreement reads as unknown). Observing
# matches: void checks its bands are empty, neutral that they are unknown. With False, every band of every pattern
# claims, all claims combine, void observes nothing and neutral reads a constant (earlier runs).
PURE_CLAIMS = False

# Solid figure: a moving body cannot enter cells that the rest of its tree makes more likely figure than not (value
# above 0.75); unknown and empty cells are passable. A blocked diagonal move keeps whichever axis is free. The motion
# leaves inside a body (an agent) move that body; it moves anything else only by contact: with SOFT_EDGES its edge
# may overlap other substance, which it then pushes (_push_contacts): a rigid unit whole, free mass cell by cell.
# Bodies in turned frames and free mass are not blocked yet. With False, bodies pass through everything and an
# agent's leaves also push the mass and other bodies in their space (earlier runs).
SOLID_FIGURE = False
SOLID_THRESHOLD = 0.75

# Soft edges: how big an agent (body) is, and where it lies, inside its region is uncertain. Imagining (with
# PURE_CLAIMS), it is drawn as the average over its placements (``_placements``: sizes 1, 2/3 and 1/3, flush or
# on the thirds), so it is certain at its centre and fades towards its edge; without PURE_CLAIMS, or with units
# nested in it, it is drawn in its core (the centre third of its region on both axes). Observing, its content is
# checked in its core; moving (SOLID_FIGURE), its edge (the ring around the core) may overlap other substance, which
# it pushes. A body whose region fills its frame has no room to vary. A
# field is a rigid structure: its extent is what its content says. With False, a body fills its region exactly.
SOFT_EDGES = False


# Static fields (docs/static_fields_spec.md; needs PURE_CLAIMS). Three roles for substance: bare leaves are the
# initial condition (drawn once into a substance grid, then moved by motion like sand); a field is a boundary
# condition (its region is re-asserted every tick: substance where it claims figure, removed where it claims void,
# so it is a source or a drain, and never moves itself); a body carries its claims (moved by its own motion and by
# pushes; a body around a field is the old rigid field). Each tree's substance grid holds a value and a strength per
# cell; motion moves cell contents, contents arriving together pile up, cells left behind read known empty. Each
# tick: issue orders, move bodies, move substance, reset fields, draw. Contact moves substance outside fields only.
# A body held by a field (the nearest unit above it is a field) is part of what the field re-asserts: whenever no
# instance of it is at its home, the field spawns a new one there, so a moving body held by a field is a stream of
# bodies (a spawner). Instances of one body pass through each other; instances that come to the same place in the
# same state merge. A body beside a field, not in it, is independent.
# With False, fields are rigid units that move, and bare leaves are drawn every tick (earlier runs).
STATIC_FIELDS = False

# Standing motion (needs STATIC_FIELDS). Every motion leaf is a standing rule: every tick it gives an order to
# everything whose substance lies in its space and has no order from it under way. Orders stay discrete (the board is
# a diagram, not a simulation): substance is ordered across the space, a body by its own extent, one cell per tick,
# committed once given. Content that belongs to a body moves with that body. A body's own content is its leaves and
# its fields' (a field never moves itself; what it holds as a body is not its content). Self-motion is the case where
# the leaf's own body is in its space: the body moves, the leaf with it, and the leaf orders it again from there. So
# a body carrying a field with motion drives itself, a field moves the bodies it holds (a bubble machine), a hollow
# box is moved by the ball inside only through contact (its walls are not in the ball's space), and a body riding
# inside a moving body is moved as well as carried. With SOLID_FIGURE a push is passed along (``_chain_move``).
# With False, a motion leaf in a body drives that body only, and other motion leaves order content once, when it
# enters their space, to cross the space (earlier runs).
STANDING_MOTION = False


def _standing() -> bool:
    return STANDING_MOTION and STATIC_FIELDS and PURE_CLAIMS


def _spawned(node, ikey: tuple, instances) -> List[tuple]:
    """STATIC_FIELDS: the keys of a unit's instances at template position ``ikey``: the first, then those a field
    spawned (``ikey + ("s", n)``)."""
    n = len(ikey)
    return [ikey] + sorted((k for k in instances if len(k) == n + 2 and k[:n] == ikey and k[n] == "s"),
                           key=lambda k: k[-1])


def _soft_body(node) -> bool:
    """SOFT_EDGES: an agent whose extent is uncertain (not one carrying a field: STATIC_FIELDS)."""
    return (SOFT_EDGES and getattr(node, "agency", False)
            and not (STATIC_FIELDS and any(isinstance(n, GroupNode) for n in walk(node) if n is not node)))


def _soft_edge(node, region: Region, frame: Tuple[int, int]) -> bool:
    """SOFT_EDGES: is this an agent of uncertain extent with room in its frame?"""
    return _soft_body(node) and tuple(region) != (0, frame[0], 0, frame[1])


def unit_core(region: Region) -> Region:
    """SOFT_EDGES: the certain core of a unit's region, the centre third on both axes."""
    r0, r1, _, _ = band_region(region, 1, 3, Orientation.VERTICAL)
    _, _, c0, c1 = band_region(region, 1, 3, Orientation.HORIZONTAL)
    return r0, r1, c0, c1
_MOTION_KIND ={"top": "bulk", "bottom": "bulk", "upper": "gradient", "lower": "gradient",
                "center": "out", "periphery": "in", "universal": "random"}


def _sg(node) -> float:
    """The gate a structure applies to its composite (1 when GATE_STRUCTURES is off)."""
    return node.gate if GATE_STRUCTURES else 1.0


def _compose(yang_complement: np.ndarray, yin_complement: np.ndarray) -> np.ndarray:
    """A rendered field from the noisy-OR complements of yang and yin paint."""
    if UNKNOWN_BACKGROUND:
        return np.clip(0.5 + 0.5 * (yin_complement - yang_complement), 0.0, 1.0)
    return np.clip(1.0 - yang_complement, 0.0, 1.0)


def _covered(yang_complement: np.ndarray, yin_complement: np.ndarray, left: np.ndarray) -> np.ndarray:
    """The yin complement once emptiness left behind by motion (``left``, PROBABILISTIC_MOTION) is added
    where nothing else claims the cell: content that moves into a vacated cell covers it."""
    if left.min() >= 1.0:
        return yin_complement
    free = (yang_complement >= 1.0) & (yin_complement >= 1.0)
    return np.where(free, yin_complement * left, yin_complement)


def _compose_claims(claims: list, shape: Tuple[int, int], boundary=frozenset()) -> Tuple[np.ndarray, np.ndarray]:
    """PURE_CLAIMS: yang and yin complements of ``_claim_values``: they compose (``_compose``) to its value and
    count as claimed (``_covered``) as strongly as claims cover the cell."""
    if not claims:
        return np.ones(shape), np.ones(shape)
    v, claimed = _claim_values(claims, shape, boundary)
    u = (1.0 - np.abs(2.0 * v - 1.0)) * claimed
    return 1.0 - np.maximum(0.0, 2.0 * v - 1.0) - u, 1.0 - np.maximum(0.0, 1.0 - 2.0 * v) - u


def _claim_values(claims: list, shape: Tuple[int, int], boundary=frozenset()) -> Tuple[np.ndarray, np.ndarray]:
    """PURE_CLAIMS: each cell's value, and how strongly claims cover it, from claims
    ``(cells, value, strength, home cells, leaf, moved)``, general first; strength may vary by cell.

    A leaf's extent is the union of its claims' home cells (home: where a moved body started), over all the copies
    templates make of it: one description, wherever it is placed. In each cell, a claim's depth is the number of
    claims covering the cell whose leaf's extent strictly contains its own leaf's; a claim drawn by a body away from
    where it started is inside every claim that is not (a moved body is the most specific thing where it now is;
    under SOLID_FIGURE it only enters passable cells). Depths are painted in order: each layer moves the cell
    towards the strength-weighted mean of its claims' values (1 figure, 0 ground, 0.5 unknown) by their noisy-OR
    strength. Cells no claim covers stay unknown (0.5, claimed 0).
    """
    if not claims:
        return np.full(shape, 0.5), np.zeros(shape)
    M = np.array([c[0] for c in claims])                                     # (n, H, W) cells
    V, w = (_per_claim([c[k] for c in claims], shape) for k in (1, 2))
    extent: Dict[str, np.ndarray] = {}
    for c in claims:
        extent[c[4]] = extent[c[4]] | c[3] if c[4] in extent else c[3]
    P = np.array([extent[c[4]].ravel() for c in claims])                     # (n, H * W) the leaf's extent
    outside = (P[:, None, :] & ~P[None, :, :]).any(-1)                       # [i, j]: some of i's cells not j's
    moved = np.array([c[5] for c in claims])
    first = moved[:, None] & ~moved[None, :]                                 # [i, j]: i moved, j did not
    inside = ((~outside & outside.T) | first) & ~first.T                     # [i, j]: i is nested in j
    if boundary:                                 # STATIC_FIELDS: a field's claims outrank everything they cover
        b = np.array([c[4] in boundary for c in claims])
        over = b[:, None] & ~b[None, :]
        inside = (inside | over) & ~over.T
    depth = (inside.astype(int) @ M.reshape(len(claims), -1).astype(int)).reshape(M.shape)
    v = np.full(shape, 0.5)
    for d in np.unique(depth[M]):
        ws = w * (M & (depth == d))
        total = ws.sum(axis=0)
        s = 1.0 - np.prod(1.0 - ws, axis=0)
        t = (ws * V).sum(axis=0) / np.maximum(total, 1e-12)
        v = np.where(total > 0, (1.0 - s) * v + s * t, v)
    return v, 1.0 - np.prod(1.0 - w * M, axis=0)


def _per_claim(xs: list, shape: Tuple[int, int]) -> np.ndarray:
    """Per-claim values or strengths, (n, 1, 1) when all are scalars (broadcast alike), else (n, H, W)."""
    if all(np.isscalar(x) for x in xs):
        return np.array(xs, dtype=float)[:, None, None]
    return np.array([np.broadcast_to(x, shape) for x in xs], dtype=float)


def _placements(h: int, w: int) -> List[Tuple[int, int, int, int, float]]:
    """SOFT_EDGES: where an agent of uncertain extent may lie in an h x w region: (rows, cols, row offset, col offset,
    probability). Sizes 1, 2/3 and 1/3 of the region on both axes, 1/3 each; at 2/3 the four corners, at 1/3 the
    3 x 3 thirds, each size's positions equally likely. Flush positions keep contact with the region's sides."""
    out = []
    if h < 1 or w < 1:                           # a region clipped away at the board's edge
        return out
    for frac, n in ((1.0, 1), (2.0 / 3.0, 2), (1.0 / 3.0, 3)):
        hs, ws = min(h, max(1, round(h * frac))), min(w, max(1, round(w * frac)))
        rows = sorted({round((h - hs) * k / (n - 1)) for k in range(n)}) if n > 1 else [0]
        cols = sorted({round((w - ws) * k / (n - 1)) for k in range(n)}) if n > 1 else [0]
        for dy in rows:
            for dx in cols:
                out.append((hs, ws, dy, dx, 1.0 / (3.0 * len(rows) * len(cols))))
    return out


def _claimed_value(v: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The value claims make where they cover a cell (``_claim_values`` blends it with unknown by their strength)."""
    return np.where(c > 0, np.clip(0.5 + (v - 0.5) / np.maximum(c, 1e-12), 0.0, 1.0), 0.5)


def _moved_mask(m: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``m`` moved by (dy, dx), either sign; what leaves the board is dropped."""
    H, W = m.shape
    out = np.zeros_like(m)
    if abs(dy) < H and abs(dx) < W:
        out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def _stuff_run(S: dict, cells: np.ndarray, dy: int, dx: int, against: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """STANDING_MOTION: substance moving by (dy, dx) pushes the substance ahead of it. Returns the run (``cells`` and
    every substance cell in its way, transitively) and its blocked cells: those whose target is off the board, in a
    field's figure or in ``against`` (content pushing the other way), and those whose target is blocked. Claims of
    absence are not pushed; they are replaced."""
    against = np.zeros_like(cells) if against is None else against & ~cells
    solid = (S["s"] > 0) & (S["v"] > 0.5) & ~S["F"] & ~against
    run = cells.copy()
    while True:
        ahead = _moved_mask(run, dy, dx) & solid & ~run
        if not ahead.any():
            break
        run |= ahead
    lands = lambda m: _moved_mask(m, -dy, -dx)                # the cells whose target is in m
    blocked = run & (~lands(np.ones_like(run)) | lands((S["F"] & (S["fe"] > 0.5) | against) & ~run))
    while True:
        more = run & lands(blocked) & ~blocked
        if not more.any():
            break
        blocked |= more
    return run, blocked


def _shifted(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``a`` moved down by dy and right by dx (both >= 0), zero filled."""
    if dy == 0 and dx == 0:
        return a
    out = np.zeros_like(a)
    if dy < a.shape[0] and dx < a.shape[1]:          # moved wholly off the board: nothing left
        out[dy:, dx:] = a[:a.shape[0] - dy, :a.shape[1] - dx]
    return out


_UNKNOWN_MOTION = {TRIGRAM_PATTERNS["universal"]: "universal", TRIGRAM_PATTERNS["neutral"]: "neutral"}   # PROBABILISTIC_MOTION

def unknown_value() -> float:
    """What "not known yet" reads as: 0.5, or 0 with the earlier renderer."""
    return 0.5 if UNKNOWN_BACKGROUND else 0.0


def blank_board(shape: Tuple[int, int]) -> np.ndarray:
    """A field with nothing drawn on it: every cell unknown."""
    return np.full(shape, unknown_value())


def _mean(values) -> float:
    """``np.mean`` of a short sequence without most of its overhead, bit-for-bit the same.

    One or two values: a plain sum is exact. More: numpy may add in vectorised lanes whose order
    depends on memory layout, so use the reduction ``np.mean`` itself performs.
    """
    values = values if isinstance(values, list) else list(values)
    n = len(values)
    if n <= 2:
        return float(sum(values)) / n
    return float(np.add.reduce(np.array(values, dtype=float))) / n


def _clamp(x, lo: float = 0.0, hi: float = 1.0) -> float:
    """Scalar clip to [lo, hi]: the same as float(np.clip(x, lo, hi)), but about 10x faster."""
    x = float(x)
    return lo if x < lo else hi if x > hi else x


def trigram_symbol(pattern: tuple) -> str:
    """Trigram name for a 3-band pattern, or '' for anything else."""
    return _TRIGRAM_NAMES.get(pattern, "")


# ============================================================
# 2. Band geometry
# ============================================================
# A region is (r0, r1, c0, c1) in numpy coordinates: row 0 is the top.

Region = Tuple[int, int, int, int]


def band_region(region: Region, i: int, n: int, orientation: Orientation) -> Region:
    """Band ``i`` of ``n`` inside ``region``.

    Vertical bands count from the bottom, horizontal bands from the left.
    """
    r0, r1, c0, c1 = region
    if orientation == Orientation.VERTICAL:
        k = n - 1 - i
        return r0 + round((r1 - r0) * k / n), r0 + round((r1 - r0) * (k + 1) / n), c0, c1
    return r0, r1, c0 + round((c1 - c0) * i / n), c0 + round((c1 - c0) * (i + 1) / n)


def is_empty(region: Region) -> bool:
    r0, r1, c0, c1 = region
    return r1 <= r0 or c1 <= c0


@lru_cache(maxsize=4096)
def _orientation_map(region: Region, v: Tuple[float, float]):
    """Which cell of the content's own upright frame each cell of ``region`` shows.

    The content's up points along ``v = (dy, dx)`` (numpy axes; (-1, 0) is upright). The frame
    is the region's extent along and across ``v``; each region cell, taken at its centre, falls
    in one frame cell. Returns ``(Hc, Wc, rows, cols, (uy, ux, ry, rx))``: the frame size, the
    frame cell of every region cell, and the frame's up and right axes in the region.
    """
    r0, r1, c0, c1 = region
    h, w = r1 - r0, c1 - c0
    norm = float(np.hypot(*v))
    uy, ux = v[0] / norm, v[1] / norm
    ry, rx = ux, -uy                                   # the content's right: up turned clockwise
    Y, X = np.meshgrid(np.arange(h) + 0.5 - h / 2, np.arange(w) + 0.5 - w / 2, indexing="ij")
    Hf, Wf = abs(uy) * h + abs(ux) * w, abs(ry) * h + abs(rx) * w
    Hc, Wc = max(1, round(Hf)), max(1, round(Wf))
    rows = np.clip(np.floor((-(Y * uy + X * ux) + Hf / 2) / Hf * Hc), 0, Hc - 1).astype(int)
    cols = np.clip(np.floor((Y * ry + X * rx + Wf / 2) / Wf * Wc), 0, Wc - 1).astype(int)
    return Hc, Wc, rows, cols, (uy, ux, ry, rx)


def _orient_field(field: "ReceptiveField", region: Region, v: Tuple[float, float]) -> "ReceptiveField":
    """``field`` inside ``region`` seen in the content's upright frame (facing ``v``).

    Each frame cell averages the region cells that fall in it; a frame cell no region cell
    falls in (a corner of a diagonal frame) reads as unknown. Motion vectors are turned too.
    """
    Hc, Wc, rows, cols, (uy, ux, ry, rx) = _orientation_map(region, v)
    r0, r1, c0, c1 = region
    total, count = np.zeros((Hc, Wc)), np.zeros((Hc, Wc))
    np.add.at(total, (rows, cols), field.activation[r0:r1, c0:c1])
    np.add.at(count, (rows, cols), 1.0)
    activation = np.where(count > 0, total / np.maximum(count, 1.0), unknown_value())
    motion = None
    if field.motion is not None:
        m = field.motion[r0:r1, c0:c1]
        motion = np.zeros((Hc, Wc, 2))
        np.add.at(motion[..., 0], (rows, cols), -(m[..., 0] * uy + m[..., 1] * ux))
        np.add.at(motion[..., 1], (rows, cols), m[..., 0] * ry + m[..., 1] * rx)
        motion /= np.maximum(count, 1.0)[..., None]
    return ReceptiveField(activation, motion)     # its own root: nested bodies carve from the frame


def _view_field(board: "ReceptiveField", chain, rect: Region) -> "ReceptiveField":
    """The board seen through a view: carried into each turned frame of ``chain``, then cropped to ``rect``."""
    field = board
    for slot, v in chain:
        field = _orient_field(field, slot, v)
    return field.subfield(*rect)


def _view_mask(shape: Tuple[int, int], chain, rect: Region) -> np.ndarray:
    """The board cells a view covers: those whose place in the last frame of ``chain`` falls in ``rect``."""
    R, C = _grid_index(tuple(shape))
    ok = np.ones(shape, dtype=bool)
    for (r0, r1, c0, c1), v in chain:
        ok &= (R >= r0) & (R < r1) & (C >= c0) & (C < c1)
        _, _, rows, cols, _ = _orientation_map((r0, r1, c0, c1), v)
        R2, C2 = np.zeros_like(R), np.zeros_like(C)
        R2[ok], C2[ok] = rows[R[ok] - r0, C[ok] - c0], cols[R[ok] - r0, C[ok] - c0]
        R, C = R2, C2
    r0, r1, c0, c1 = rect
    return ok & (R >= r0) & (R < r1) & (C >= c0) & (C < c1)


def _view_turn(chain, vector: Tuple[float, float]) -> Tuple[float, float]:
    """A direction in a view's last frame, expressed on the board (each frame's turn applied outwards)."""
    fy, fx = vector
    for _, v in reversed(chain):
        norm = float(np.hypot(*v))
        uy, ux = v[0] / norm, v[1] / norm
        ry, rx = ux, -uy
        fy, fx = -fy * uy + fx * ry, -fy * ux + fx * rx    # frame rows run along -up, columns along right
    return fy, fx


def _view_unturn(chain, vector: Tuple[float, float]) -> Tuple[float, float]:
    """A direction on the board, expressed in a view's last frame (the inverse of ``_view_turn``)."""
    by, bx = vector
    for _, v in chain:
        norm = float(np.hypot(*v))
        uy, ux = v[0] / norm, v[1] / norm
        ry, rx = ux, -uy
        by, bx = -(by * uy + bx * ux), by * ry + bx * rx
    return by, bx


def _frame_shape(chain, shape: Tuple[int, int]) -> Tuple[int, int]:
    """The size in cells of a view's last frame (the board when the chain is empty)."""
    if not chain:
        return shape
    Hc, Wc, *_ = _orientation_map(*chain[-1])
    return Hc, Wc


@lru_cache(maxsize=16)
def _grid_index(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Row and column index arrays of a grid (read-only; shared)."""
    R, C = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    R.flags.writeable = C.flags.writeable = False
    return R, C


def _scatter(complement: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Move each cell's paint (a noisy-OR complement) by its whole-cell offset; overlaps combine."""
    H, W = complement.shape
    R, C = _grid_index((H, W))
    out = np.ones_like(complement)
    np.multiply.at(out, (np.clip(R + offset[..., 0], 0, H - 1), np.clip(C + offset[..., 1], 0, W - 1)), complement)
    return out


def _has_dynamic(node: "Node") -> bool:
    return any(isinstance(n, ObjectNode) and n.is_dynamic for n in walk(node))


def orient_slots(template: "Node", region: Region) -> List[Tuple[Region, Optional[Tuple[float, float]]]]:
    """Where a template puts its content, and which way the content faces there.

    Returns ``(slot, direction)`` pairs; direction None means upright. Static parts give the
    usual slots (a leaf's yang bands, a serial's strips). A dynamic leaf, or any non-serial
    structure, gives one slot over its region facing the sum of its dynamic leaves' motions.
    """
    if isinstance(template, ObjectNode) and not template.is_dynamic:
        bands = [band_region(region, i, template.n_bands, template.orientation)
                 for i, bit in enumerate(template.pattern) if bit > 0]
        return [(b, None) for b in bands if not is_empty(b)]
    if isinstance(template, SerialNode):
        kids = content_children(template)
        slots: List[Tuple[Region, Optional[Tuple[float, float]]]] = []
        for i, child in enumerate(kids):
            strip = band_region(region, i, len(kids), template.split_orientation)
            slots.extend(orient_slots(child, strip) or [(strip, None)])
        return slots
    dy = dx = 0.0
    for n in walk(template):
        if isinstance(n, ObjectNode) and n.is_dynamic:
            my, mx = BodyNode._template_motion(n)
            dy, dx = dy + my, dx + mx
    return [(region, None if abs(dy) < 1e-12 and abs(dx) < 1e-12 else (dy, dx))]


# ============================================================
# 3. Receptive field
# ============================================================

class ReceptiveField:
    """A 2-D yang-salience map in [0, 1], plus an optional motion channel.

    ``activation`` has shape (H, W). Its yin reading is ``1 - activation``.
    ``motion`` has shape (H, W, 2) holding (dy, dx); dynamic scanners read it.
    ``bounds`` gives this field's position inside its root field, normalised to [0, 1].
    """

    def __init__(
        self,
        activation: np.ndarray,
        motion: Optional[np.ndarray] = None,
        bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        root_activation: Optional[np.ndarray] = None,
        root_motion: Optional[np.ndarray] = None,
    ) -> None:
        if activation.ndim != 2:
            raise ValueError("activation must be 2-D (H, W).")
        # A band thinner than one pixel gives an empty field, which scans to 0.
        if activation.size and not (0 <= activation.min() and activation.max() <= 1 + 1e-6):
            warnings.warn("Clipping activation values to [0, 1].")
        self.activation = np.clip(activation, 0.0, 1.0)            # a new array
        if self.activation.dtype != float:
            self.activation = self.activation.astype(float)
        self.motion = motion
        self.bounds = tuple(bounds)
        # Bodies carve their window from the root, not from a strip.
        self._root_activation = self.activation if root_activation is None else root_activation
        self._root_motion = self.motion if root_motion is None else root_motion

    @classmethod
    def from_array(cls, arr: np.ndarray, motion=None) -> "ReceptiveField":
        lo, hi = arr.min(), arr.max()
        return cls(np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1), motion)

    @classmethod
    def uniform(cls, height: int, width: int, value: float = 0.5) -> "ReceptiveField":
        return cls(np.full((height, width), value))

    @classmethod
    def random(cls, height: int, width: int, seed: Optional[int] = None) -> "ReceptiveField":
        return cls(np.random.default_rng(seed).random((height, width)))

    @property
    def height(self) -> int:
        return self.activation.shape[0]

    @property
    def width(self) -> int:
        return self.activation.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.activation.shape

    @property
    def region(self) -> Region:
        return 0, self.height, 0, self.width

    @property
    def yin_activation(self) -> np.ndarray:
        return 1.0 - self.activation

    def subfield(self, row_start: int, row_end: int,
                 col_start: int, col_end: int) -> "ReceptiveField":
        """Crop a rectangle, keeping track of its bounds in the root field."""
        sub_mot = None if self.motion is None else self.motion[row_start:row_end, col_start:col_end]
        br0, br1, bc0, bc1 = self.bounds
        h, w = max(self.height, 1), max(self.width, 1)
        sub = ReceptiveField.__new__(ReceptiveField)       # the parent is already checked and clipped
        sub.activation = self.activation[row_start:row_end, col_start:col_end]
        sub.motion = sub_mot
        sub.bounds = (br0 + (br1 - br0) * row_start / h, br0 + (br1 - br0) * row_end / h,
                      bc0 + (bc1 - bc0) * col_start / w, bc0 + (bc1 - bc0) * col_end / w)
        sub._root_activation = self._root_activation
        sub._root_motion = self._root_motion
        return sub

    def band(self, i: int, n: int, orientation: Orientation) -> "ReceptiveField":
        """Band ``i`` of ``n`` (vertical: from the bottom; horizontal: from the left)."""
        return self.subfield(*band_region(self.region, i, n, orientation))

    def __repr__(self) -> str:
        return f"ReceptiveField(shape={self.shape}, mean={self.activation.mean():.3f})"


# ============================================================
# 4. Edges
# ============================================================

class EdgeType(Enum):
    SOLID  = "solid"
    DASHED = "dashed"


class _Edge:
    edge_type: EdgeType
    arrow = "->"

    def __init__(self, source: "Node", target: "Node") -> None:
        self.source = source
        self.target = target

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source.node_id!r} {self.arrow} {self.target.node_id!r})"


class SolidEdge(_Edge):
    """Composition: the parent passes (part of) its field to the child."""
    edge_type = EdgeType.SOLID


class DashedEdge(_Edge):
    """Positive conditional gate: ``target.gate = mean(source activations)``."""
    edge_type = EdgeType.DASHED
    arrow = "-->"


# ============================================================
# 5. Nodes
# ============================================================

class Node:
    """Base node: an activation in [0, 1], a gate (default 1), and solid children.

    Subclasses must define ``evaluate``. (This is a plain class rather than an
    ABC because ``isinstance`` against ABCs is several times slower, and the
    evaluator calls it constantly.)
    """

    node_type: str = "node"

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.activation: float = 0.0
        self.gate: float = 1.0
        self.children: List["Node"] = []

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        """Compute and store this node's activation. ``field`` is None when there is no field context."""
        raise NotImplementedError(f"{type(self).__name__} must define evaluate()")

    @property
    def yin_activation(self) -> float:
        return 1.0 - self.activation

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.node_id!r}, act={self.activation:.3f})"


def walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal of a solid-edge subtree."""
    yield node
    for child in node.children:
        yield from walk(child)


# ---- Portals --------------------------------------------------------------

class InputNode(Node):
    """Sensory window. A ``ParallelNode`` that owns one passes its field to sibling scanners.

    Its gate is the modality's presence weight, used when fields from
    several modalities are pooled. A dashed edge can set it (attention).
    """

    node_type = "input"
    modality: str = "generic"

    def __init__(self, node_id: str, field: Optional[ReceptiveField] = None,
                 modality: Optional[str] = None, initial_weight: float = 1.0) -> None:
        super().__init__(node_id)
        self.field = field
        self.gate = _clamp(initial_weight)
        self.activation = self.gate
        if modality is not None:
            self.modality = modality

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        return self.gate

    def __repr__(self) -> str:
        info = f"shape={self.field.shape}" if self.field else "no field"
        return f"InputNode(id={self.node_id!r}, modality={self.modality!r}, gate={self.gate:.3f}, {info})"


class RetinaNode(InputNode):
    modality = "retina"


class TactileNode(InputNode):
    modality = "tactile"


class ProprioNode(InputNode):
    modality = "proprio"


class OutputNode(Node):
    """Output window. Its parent ``ParallelNode`` writes its result into this node.

    ``ConceptGraph.evaluate()`` returns the activation of the last output node.
    """

    node_type = "return"   # serialised name kept for the visualizer

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        return self.activation


PORTALS = (InputNode, OutputNode)


class WindowNode(Node):
    """An output window: a leaf that reads whatever is drawn in the region its place in the tree gives it.

    Like an input window it holds no pattern. It draws nothing, scans nothing and takes no dashed
    edges. A program's output channels are its windows: channel k at each tick is the board inside
    window k's region (its strip, its body's window, or its turned frame), unknown elsewhere. If what
    was drawn there moves away, the channel goes unknown.
    """

    node_type = "out"

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        self.activation = 0.0
        return 0.0


class InWindowNode(Node):
    """An input window: the mirror image of an output window. A leaf that shows an external field.

    Its region comes from its place in the tree (a strip, a body's window, a turned frame), and it
    writes its field into its tree's space there, resized to the window's cells. Values read on the
    board's scale: 1 figure, 0 known empty, 0.5 unknown. Its gate weights what it writes. Observers
    whose region covers it perceive it like anything drawn. It scans nothing and takes no dashed
    edges. Set its data with ``ConceptGraph.set_window_input``. (``InputNode`` is the earlier portal, kept
    for the original demos and learners.)
    """

    node_type = "in"

    def __init__(self, node_id: str, field: Optional[np.ndarray] = None) -> None:
        super().__init__(node_id)
        self.field = field

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        self.activation = 0.0
        return 0.0


_WINDOW_LEAVES = (WindowNode, InWindowNode)       # read or write a region; never scanned or averaged


def content_children(node: Node) -> List[Node]:
    """A node's children, without its input/output windows."""
    return [c for c in node.children if not isinstance(c, PORTALS)]


# ---- Scanner --------------------------------------------------------------

class ObjectNode(Node):
    """Trigram scanner: an image schema over ``len(pattern)`` bands.

    Only yang bands (pattern > 0) contribute to the match:

        activation = gate * mean(1 - |band_mean_i - pattern_i|  for yang i)

    Yin bands contribute no positive term, so ``VOID`` scans to 0. With no
    field the node is imagined and ``activation = gate``. A neutral
    (all 0.5) pattern returns ``0.5 * gate``. With ``is_dynamic=True`` it
    scans |motion| along its axis instead of salience.
    """

    node_type = "object"

    def __init__(self, node_id: str, pattern: Tuple[float, ...],
                 orientation: Orientation = Orientation.VERTICAL,
                 is_dynamic: bool = False, name: Optional[str] = None) -> None:
        super().__init__(node_id)
        self.pattern = tuple(pattern)
        self.orientation = orientation
        self.is_dynamic = is_dynamic
        self.name = name or PATTERN_TO_NAME.get(self.pattern, str(pattern))

    @property
    def n_bands(self) -> int:
        return len(self.pattern)

    def _band_means(self, field: ReceptiveField) -> List[float]:
        axis = 0 if self.orientation == Orientation.VERTICAL else 1   # dy or dx
        means = []
        for i in range(self.n_bands):
            r0, r1, c0, c1 = band_region(field.region, i, self.n_bands, self.orientation)
            values = field.activation[r0:r1, c0:c1]
            if self.is_dynamic and field.motion is not None:
                values = np.abs(field.motion[r0:r1, c0:c1, axis])
            means.append(float(values.sum()) / values.size if values.size > 0 else 0.0)   # == values.mean()
        return means

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if field is None:
            self.activation = self.gate
        elif PURE_CLAIMS and len(set(self.pattern)) == 1:      # centre axis: the substance in every band
            target = self.pattern[0]
            sims = [1.0 - 2.0 * abs(obs - 0.5) if target == 0.5 else 1.0 - abs(obs - target)
                    for obs in self._band_means(field)]
            self.activation = self.gate * _mean(sims)
        elif all(v == 0.5 for v in self.pattern):
            self.activation = 0.5 * self.gate
        else:
            sims = [1.0 - abs(obs - exp)
                    for obs, exp in zip(self._band_means(field), self.pattern) if exp > 0]
            self.activation = self.gate * _mean(sims) if sims else 0.0
        return self.activation

    def __repr__(self) -> str:
        dyn = "dynamic" if self.is_dynamic else "static"
        ori = "V" if self.orientation == Orientation.VERTICAL else "H"
        return f"ObjectNode(id={self.node_id!r}, {trigram_symbol(self.pattern)}:{self.name}:{dyn} {ori})"


# ---- Structure nodes ------------------------------------------------------

class StructureNode(Node):
    """A node that combines its children's activations."""


class ParallelNode(StructureNode):
    """Attention hub and averaging compositor.

    It pools fields from its ``InputNode`` children (weighted by their gate)
    and from its parent (weighted by ``inherited_weight``). It then scores
    each scanner child on the pooled fields:

        scanner_score = sum_f(w_f * scanner.evaluate(f)) / sum_f(w_f)
        activation    = mean(scanner_score)

    With no field, children are evaluated field-free (imagined) and the
    mean of their non-zero activations is used. The result is copied into
    every ``OutputNode`` child.
    """

    node_type = "parallel"

    def __init__(self, node_id: str, inherited_weight: float = 1.0) -> None:
        super().__init__(node_id)
        self.inherited_weight = float(inherited_weight)

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        inputs   = [c for c in self.children if isinstance(c, InputNode)]
        outputs  = [c for c in self.children if isinstance(c, OutputNode)]
        scanners = [c for c in content_children(self) if not isinstance(c, _WINDOW_LEAVES)]   # windows only read/write

        pooled = [(n.field, n.gate) for n in inputs if n.field is not None and n.gate > 1e-8]
        if field is not None and self.inherited_weight > 1e-8:
            pooled.append((field, self.inherited_weight))

        if not pooled:
            acts = [a for a in (s.evaluate(None) for s in scanners) if a > 0]
            result = _mean(acts) if acts else 0.0
        elif not scanners:
            result = _clamp(sum(w for _, w in pooled) / len(pooled))
        else:
            total_w = sum(w for _, w in pooled)
            scores = [sum(w * s.evaluate(f) for f, w in pooled) / total_w for s in scanners]
            result = _clamp(_mean(scores))

        result *= _sg(self)
        for out in outputs:
            out.gate, out.activation = 1.0, result
        self.activation = result
        return result


class SerialNode(StructureNode):
    """Splits the field into one strip per child and averages the children's scores.

    The mean is intentional: a partial match still counts. Strips are read
    like bands, so for vertical splits children[0] is the bottom strip and
    for horizontal splits children[0] is the left strip. "A above B" is
    therefore ``[B, A]`` with a vertical split. With no field (imagined), every child is
    evaluated field-free and the activation is their mean (see EVALUATE_IMAGINED_SUBTREES).
    """

    node_type = "serial"

    def __init__(self, node_id: str,
                 split_orientation: Orientation = Orientation.HORIZONTAL) -> None:
        super().__init__(node_id)
        self.split_orientation = split_orientation

    def _make_strip(self, field: ReceptiveField, i: int, n: int) -> ReceptiveField:
        return field.band(i, n, self.split_orientation)

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if field is None and self.children and EVALUATE_IMAGINED_SUBTREES:
            # Imagined: no field to split into strips; evaluate every child and average.
            kids = [c for c in self.children if not isinstance(c, _WINDOW_LEAVES)]
            self.activation = (_mean([child.evaluate(None) for child in kids]) if kids else 0.0) * _sg(self)
            return self.activation
        if not self.children or field is None:
            self.activation = 0.0
            return 0.0
        n = len(self.children)
        scores = [child.evaluate(self._make_strip(field, i, n)) for i, child in enumerate(self.children)
                  if not isinstance(child, _WINDOW_LEAVES)]            # a window keeps its strip but only reads/writes
        self.activation = (_mean(scores) if scores else 0.0) * _sg(self)
        return self.activation


class BodyNode(StructureNode):
    """A plan that occupies a moving window of the field.

    Coordinates are normalised to [0, 1]: position = (y, x) centre,
    extent = (h, w), velocity = (dy, dx). Each evaluation carves the window
    from the root field and applies the replacement rule. Children are
    ``[template, ..., content]``, and for each template with p = template.gate:

        activation = p * mean(content on template's yang bands)
                   + (1 - p) * content on the whole window

    With no field this becomes ``(1 - p) * content``, where the content is
    evaluated field-free (an imagined leaf returns its gate). This is how a
    ``VOID`` template blocks its content (the explicit yin reading).

    ``ConceptGraph.step()`` then calls ``update_motion()``. That moves the
    body along the motion of its first dynamic trigram, scaled by that
    trigram's activation and by ``step_size`` (the default is one extent).
    """

    node_type = "body"

    def __init__(self, node_id: str,
                 position: Optional[Tuple[float, float]] = None,
                 extent: Optional[Tuple[float, float]] = None,
                 velocity: Tuple[float, float] = (0.0, 0.0),
                 step_size: Optional[float] = None,
                 subfield_shape: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(node_id)
        self.position = list(position) if position is not None else None
        self.extent = list(extent) if extent is not None else None
        self.velocity = list(velocity)
        self.step_size = None if step_size is None else float(step_size)
        self.subfield_shape = subfield_shape   # used by PlanNode only
        self.orders: List[list] = []             # gameboard: [axis, direction, cells left, rate, credit, leaf id, delay]
        self.jitter = [0.0, 0.0]                 # gameboard: accumulated random pushes (universal), per axis
        self._grid: Optional[np.ndarray] = None

    # -- window --------------------------------------------------------

    def _ensure_support_from_field(self, field: ReceptiveField) -> None:
        """If position or extent is unset, take them from the incoming field's bounds."""
        if self.position is not None and self.extent is not None:
            return
        r0, r1, c0, c1 = field.bounds
        self.position = [(r0 + r1) / 2.0, (c0 + c1) / 2.0]
        self.extent = [max(r1 - r0, 1e-9), max(c1 - c0, 1e-9)]

    def _carve_subfield(self, field: Optional[ReceptiveField]) -> Optional[ReceptiveField]:
        """Cut the body's window from the root field. The window is clamped to the field edges."""
        if field is None:
            return None
        self._ensure_support_from_field(field)
        root = ReceptiveField(field._root_activation, field._root_motion)
        return root.subfield(*self.window(*root.shape))

    def window(self, H: int, W: int) -> Region:
        """The body's rectangle in an H x W root field, clamped to the field edges."""
        cy, cx = self.position
        eh, ew = self.extent
        h_px, w_px = max(1, round(eh * H)), max(1, round(ew * W))
        if TREE_SIZED_BODIES:                  # position is the exact centre: the window starts at centre - size/2
            r0, c0 = max(0, round(cy * H - h_px / 2)), max(0, round(cx * W - w_px / 2))
        else:
            r0 = max(0, round(cy * H) - h_px // 2)
            c0 = max(0, round(cx * W) - w_px // 2)
        r1, c1 = min(H, r0 + h_px), min(W, c0 + w_px)
        if r1 <= r0: r1 = min(H, r0 + 1)
        if c1 <= c0: c1 = min(W, c0 + 1)
        return r0, r1, c0, c1

    # -- motion --------------------------------------------------------

    @staticmethod
    def _template_motion(template: ObjectNode) -> Tuple[float, float]:
        """(dy, dx) of a dynamic trigram: the mean of its signed half-motions along its axis.

        Opposite half-motions cancel out. Numpy coordinates increase downward
        and to the right.
        """
        directions, magnitudes = TRIGRAM_MOTION.get(
            PATTERN_TO_NAME.get(template.pattern), ((0, 0), (0, 0)))
        motion = _mean([float(d) * float(m) for d, m in zip(directions, magnitudes)])
        if template.orientation == Orientation.VERTICAL:
            return motion, 0.0                 # bands count up, rows count down: away from the figure
        return 0.0, (-motion if MOTION_AWAY_FROM_FIGURE else motion)

    def update_motion(self) -> None:
        """Set the velocity from the first dynamic trigram, then move and clamp the centre inside the field."""
        template = next((n for n in walk(self)
                         if isinstance(n, ObjectNode) and n.is_dynamic), None)
        if template is not None and PATTERN_TO_NAME.get(template.pattern) in TRIGRAM_MOTION:
            dy, dx = self._template_motion(template)
            if self.step_size is not None:
                step = self.step_size
            elif self.extent is None:
                step = 0.0
            else:
                step = self.extent[0] if abs(dy) >= abs(dx) else self.extent[1]
            scale = template.activation * step
            self.velocity = [dy * scale, dx * scale]

        if self.position is None or self.extent is None:
            return
        for axis in (0, 1):
            half = self.extent[axis] / 2
            self.position[axis] = _clamp(
                self.position[axis] + self.velocity[axis], half, 1.0 - half)

    # -- replacement rule ----------------------------------------------

    def _yang_subfields(self, template: ObjectNode, field: ReceptiveField) -> List[ReceptiveField]:
        """The sub-fields under the template's yang bands. Empty for an all-yin template."""
        n = template.n_bands
        return [field.band(i, n, template.orientation)
                for i, bit in enumerate(template.pattern) if bit > 0]

    def _motion_subfield(self, template: ObjectNode, field: ReceptiveField) -> ReceptiveField:
        """The half of the field lying in a dynamic template's motion direction."""
        dy, dx = self._template_motion(template)
        H, W = field.shape
        if dy > 0:   return field.subfield(H // 2, H, 0, W)
        if dy < 0:   return field.subfield(0, H // 2, 0, W)
        if dx > 0:   return field.subfield(0, H, W // 2, W)
        if dx < 0:   return field.subfield(0, H, 0, W // 2)
        return field

    def _plan_activate(self, children: List[Node], field: Optional[ReceptiveField]) -> float:
        """Apply the replacement rule recursively: ``children[0]`` is a template and the rest is the plan's content."""
        if len(children) == 1:
            if field is None and not EVALUATE_IMAGINED_SUBTREES:
                children[0].activation = children[0].gate
                return children[0].gate
            return children[0].evaluate(field)      # imagined leaves return their gate

        template, rest = children[0], children[1:]
        p = _clamp(template.gate)
        template.evaluate(field)
        orienting = ORIENTING_TEMPLATES and _has_dynamic(template)

        if field is None:
            if orienting:
                # Imagined: turning the content does not change it, so both terms are the content.
                self.activation = self._plan_activate(rest, None)
                return self.activation
            # Imagined: there is no region to scan, so the yang term is 0.
            self.activation = (1.0 - p) * self._plan_activate(rest, None)
            return self.activation

        if orienting:
            subs = [field.subfield(*slot) if v is None else _orient_field(field, slot, v)
                    for slot, v in orient_slots(template, field.region) if not is_empty(slot)]
        elif not isinstance(template, ObjectNode):
            subs = [field]
        elif template.is_dynamic:
            subs = [self._motion_subfield(template, field)]
        else:
            subs = self._yang_subfields(template, field)

        yang_scan = _mean([self._plan_activate(rest, s) for s in subs]) if subs else 0.0
        full_scan = self._plan_activate(rest, field)
        return p * yang_scan + (1.0 - p) * full_scan

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if not self.children:
            self.activation = 0.0
            return 0.0
        window = self._carve_subfield(field)
        if self._grid is None and window is not None:
            self._grid = np.zeros(window.shape)
        self.activation = self._plan_activate(self.children, window) * _sg(self)
        if window is not None and isinstance(self.children[-1], ObjectNode):
            self._update_grid(self.children[-1])
        return self.activation

    def _update_grid(self, content: ObjectNode) -> None:
        """Paint the content's pattern into ``_grid`` (a record of the imagined layout)."""
        H, W = self._grid.shape
        for i, bit in enumerate(content.pattern):
            r0, r1, c0, c1 = band_region((0, H, 0, W), i, content.n_bands, content.orientation)
            self._grid[r0:r1, c0:c1] = bit


class PlanNode(BodyNode):
    """A body that never moves. It works on the whole field, or on a fixed ``subfield_shape`` crop."""

    node_type = "plan"

    def __init__(self, node_id: str, subfield_shape: Optional[Tuple[int, int]] = None,
                 **_ignored) -> None:
        super().__init__(node_id, position=(0.5, 0.5), extent=(1.0, 1.0),
                         subfield_shape=subfield_shape)

    def _carve_subfield(self, field):
        if field is None or self.subfield_shape is None:
            return field
        H, W = self.subfield_shape
        return field.subfield(0, min(H, field.height), 0, min(W, field.width))

    def update_motion(self) -> None:
        pass


class TemplateNode(PlanNode):
    """Copies its material (last child) into the slots of its templates; groups nothing (GROUPING_NODES)."""

    node_type = "template"


class _AsMotion:
    """A leaf seen with another pattern: how an unknown-motion leaf acts in its chosen state."""

    def __init__(self, node: "ObjectNode", pattern: Tuple[float, ...]) -> None:
        self._node, self.pattern = node, pattern

    def __getattr__(self, name):
        return getattr(self._node, name)


class GroupToken:
    """One instance of a field or body: a rigid unit with its own place, orders and drive."""

    window = BodyNode.window

    def __init__(self, node: "GroupNode", key: tuple) -> None:
        self.node, self.key = node, key
        self.node_id = node.node_id if not key else f"{node.node_id}@{'.'.join(map(str, key))}"
        self.position: Optional[List[float]] = None
        self.extent: Optional[List[float]] = None
        self.velocity, self.step_size = [0.0, 0.0], None
        self.orders: List[list] = []
        self.jitter = [0.0, 0.0]
        self.drive = [0.0, 0.0]                  # agency: accumulated self-propulsion, per axis
        self.anchor: Optional[Tuple[int, int]] = None   # where its region started when placed (frame cells)
        self.home: Optional[List[float]] = None          # where it was placed (STATIC_FIELDS: where a field spawns it)

    def rect(self, region: Region, fh: int, fw: int) -> Region:
        """Its window, carried along with the region it receives: a nested instance follows its parent."""
        r0, r1, c0, c1 = self.window(fh, fw)
        if self.anchor is not None:
            dr, dc = region[0] - self.anchor[0], region[2] - self.anchor[1]
            r0, r1, c0, c1 = r0 + dr, r1 + dr, c0 + dc, c1 + dc
        return max(0, r0), min(fh, r1), max(0, c0), min(fw, c1)


class GroupNode(BodyNode):
    """A rigid unit over its child (GROUPING_NODES). Each copy a template makes of it is a separate instance."""

    agency = False

    def __init__(self, node_id: str, **_ignored) -> None:
        super().__init__(node_id)
        self.instances: Dict[tuple, GroupToken] = {}

    def token(self, key: tuple) -> GroupToken:
        t = self.instances.get(key)
        if t is None:
            t = self.instances[key] = GroupToken(self, key)
        return t

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        kids = [c for c in content_children(self) if not isinstance(c, _WINDOW_LEAVES)]
        if not kids:
            result = 0.0
        elif field is None:
            acts = [a for a in (k.evaluate(None) for k in kids) if a > 0]
            result = _mean(acts) if acts else 0.0
        else:
            if _soft_body(self) and tuple(field.bounds) != (0.0, 1.0, 0.0, 1.0):   # check the core
                field = field.subfield(*unit_core(field.region))
            result = _clamp(_mean([k.evaluate(field) for k in kids]))
        self.activation = result * _sg(self)
        return self.activation

    def update_motion(self) -> None:
        pass


class FieldNode(GroupNode):
    """A rigid unit without agency: only outside motion moves it."""

    node_type = "field"


class AgentBodyNode(GroupNode):
    """A rigid unit with agency: its own dynamic leaves propel it."""

    node_type = "body"
    agency = True


# ============================================================
# 6. Concept graph
# ============================================================

class ConceptGraph:
    """A concept: nodes joined by solid (composition) and dashed (gate) edges.

    A graph may have several solid-edge roots. The usual pattern is an
    observed root ``ParallelNode(I, scanners)`` and an imagined root
    ``ParallelNode(O, structure)``, joined only by dashed edges.

    Evaluation:
        evaluate(): reset gates -> scan -> propagate dashed -> scan
        step():     scan -> propagate dashed -> move bodies   (stateful)
    """

    @property
    def dynamics(self) -> str:
        return self._dynamics

    @dynamics.setter
    def dynamics(self, value: str) -> None:
        self._dynamics = DYNAMICS_ALIASES.get(value, value)

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: Dict[str, Node] = {}
        self._solid: List[SolidEdge] = []
        self._dashed: List[DashedEdge] = []
        self.root: Optional[Node] = None   # primary root; the default target of add_field_slot()
        self.dynamics = DEFAULT_DYNAMICS   # "legacy" or "gameboard" (see the module constant)
        self.grid: Tuple[int, int] = (12, 12)   # gameboard: the board, in cells
        self._imagery: Optional[np.ndarray] = None
        self._prev_act: Dict[str, float] = {}
        self._inside: Dict[str, set] = {}        # gameboard: bodies inside each switched-on motion leaf's region
        self._mass: Dict[str, dict] = {}         # gameboard: each root's mass particles (MOTION_KINDS)
        self._tick = 0
        self._trails: Dict[tuple, np.ndarray] = {}   # probabilistic motion: each moved body's strongest claim per cell so far
        self._version = 0                        # bumped by every structural change (metadata cache key)
        self._meta_key = None
        self._meta_cache: Dict[str, object] = {}

    def _meta(self, name: str, build):
        """Structure-derived metadata (roots, parents, roles ...), rebuilt only after the structure changes."""
        key = (self._version, len(self._nodes), len(self._solid), len(self._dashed), id(self._solid), id(self._dashed))
        if key != self._meta_key:
            self._meta_key, self._meta_cache = key, {}
        if name not in self._meta_cache:
            self._meta_cache[name] = build()
        return self._meta_cache[name]

    # -- construction / access ----------------------------------------

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.node_id!r}.")
        self._nodes[node.node_id] = node
        self._version += 1

    def add_nodes(self, nodes: List[Node]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_solid_edge(self, parent: Node, child: Node) -> None:
        self._solid.append(SolidEdge(parent, child))
        if child not in parent.children:
            parent.children.append(child)
        self._version += 1

    def connect(self, parent: Node, *children: Node) -> Node:
        """Add solid edges from ``parent`` to each child, adding any node that is new. Returns ``parent``.

        ``g.root = g.connect(ParallelNode("par"), InputNode("I"), scanner)``
        """
        for node in (parent, *children):
            if self._nodes.get(node.node_id) is not node:
                self.add_node(node)
        for child in children:
            self.add_solid_edge(parent, child)
        return parent

    def add_dashed_edge(self, condition: Node, target: Node) -> None:
        """``target.gate`` becomes the mean activation of all its dashed sources.

        Under ``step()`` the gate applies on the next frame, so a chain
        A -> B -> C plays out over t, t+1, t+2. ``evaluate()`` rescans once
        within the same call.
        """
        self._dashed.append(DashedEdge(condition, target))
        self._version += 1

    def remove_node(self, node: Node) -> None:
        """Delete a node, its edges, and its place in any parent's children."""
        self._solid = [e for e in self._solid if node not in (e.source, e.target)]
        self._dashed = [e for e in self._dashed if node not in (e.source, e.target)]
        for other in self._nodes.values():
            if node in other.children:
                other.children.remove(node)
        del self._nodes[node.node_id]
        self._version += 1

    def replace_node(self, old: Node, new: Node) -> None:
        """Put ``new`` in place of ``old`` (same id), keeping all of old's edges."""
        new.children = old.children
        self._nodes[old.node_id] = new
        self._version += 1
        for edge in self._solid + self._dashed:
            if edge.source is old: edge.source = new
            if edge.target is old: edge.target = new
        for node in self._nodes.values():
            node.children = [new if c is old else c for c in node.children]
        if self.root is old:
            self.root = new
        self._version += 1

    def __getitem__(self, node_id: str) -> Node:
        return self._nodes[node_id]

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def solid_edges(self) -> List[SolidEdge]:
        return list(self._solid)

    @property
    def dashed_edges(self) -> List[DashedEdge]:
        return list(self._dashed)

    @property
    def roots(self) -> List[Node]:
        """Nodes with no solid-edge parent, in insertion order."""
        def build():
            has_parent = {e.target.node_id for e in self._solid}
            return [n for nid, n in self._nodes.items() if nid not in has_parent]
        return list(self._meta("roots", build))

    def _parents(self) -> Dict[str, List[Node]]:
        def build():
            parents: Dict[str, List[Node]] = defaultdict(list)
            for edge in self._solid:
                parents[edge.target.node_id].append(edge.source)
            return parents
        return self._meta("parents", build)

    def _has_input_context(self, node: Node, parents: Optional[Dict[str, List[Node]]] = None) -> bool:
        """True if ``node`` is observed: some ancestor ``ParallelNode`` owns an ``InputNode``.

        Memoised per node until the graph's structure changes.
        """
        memo = self._meta("input_context", dict)
        if node.node_id not in memo:
            memo[node.node_id] = self._input_context(node, self._parents())
        return memo[node.node_id]

    def _input_context(self, node: Node, parents: Dict[str, List[Node]]) -> bool:
        seen: Set[str] = set()
        stack = list(parents.get(node.node_id, []))
        while stack:
            parent = stack.pop()
            if parent.node_id in seen:
                continue
            seen.add(parent.node_id)
            if isinstance(parent, ParallelNode) and any(isinstance(c, InputNode) for c in parent.children):
                return True
            stack.extend(parents.get(parent.node_id, []))
        return False

    # -- evaluation ----------------------------------------------------

    def reset_gates(self) -> None:
        """Set every gate back to 1. ``evaluate()`` calls this itself and ``step()`` does not.

        With ``TREE_PLACED_BODIES``, every body is first centred in its tree region (``place_bodies``).
        In gameboard dynamics, dashed targets start unknown (gate ``unknown_value()``, 0.5: not yet
        conditioned), and pending
        movement orders, onset memory and imagery are cleared.
        """
        self._trails = {}
        for node in self._nodes.values():
            node.gate = 1.0
            if isinstance(node, PORTALS):
                node.activation = 1.0
            if isinstance(node, GroupNode):
                node.instances = {}
        if TREE_PLACED_BODIES:
            self.place_bodies()
        if self.dynamics == "gameboard":
            for edge in self._dashed:
                edge.target.gate = unknown_value()
                if not isinstance(edge.target, PORTALS):
                    edge.target.activation = unknown_value()
            for node in self._nodes.values():
                if isinstance(node, BodyNode):
                    node.orders, node.jitter = [], [0.0, 0.0]
            for tok in self._group_tokens():
                tok.orders, tok.jitter, tok.drive = [], [0.0, 0.0], [0.0, 0.0]
            self._mass, self._tick, self._spaces = {}, 0, {}
            self._sub = {}                          # STATIC_FIELDS: each tree's substance grid
            self._prev_act = {}
            self._inside = {}
            self._imagery = None

    def _dashed_topological_order(self) -> List[Node]:
        in_degree: Dict[str, int] = defaultdict(int)
        successors: Dict[str, List[Node]] = defaultdict(list)
        for edge in self._dashed:
            in_degree[edge.target.node_id] += 1
            successors[edge.source.node_id].append(edge.target)

        queue = deque(nid for nid in self._nodes if in_degree[nid] == 0)
        order: List[Node] = []
        while queue:
            nid = queue.popleft()
            order.append(self._nodes[nid])
            for succ in successors[nid]:
                in_degree[succ.node_id] -= 1
                if in_degree[succ.node_id] == 0:
                    queue.append(succ.node_id)
        seen = {n.node_id for n in order}          # nodes in dashed cycles go last
        return order + [n for nid, n in self._nodes.items() if nid not in seen]

    def _scan(self) -> float:
        """Evaluate every root. Return the last output node's activation, or else the last root's."""
        roots = self.roots
        if not roots:
            raise RuntimeError("ConceptGraph has no nodes.")
        for r in roots:
            r.evaluate(None)
        outputs = [n for n in self._nodes.values() if isinstance(n, OutputNode)]
        last = outputs[-1] if outputs else roots[-1]
        return _clamp(last.activation)

    def _propagate_dashed(self) -> None:
        """Set each dashed target's gate, in topological order.

        Portals and imagined (field-free) object nodes copy the new gate
        into their activation straight away, so imagined chains propagate in
        one pass. Observed nodes wait for the next scan.
        """
        if not self._dashed:
            return
        sources: Dict[str, List[Node]] = defaultdict(list)
        for edge in self._dashed:
            sources[edge.target.node_id].append(edge.source)
        parents = self._parents()
        for node in self._dashed_topological_order():
            srcs = sources.get(node.node_id)
            if not srcs:
                continue
            node.gate = _clamp(_mean([s.activation for s in srcs]))
            if isinstance(node, PORTALS) or (
                    isinstance(node, ObjectNode) and not self._has_input_context(node, parents)):
                node.activation = node.gate

    def _update_bodies(self) -> None:
        for node in self._nodes.values():
            if isinstance(node, BodyNode):
                node.update_motion()

    def evaluate(self) -> float:
        """Stateless: reset gates, scan, propagate dashed gates, scan again."""
        if not self._nodes:
            raise RuntimeError("ConceptGraph has no nodes.")
        self.reset_gates()
        self._scan()
        self._propagate_dashed()
        return self._scan()

    def step(self, field: Optional[ReceptiveField] = None, at_node_id: Optional[str] = None,
             modality: str = "retina", detach_after: bool = True) -> float:
        """One timestep: scan, propagate dashed gates, move bodies. Gates carry over.

        If ``field`` is given, it is shown on the matching InputNode for this
        step only. Call ``reset_gates()`` to replay from t=0.
        """
        restore = None
        if field is not None:
            node = self._find_input_node(modality, at_node_id=at_node_id)
            restore = (node, node.field)
            node.field = field
        if self.dynamics == "gameboard":
            result = self._step_gameboard()
        else:
            result = self._scan()
            self._propagate_dashed()
            self._update_bodies()
        if restore is not None and detach_after:
            restore[0].field = restore[1]
        return result

    # -- gameboard dynamics: a board-game grid world ------------------------

    def assign_roles(self) -> None:
        """Gameboard: mark every node as observing (perceives, not drawn) or imagined (drawn).

        Observer roots are dashed sources that are not dashed targets (any node type), and leaves
        with an input window. Every node in an observer root's subtree belongs to its perceptual
        pattern. ``self._observer_roots`` keeps the top-most ones, which perceive directly.
        """
        if self._meta("roles_done", lambda: [False])[0]:
            return                                        # roles depend only on structure
        self._meta_cache["roles_done"] = [True]
        sources = {e.source.node_id for e in self._dashed}
        targets = {e.target.node_id for e in self._dashed}
        parents = self._parents()
        roots = [n for n in self._nodes.values() if not isinstance(n, PORTALS) and (
            (n.node_id in sources and n.node_id not in targets)
            or (isinstance(n, ObjectNode) and self._has_input_context(n, parents)))]
        self._measures = []
        if MEASURE_TREES:                                 # a whole tree with nothing to perceive measures itself
            tops = {c.node_id for g in self.roots for c in content_children(g)}
            self._measures = [r for r in roots if r.node_id in tops
                              and not any(isinstance(n, InWindowNode) for n in walk(r))]
            roots = [r for r in roots if r not in self._measures]
        observing = set()
        for r in roots:
            observing.update(n.node_id for n in walk(r))
        for node in self._nodes.values():
            node.role = "observer" if node.node_id in observing else "imagined"
        inside = set()
        for r in roots:
            inside.update(n.node_id for n in walk(r) if n is not r)
        self._observer_roots = [r for r in roots if r.node_id not in inside]

    def place_bodies(self) -> None:
        """Centre every body in the region its place in the tree gives it (on ``self.grid``).

        A body inside turned content is placed in its turned frame (``node_views``).
        """
        self.node_views(self.grid, place=True)

    def _view_inputs(self) -> tuple:
        """Everything node views depend on besides structure: body placement, template gates, flags."""
        def build():
            bodies = [n for n in self._nodes.values() if isinstance(n, BodyNode) and not isinstance(n, PlanNode)]
            templates = [t for n in self._nodes.values() if isinstance(n, BodyNode)
                         for t in content_children(n)[:-1]]
            return bodies, templates
        bodies, templates = self._meta("view_inputs", build)
        tokens = tuple((t.node_id, None if t.position is None else tuple(t.position)) for t in self._group_tokens())
        return (tuple((None if b.position is None else tuple(b.position), None if b.extent is None else tuple(b.extent))
                      for b in bodies),
                tuple(t.gate for t in templates), ORIENTING_TEMPLATES, TREE_SIZED_BODIES, MOTION_AWAY_FROM_FIGURE, tokens)

    def _group_tokens(self) -> List["GroupToken"]:
        """Every instance of every field and body (GROUPING_NODES)."""
        return [t for n in self._meta("group_nodes", lambda: [n for n in self._nodes.values() if isinstance(n, GroupNode)])
                for t in n.instances.values()]

    def _body_nodes(self) -> List["BodyNode"]:
        """Every BodyNode (plans, fields and bodies included), in node order: the nodes motion and drawing visit."""
        return self._meta("body_nodes", lambda: [n for n in self._nodes.values() if isinstance(n, BodyNode)])

    def _tokens(self) -> Dict[str, object]:
        """Every moving unit by id: earlier bodies (the node itself) and group instances."""
        out = {}
        for n in self._body_nodes():
            if isinstance(n, GroupNode):
                for t in n.instances.values():
                    out[t.node_id] = t
            elif isinstance(n, BodyNode) and not isinstance(n, PlanNode):
                out[n.node_id] = n
        return out

    def view_meta(self, nid: str) -> List[tuple]:
        """Per view of a node (aligned with ``node_views``): its instance key and the groups enclosing it."""
        return getattr(self, "_view_meta", {}).get(nid, [])

    def node_views(self, shape: Tuple[int, int], place: bool = False) -> Dict[str, List[tuple]]:
        """Memoised ``_node_views``: recomputed only when structure, body placement or template gates change."""
        if place:
            return self._node_views(shape, place)          # placing changes body positions: never cached
        key = (self._meta("structure", lambda: object()), tuple(shape), self._view_inputs())
        memo = getattr(self, "_views_memo", None)
        if memo is None or memo[0] != key:
            views = self._node_views(shape)
            self._views_memo = (key, views, self._view_meta)
        self._view_meta = self._views_memo[2]
        return self._views_memo[1]

    def _node_views(self, shape: Tuple[int, int], place: bool = False) -> Dict[str, List[tuple]]:
        """Where every node sits on the board, as ``(weight, chain, rect)`` views.

        A view reaches the node's region through ``chain``, a tuple of turned frames
        ``(slot, direction)``; each slot is in the previous frame's cells (the board first).
        ``rect`` is the node's region in the last frame. Content under a template gets one view
        per slot (weight p / n; turned when the template is dynamic) and one unturned view over
        the whole region (weight 1 - p), mirroring the replacement rule. A turn nested inside
        turned content is added to the chain, so the turns compose and their angles add.

        A body lives in its view's last frame: its position and extent are fractions of that
        frame, and its ``rect`` is its window there. With ``place=True``, each body is first
        centred in the region it receives (on its first view).
        """
        H, W = shape
        views: Dict[str, List[tuple]] = defaultdict(list)
        meta: Dict[str, List[tuple]] = defaultdict(list)
        placed = set()

        def visit(node: Node, weight: float, chain: tuple, rect: Region, ikey: tuple = (), groups: tuple = ()) -> None:
            if (weight < 1e-9 and not place) or is_empty(rect):    # placing reaches unweighted bodies too
                return
            if isinstance(node, GroupNode):                      # an instance: its own place in its frame
                if weight < 1e-9:
                    return                                       # a copy that is not there has no instance
                home = rect
                for key in (_spawned(node, ikey, node.instances) if STATIC_FIELDS else [ikey]):
                    tok = node.token(key)
                    fh, fw = _frame_shape(chain, (H, W))
                    if (place or tok.position is None) and tok.node_id not in placed:   # new instances start in place
                        r0, r1, c0, c1 = home
                        tok.position = [(r0 + r1) / (2.0 * fh), (c0 + c1) / (2.0 * fw)]
                        tok.extent = [(r1 - r0) / fh, (c1 - c0) / fw]
                        tok.anchor = (r0, c0)
                        tok.home = list(tok.position)
                        placed.add(tok.node_id)
                    rect = tok.rect(home, fh, fw) if tok.position is not None else home
                    views[node.node_id].append((weight, chain, rect))
                    meta[node.node_id].append((key, groups))
                    for child in content_children(node):
                        visit(child, weight, chain, rect, key, groups + (tok.node_id,))
                return
            if isinstance(node, BodyNode) and not isinstance(node, PlanNode):
                fh, fw = _frame_shape(chain, (H, W))
                if place and (TREE_SIZED_BODIES or node.extent is not None) and node.node_id not in placed:
                    r0, r1, c0, c1 = rect
                    node.position = [(r0 + r1) / (2.0 * fh), (c0 + c1) / (2.0 * fw)]
                    if TREE_SIZED_BODIES:              # the body fills its region
                        node.extent = [(r1 - r0) / fh, (c1 - c0) / fw]
                    placed.add(node.node_id)
                if node.position is not None:
                    rect = node.window(fh, fw)
            if weight >= 1e-9:
                views[node.node_id].append((weight, chain, rect))
                meta[node.node_id].append((ikey, groups))
            kids = content_children(node)
            if isinstance(node, BodyNode) and len(kids) > 1:
                plan(kids, weight, chain, rect, ikey, groups)
                return
            for i, child in enumerate(kids):
                visit(child, weight, chain, band_region(rect, i, len(kids), node.split_orientation)
                      if isinstance(node, SerialNode) else rect, ikey, groups)

        def plan(children: List[Node], weight: float, chain: tuple, rect: Region, ikey: tuple, groups: tuple) -> None:
            if len(children) == 1:
                visit(children[0], weight, chain, rect, ikey, groups)
                return
            template, rest = children[0], children[1:]
            visit(template, weight, chain, rect, ikey, groups)
            p = _clamp(template.gate)
            slots = [(i, s, v) for i, (s, v) in enumerate(orient_slots(template, rect)) if not is_empty(s)] \
                if ORIENTING_TEMPLATES else [(0, rect, None)]
            for i, slot, v in slots:                       # each copy is an instance: its slot extends the key
                if v is None:
                    plan(rest, weight * p / len(slots), chain, slot, ikey + (i,), groups)
                else:
                    Hc, Wc, *_ = _orientation_map(slot, v)
                    plan(rest, weight * p / len(slots), chain + ((slot, v),), (0, Hc, 0, Wc), ikey + (i,), groups)
            plan(rest, weight * (1.0 - p), chain, rect, ikey + ("w",), groups)

        for r in self.roots:
            visit(r, 1.0, (), (0, H, 0, W))
        self._view_meta = meta
        return views

    def windows(self) -> List["WindowNode"]:
        """The output windows, in program order (each is one output channel)."""
        return [n for n in self._nodes.values() if isinstance(n, WindowNode)]

    def input_windows(self) -> List["InWindowNode"]:
        """The input windows, in program order."""
        return [n for n in self._nodes.values() if isinstance(n, InWindowNode)]

    def set_window_input(self, which, field: np.ndarray) -> None:
        """Give input window ``which`` (its index in program order, or its node id) a field to show."""
        node = self._nodes[which] if isinstance(which, str) else self.input_windows()[which]
        node.field = np.clip(np.asarray(field, dtype=float), 0.0, 1.0)

    def view_masks(self, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Every node's board cells: the union of its views' cells."""
        out = {}
        for nid, vs in self.node_views(shape).items():
            out[nid] = np.logical_or.reduce([_view_mask(shape, chain, rect) for _, chain, rect in vs])
        return out

    def _render_imagery(self) -> np.ndarray:
        """The whole imagined scene on the grid: every root drawn with bodies in place (noisy-OR)."""
        return self._render_all()[1]

    def _render_all(self):
        """Each tree's own space, and all of them overlaid (for display, and perception with shared spaces).

        A tree is redrawn only when something its drawing depends on changed: any gate in the tree, its
        bodies' placement, its mass offsets, its input fields, or the semantics flags. Otherwise the last
        drawing is reused (the same inputs give the same drawing).
        """
        cache = self.__dict__.setdefault("_draw_cache", {})
        yang, yin, left, spaces = np.ones(self.grid), np.ones(self.grid), np.ones(self.grid), {}
        for r in self.roots:
            key = self._draw_key(r.node_id)
            hit = cache.get(r.node_id)
            if hit is None or hit[0] != key:
                a, b, v = self._paint_root(self.grid, r.node_id, place_bodies=True)
                hit = cache[r.node_id] = (key, a, b, v, _compose(a, _covered(a, b, v)))
            _, a, b, v, spaces[r.node_id] = hit
            yang, yin, left = yang * a, yin * b, left * v
        return spaces, _compose(yang, _covered(yang, yin, left))

    def _draw_key(self, root_id: str, substance: bool = True) -> tuple:
        """Everything a tree's drawing depends on (besides structure, which is part of the key); without the
        substance grid for a drawing of only some leaves, which never draws it."""
        def build():
            per = {}
            for r in self.roots:
                ns = list(walk(r))
                per[r.node_id] = (ns, [n for n in ns if isinstance(n, BodyNode) and not isinstance(n, PlanNode)],
                                  [n for n in ns if isinstance(n, InWindowNode)])
            return per
        nodes, bodies, inputs = self._meta("draw_nodes", build)[root_id]
        groups = tuple((t.node_id, None if t.position is None else tuple(t.position))
                       for n in nodes if isinstance(n, GroupNode) for t in n.instances.values())
        st = self._mass.get(root_id)
        return (self._meta("structure", lambda: object()), tuple(self.grid),
                tuple(n.gate for n in nodes),
                tuple((None if b.position is None else tuple(b.position), None if b.extent is None else tuple(b.extent))
                      for b in bodies),
                None if st is None else st["off"].tobytes(),
                tuple(id(n.field) for n in inputs),
                UNKNOWN_BACKGROUND, MOTION_KINDS, ORIENTING_TEMPLATES, GATE_STRUCTURES, TREE_SIZED_BODIES,
                MOTION_AWAY_FROM_FIGURE, SEPARATE_TREE_SPACES, self.dynamics, groups, GROUPING_NODES,
                PROBABILISTIC_MOTION, PURE_CLAIMS, STATIC_FIELDS, self._substance_key(root_id) if substance else None,
                tuple((k, t.tobytes()) for k, t in self._trails.items() if k[2] == root_id))

    def tree_index(self) -> Dict[str, str]:
        """Which root (tree) each node belongs to."""
        def build():
            out: Dict[str, str] = {}
            for r in self.roots:
                for n in walk(r):
                    out.setdefault(n.node_id, r.node_id)
            return out
        return self._meta("tree_index", build)

    def _paint_root(self, shape: Tuple[int, int], root_id: Optional[str], place_bodies: bool):
        """One root's paint, with its mass drawn at the displaced positions (gameboard, MOTION_KINDS)."""
        rid = root_id if root_id is not None else self.roots[-1].node_id
        st = self._mass.get(rid) if MOTION_KINDS and self.dynamics == "gameboard" else None
        if st is None or not place_bodies or tuple(shape) != tuple(self.grid) or not st["off"].any():
            return self._paint(shape, root_id, place_bodies)
        fy, fn, _ = self._paint(shape, rid, True, layer="free")
        sy, sn = _scatter(fy, st["off"]), _scatter(fn, st["off"])
        if PURE_CLAIMS:                           # one composition: nested claims keep their order across layers
            py, pn, bv = self._paint(shape, rid, True, mass_off=st["off"])
            if PROBABILISTIC_MOTION:
                vacated = ((fy < 1) | (fn < 1)) & ~((sy < 1) | (sn < 1))
                bv = np.where(vacated, 0.0, bv)
            return py, pn, bv
        by, bn, bv = self._paint(shape, rid, True, layer="bodies")
        if PROBABILISTIC_MOTION:                  # what moved away leaves known emptiness, not "no claim"
            vacated = ((fy < 1) | (fn < 1)) & ~((sy < 1) | (sn < 1))
            bv = np.where(vacated, 0.0, bv)
        return sy * by, sn * bn, bv

    def _perceive(self, views=None) -> None:
        """Each top-most observer evaluates its subtree against last tick's imagery, through its views.

        A view is a sub-field of the whole board, carried into any turned frames above the
        observer (``node_views``), so bodies inside an observing pattern carve their windows in
        absolute grid coordinates. An observer with several views (it sits in a template's
        content) takes their weighted mean. Observers with an input window keep scanning it.
        """
        shared = ReceptiveField(self._imagery if self._imagery is not None else blank_board(self.grid))
        spaces = getattr(self, "_spaces", None) or {}
        tree = self.tree_index() if SEPARATE_TREE_SPACES else {}
        views = self.node_views(self.grid) if views is None else views
        parents = self._parents()
        for node in getattr(self, "_observer_roots", []):
            if isinstance(node, ObjectNode) and self._has_input_context(node, parents):
                continue
            board = shared
            if SEPARATE_TREE_SPACES:                          # an observer sees only its own tree's space
                own = spaces.get(tree.get(node.node_id))
                board = ReceptiveField(own if own is not None else blank_board(self.grid))
            vs = views.get(node.node_id, [])
            if not vs:
                for n in walk(node):
                    n.activation = 0.0
                continue
            scores = [w * node.evaluate(_view_field(board, chain, rect)) for w, chain, rect in vs]
            node.activation = float(sum(scores) / sum(w for w, _, _ in vs))
        for node in getattr(self, "_measures", []):       # measure trees: their own figure, as drawn last tick
            own = spaces.get(self.tree_index().get(node.node_id))
            node.activation = unknown_value() if own is None else float(np.maximum(0.0, 2.0 * own - 1.0).mean())

    def _propagate_delayed(self) -> None:
        """Every dashed target takes the mean of its sources' current activations; it acts next tick."""
        sources: Dict[str, List[Node]] = defaultdict(list)
        for edge in self._dashed:
            sources[edge.target.node_id].append(edge.source)
        new = {nid: _clamp(_mean([s.activation for s in srcs])) for nid, srcs in sources.items()}
        for nid, gate in new.items():
            node = self._nodes[nid]
            node.gate = gate
            if isinstance(node, PORTALS):
                node.activation = gate

    def _moving_bodies(self, views, cells: bool = True) -> Dict[str, tuple]:
        """Each placed body's home (its first view's chain and frame size) and its board cells."""
        out = {}
        for b in self._body_nodes():
            vs = views.get(b.node_id)
            if isinstance(b, BodyNode) and not isinstance(b, PlanNode) and b.position is not None \
                    and b.extent is not None and vs:
                mask = np.logical_or.reduce([_view_mask(self.grid, chain, rect) for _, chain, rect in vs])                     if cells else None
                out[b.node_id] = (vs[0][1], _frame_shape(vs[0][1], self.grid), mask)
            elif isinstance(b, GroupNode) and vs:                # each instance moves on its own view
                for (_, chain, rect), (ikey, enclosing) in zip(vs, self.view_meta(b.node_id)):
                    tok = b.instances.get(ikey)
                    tok_groups = getattr(self, "_unit_groups", None)
                    if tok_groups is None:
                        tok_groups = self._unit_groups = {}
                    if tok is not None:
                        tok_groups[tok.node_id] = enclosing
                    if tok is not None and tok.position is not None and tok.extent is not None:
                        out[tok.node_id] = (chain, _frame_shape(chain, self.grid),
                                            _view_mask(self.grid, chain, rect) if cells else None)
        return out

    def _issue_orders(self, views=None) -> None:
        """A switched-on imagined dynamic trigram orders every body that enters its space.

        Its space is its own region on the board, through each of its views (``node_views``). A
        body is ordered when its window comes to overlap a view's cells while the trigram is
        on: at the tick it switches on, or whenever a body walks in, including one that left
        and came back. A body that stays inside gets no further orders. Only presence is
        remembered, not who was ordered. The push is the trigram's motion turned by the view's
        frames; a diagonal push is one order per axis.
        """
        parents = self._parents()
        views = self.node_views(self.grid) if views is None else views
        bodies_at = self._moving_bodies(views)
        self._units = self._tokens()
        if SEPARATE_TREE_SPACES:                              # a motion leaf moves only bodies of its own tree
            tree = self.tree_index()
            by_tree = defaultdict(dict)
            for b, v in bodies_at.items():
                by_tree[tree.get(getattr(self._units[b], "node", self._units[b]).node_id)][b] = v
        def build_templates():                   # dynamic leaves in templates give orientation, not orders
            out = set()
            for b in self._nodes.values():
                if isinstance(b, BodyNode) and not isinstance(b, GroupNode):
                    for t in content_children(b)[:-1]:
                        out.update(n.node_id for n in walk(t))
            return out
        in_templates = self._meta("in_templates", build_templates) if ORIENTING_TEMPLATES else set()
        for node in self._nodes.values():
            if not (isinstance(node, ObjectNode) and node.is_dynamic
                    and getattr(node, "role", None) == "imagined") or node.node_id in in_templates:
                continue
            a = node.gate                                 # an imagined leaf's gate (its activation equals it)
            anc = node
            while parents.get(anc.node_id):              # a gated-off composite issues no orders
                anc = parents[anc.node_id][0]
                a *= _sg(anc) if not isinstance(anc, ObjectNode) else 1.0
            self._prev_act[node.node_id] = a
            keys = [f"{node.node_id}#{k}" for k in range(len(views.get(node.node_id, [])))]
            if not a > ONSET_THRESHOLD:                   # strictly above: unknown (0.5) is not "on"
                for key in [k for k in self._inside if k.split("#")[0] == node.node_id]:
                    del self._inside[key]
                continue
            near = by_tree[tree.get(node.node_id)] if SEPARATE_TREE_SPACES else bodies_at
            acting = self._resolve_motion(node) if PROBABILISTIC_MOTION else node
            if acting is None:                            # unknown motion, still in this realization
                for key in [k for k in self._inside if k.split("#")[0] == node.node_id]:
                    del self._inside[key]
                continue
            if MOTION_KINDS:
                self._issue_kinds(acting, a, list(zip(keys, views.get(node.node_id, []))), near, parents)
                continue
            motion = BodyNode._template_motion(node)
            meta = self.view_meta(node.node_id)
            for k, (key, (_, chain, rect)) in enumerate(zip(keys, views.get(node.node_id, []))):
                space = _view_mask(self.grid, chain, rect)
                own = meta[k][1] if k < len(meta) else ()
                bodies = [self._units[b] for b, (_, _, cells) in near.items()
                          if self._pushable(b, own) and (space & cells).any()]
                was_inside = self._inside.get(key, set())
                self._inside[key] = {b.node_id for b in bodies}
                push = _view_turn(chain, motion)              # on the board
                for body in bodies:
                    if body.node_id in was_inside:        # already inside: no new order
                        continue
                    home, frame, _ = bodies_at[body.node_id]
                    for axis, m in enumerate(_view_unturn(home, push)):   # in the body's own frame
                        if abs(m) < 1e-9:
                            continue
                        cells_on_axis = frame[axis]
                        extent_cells = max(1, round(body.extent[axis] * cells_on_axis))
                        rate = a * abs(m) if TREE_SIZED_BODIES else (body.step_size or 0.0) * cells_on_axis * a * abs(m)
                        if rate > 0:
                            body.orders.append([axis, 1 if m > 0 else -1, extent_cells, rate, 0.0, node.node_id])

    # -- motion kinds: bodies and mass ------------------------------------------------

    def _mass_state(self, root_id: str) -> dict:
        """A root's mass: each board cell's drawn content is a particle with a whole-cell offset."""
        st = self._mass.get(root_id)
        if st is None:
            H, W = self.grid
            st = {"off": np.zeros((H, W, 2), dtype=int), "jit": np.zeros((H, W, 2)), "orders": []}
            self._mass[root_id] = st
        return st

    def _issue_kinds(self, node: "ObjectNode", a: float, keyed_views, bodies_at, parents) -> None:
        """One switched-on motion leaf's motion this tick, by its trigram's kind (see MOTION_KINDS)."""
        import zlib
        kind = _MOTION_KIND.get(PATTERN_TO_NAME.get(node.pattern))
        if kind is None:
            return
        root = node
        while parents.get(root.node_id):
            root = parents[root.node_id][0]
        meta = self.view_meta(node.node_id)
        units = getattr(self, "_units", None) or self._tokens()
        in_agent = lambda k: (k < len(meta) and bool(meta[k][1])               # its innermost unit is an agent
                              and getattr(getattr(units.get(meta[k][1][-1]), "node", None), "agency", False))
        keyed_views = [(key, view) for key, view in keyed_views
                       if not (SOLID_FIGURE and not _standing() and in_agent(int(key.split("#")[1])))]
        if not keyed_views:
            return                                # an agent's leaves move the agent (_drive_bodies), nothing else
        if STATIC_FIELDS and PURE_CLAIMS:
            return self._issue_substance(node, kind, a, keyed_views, bodies_at, root.node_id, meta, units)
        st = self._mass_state(root.node_id)
        H, W = self.grid
        fy, fn, _ = self._paint(self.grid, root.node_id, True, layer="free")
        paint = (fy < 1 - 1e-9) | (fn < 1 - 1e-9)                      # source cells that carry mass
        R, C = _grid_index((H, W))
        where = (R + st["off"][..., 0], C + st["off"][..., 1])          # where each source cell's mass is now
        motion = BodyNode._template_motion(node)
        axis_vec = (1.0, 0.0) if node.orientation == Orientation.VERTICAL else (0.0, 1.0)
        for key, (_, chain, rect) in keyed_views:
            space = _view_mask(self.grid, chain, rect)
            rows, cols = np.nonzero(space)
            if not rows.size:
                continue
            k = int(key.split("#")[1])
            own = meta[k][1] if k < len(meta) else ()                     # a leaf never moves its own groups
            bodies = [units[b] for b, (_, _, cells) in bodies_at.items()
                      if self._pushable(b, own) and (space & cells).any()]
            was = self._inside.get(key, set())
            self._inside[key] = {b.node_id for b in bodies}
            fresh = [b for b in bodies if b.node_id not in was]
            inside = paint & space[where]
            was_m = self._inside.get(key + "m", set())
            now_m = set(np.flatnonzero(inside).tolist())
            self._inside[key + "m"] = now_m
            fresh_m = np.zeros((H, W), dtype=bool)
            fresh_m.flat[sorted(now_m - was_m)] = True
            if kind in ("bulk", "gradient"):
                norm = float(np.hypot(*motion))
                if norm < 1e-12:
                    continue
                d = _view_turn(chain, (motion[0] / norm, motion[1] / norm))
            else:
                d = _view_turn(chain, axis_vec)
            rng = np.random.default_rng(zlib.crc32(f"{node.node_id}|{key}|{self._tick}".encode()))
            for axis, comp in enumerate(d):
                if abs(comp) < 1e-9:
                    continue
                lo, hi = (rows.min(), rows.max()) if axis == 0 else (cols.min(), cols.max())
                span, centre = int(hi - lo + 1), (lo + hi + 1) / 2.0
                coord = where[axis]
                sign = 1 if comp > 0 else -1
                rate = a * abs(comp)
                if kind == "random":
                    st["jit"][..., axis] += np.where(inside, rng.uniform(-1, 1, (H, W)) * rate, 0.0)
                    for b in bodies:
                        self._push_body(b, bodies_at, axis, float(rng.uniform(-1, 1)) * rate, jitter=True)
                    continue
                if kind == "bulk":
                    if fresh_m.any():
                        st["orders"].append([fresh_m, axis, sign, span, rate, 0.0, 0.0])
                    for b in fresh:
                        self._push_body(b, bodies_at, axis, sign * rate, node_id=node.node_id)
                elif kind == "gradient":                                # layers from the leading edge
                    levels = sorted(set(coord[fresh_m].tolist()), reverse=sign > 0)
                    for k, v in enumerate(levels):
                        st["orders"].append([fresh_m & (coord == v), axis, sign, span, rate, 0.0, k / a])
                    lead = lambda b: -sign * float(np.nonzero(bodies_at[b.node_id][2])[axis].mean())
                    for k, b in enumerate(sorted(fresh, key=lead)):
                        self._push_body(b, bodies_at, axis, sign * rate, node_id=node.node_id, delay=k / a)
                else:                                                   # out / in: the two halves
                    for half in (-1, 1):
                        part = fresh_m & (np.sign(coord + 0.5 - centre) == half)
                        go = half if kind == "out" else -half
                        if part.any():
                            st["orders"].append([part, axis, go, max(1, round(span / 4)), rate, 0.0, 0.0])
                    for b in fresh:
                        side = np.sign(float(np.nonzero(bodies_at[b.node_id][2])[axis].mean()) + 0.5 - centre)
                        if side:
                            go = side if kind == "out" else -side
                            self._push_body(b, bodies_at, axis, go * rate, node_id=node.node_id, half=True)

    def _own_figure(self, root_id: str, node) -> np.ndarray:
        """STANDING_MOTION: where a body's own content is figure."""
        v, c = self._paint_only_values(root_id, self._own_content(node))
        return (c > 0) & (_claimed_value(v, c) > 0.5)

    def _own_content(self, node) -> frozenset:
        """STANDING_MOTION: a body's own content: its nodes and its fields', not its nested bodies'."""
        def build():
            out = {}
            for g in self._body_nodes():
                ids, stack = set(), [g]
                while stack:
                    n = stack.pop()
                    ids.add(n.node_id)
                    stack.extend(c for c in getattr(n, "children", ())
                                 if not (isinstance(c, GroupNode) and not isinstance(c, FieldNode)))
                out[g.node_id] = frozenset(ids)
            return out
        return self._meta("own_content", build)[node.node_id]

    def _issue_substance(self, node, kind: str, a: float, keyed_views, bodies_at, root_id: str, meta, units) -> None:
        """STATIC_FIELDS: ``_issue_kinds`` on the substance grid. The substance in a view's space that this leaf has
        not ordered yet (its marks, which follow the content and are dropped when it leaves the space) gets orders;
        a field's cells are unmarked at every reset, so motion keeps carrying fresh copies of their substance away.
        "In" moves each row (or column) only as far as the centre, so the halves pile up there."""
        S = self._substance(root_id)
        H, W = self.grid
        R, C = _grid_index((H, W))
        none = np.zeros((H, W), dtype=bool)
        src = S.get("src", {})
        loose = (S["s"] > 0) & (S["v"] > 0.5)
        motion = BodyNode._template_motion(node)
        axis_vec = (1.0, 0.0) if node.orientation == Orientation.VERTICAL else (0.0, 1.0)
        for key, (_, chain, rect) in keyed_views:
            space = _view_mask(self.grid, chain, rect)
            rows, cols = np.nonzero(space)
            if not rows.size:
                continue
            k = int(key.split("#")[1])
            own = meta[k][1] if k < len(meta) else ()
            mine = getattr(getattr(units.get(own[-1]), "node", None), "node_id", None) if own else None
            stuff = loose | src.get(None, none) | src.get(mine, none)    # a carried field: only its own motion
            n_orders = len(S["orders"])
            if _standing():                      # everything whose substance is here, without an order from it
                bodies = [units[b] for b, (_, _, cells) in sorted(bodies_at.items())
                          if not isinstance(getattr(units[b], "node", None), FieldNode) and (space & cells).any()
                          and (self._own_figure(root_id, getattr(units[b], "node", units[b])) & cells & space).any()]
                self._inside[key] = {b.node_id for b in bodies}
                fresh = [b for b in bodies if not any(o[5] == node.node_id for o in b.orders)]
                under_way = np.logical_or.reduce([o[0] for o in S["orders"] if o[7:] == [key]] + [none])
                fresh_m = stuff & space & ~under_way
            else:
                bodies = [units[b] for b, (_, _, cells) in bodies_at.items()
                          if self._pushable(b, own) and (space & cells).any()
                          and not isinstance(getattr(units[b], "node", None), FieldNode)]       # fields do not move
                was = self._inside.get(key, set())
                self._inside[key] = {b.node_id for b in bodies}
                fresh = [b for b in bodies if b.node_id not in was]
                marks = S["marks"].get(key, np.zeros((H, W), dtype=bool)) & space
                fresh_m = stuff & space & ~marks
                S["marks"][key] = marks | fresh_m    # emitted copies carry the mark; the reset unmarks field cells
            if kind in ("bulk", "gradient"):
                norm = float(np.hypot(*motion))
                if norm < 1e-12:
                    continue
                d = _view_turn(chain, (motion[0] / norm, motion[1] / norm))
            else:
                d = _view_turn(chain, axis_vec)
            for axis, comp in enumerate(d):
                if abs(comp) < 1e-9 or kind == "random":
                    continue
                lo, hi = (rows.min(), rows.max()) if axis == 0 else (cols.min(), cols.max())
                span, centre = int(hi - lo + 1), (lo + hi + 1) / 2.0
                coord = (R, C)[axis]
                sign = 1 if comp > 0 else -1
                rate = a * abs(comp)
                if kind == "bulk":
                    if fresh_m.any():
                        S["orders"].append([fresh_m.copy(), axis, sign, span, rate, 0.0, 0.0])
                    for b in fresh:
                        self._push_body(b, bodies_at, axis, sign * rate, node_id=node.node_id)
                elif kind == "gradient":                                # layers from the leading edge
                    levels = sorted(set(coord[fresh_m].tolist()), reverse=sign > 0)
                    for j, v in enumerate(levels):
                        S["orders"].append([fresh_m & (coord == v), axis, sign, span, rate, 0.0, j / a])
                    lead = lambda b: -sign * float(np.nonzero(bodies_at[b.node_id][2])[axis].mean())
                    for j, b in enumerate(sorted(fresh, key=lead)):
                        self._push_body(b, bodies_at, axis, sign * rate, node_id=node.node_id, delay=j / a)
                else:                                                   # out / in: the two halves
                    for half in (-1, 1):
                        part = fresh_m & (np.sign(coord + 0.5 - centre) == half)
                        go = half if kind == "out" else -half
                        if not part.any():
                            continue
                        if kind == "out":
                            S["orders"].append([part, axis, go, max(1, round(span / 4)), rate, 0.0, 0.0])
                            continue
                        for v in sorted(set(coord[part].tolist())):      # in: as far as the centre, no further
                            dist = int(round(abs(v + 0.5 - centre) - 0.5))
                            if dist > 0:
                                S["orders"].append([part & (coord == v), axis, go, dist, rate, 0.0, 0.0])
                    for b in fresh:
                        side = np.sign(float(np.nonzero(bodies_at[b.node_id][2])[axis].mean()) + 0.5 - centre)
                        if side:
                            go = side if kind == "out" else -side
                            self._push_body(b, bodies_at, axis, go * rate, node_id=node.node_id, half=True)
            if _standing():
                for o in S["orders"][n_orders:]:
                    o.append(key)

    def _push_body(self, body, bodies_at, axis: int, signed_rate: float, node_id: str = "",
                   delay: float = 0.0, half: bool = False, jitter: bool = False) -> None:
        """Order (or jitter) a body along a board axis, expressed in the body's own frame."""
        home, frame, _ = bodies_at[body.node_id]
        vec = [0.0, 0.0]
        vec[axis] = signed_rate
        for ax, m in enumerate(_view_unturn(home, tuple(vec))):
            if abs(m) < 1e-9:
                continue
            if jitter:
                body.jitter[ax] += m
                continue
            cells = max(1, round(body.extent[ax] * frame[ax]))
            body.orders.append([ax, 1 if m > 0 else -1, max(1, round(cells / 2)) if half else cells,
                                abs(m), 0.0, node_id, delay])

    def _move_mass(self) -> None:
        """Mass orders (and random pushes) move particles at most one cell per tick; mass stays on the board."""
        H, W = self.grid
        R, C = _grid_index((H, W))
        for st in self._mass.values():
            keep = []
            for o in st["orders"]:                     # [particles, axis, direction, cells left, rate, credit, delay]
                if o[6] >= 1 - 1e-9:
                    o[6] -= 1
                    keep.append(o)
                    continue
                o[5] += o[4]
                k = min(int(o[5]), 1, o[3])
                o[5] -= k
                o[3] -= k
                if k:
                    st["off"][o[0], o[1]] += o[2] * k
                if o[3] > 0:
                    keep.append(o)
            st["orders"] = keep
            step = np.trunc(st["jit"]).clip(-1, 1).astype(int)
            st["off"] += step
            st["jit"] -= step
            st["off"][..., 0] = np.clip(R + st["off"][..., 0], 0, H - 1) - R
            st["off"][..., 1] = np.clip(C + st["off"][..., 1], 0, W - 1) - C

    def _substance_key(self, root_id: str):
        S = getattr(self, "_sub", {}).get(root_id)
        return None if S is None else (S["v"].tobytes(), S["s"].tobytes(), S["moved"].tobytes())

    def motion_state(self) -> tuple:
        """Everything about mass that can change a later frame (for periodicity checks)."""
        out = tuple((rid, st["off"].tobytes(), np.round(st["jit"], 6).tobytes(),
                     tuple((o[0].tobytes(), o[1], o[2], o[3], round(o[4], 6), round(o[5], 6), round(o[6], 6))
                           for o in st["orders"]))
                    for rid, st in sorted(self._mass.items()))
        jit = tuple(tuple(round(v, 6) for v in getattr(n, "jitter", ())) for n in self._body_nodes())
        random_now = any(n.is_dynamic and PATTERN_TO_NAME.get(n.pattern) == "universal" for n in self.leaf_nodes())
        groups = tuple((t.node_id, None if t.position is None else tuple(round(v, 9) for v in t.position),
                        tuple(tuple(o) for o in t.orders), tuple(round(v, 6) for v in t.jitter + t.drive))
                       for t in self._group_tokens())
        subs = tuple((rid, self._substance_key(rid), tuple(sorted((k, m.tobytes()) for k, m in S["marks"].items())),
                      tuple((o[0].tobytes(), o[1], o[2], o[3], round(o[4], 6), round(o[5], 6), round(o[6], 6))
                            for o in S["orders"]))
                     for rid, S in sorted(getattr(self, "_sub", {}).items()))
        return (out, jit, (self._tick if random_now else 0)) + ((groups,) if groups else ()) + ((subs,) if subs else ())

    def _move_bodies_gameboard(self, views=None) -> None:
        """Each order advances at most one cell per tick once its credit reaches a whole cell; moves add up.

        A body moves on its own frame's cells (the board, or the turned frame it lives in).
        """
        homes = self._moving_bodies(self.node_views(self.grid) if views is None else views, cells=False)
        self._solid_cache, self._moves = {}, Counter()     # solidity maps, valid until something else moves
        for body in self._tokens().values():
            if body.position is None:
                continue
            if STATIC_FIELDS and PURE_CLAIMS and isinstance(getattr(body, "node", None), FieldNode):
                continue                              # a field never moves itself (a body around it can)
            H, W = homes[body.node_id][1] if body.node_id in homes else self.grid
            net = [0, 0]
            for order in body.orders:
                if len(order) > 6 and order[6] >= 1 - 1e-9:     # queued: waits its turn
                    order[6] -= 1
                    continue
                order[4] += order[3]
                k = min(int(order[4]), 1, order[2])
                order[4] -= k
                order[2] -= k
                net[order[0]] += order[1] * k
            body.orders = [o for o in body.orders if o[2] > 0]
            for ax in (0, 1):                                    # random pushes (universal), whole cells
                step = max(-1, min(1, math.trunc(body.jitter[ax])))
                net[ax] += step
                body.jitter[ax] -= step
            drive = getattr(body, "drive", None)
            for ax in ((0, 1) if drive is not None else ()):     # agency: a body's own leaves, whole cells
                step = max(-1, min(1, math.trunc(drive[ax])))
                net[ax] += step
                drive[ax] -= step
            if SOLID_FIGURE and any(net) and (H, W) == tuple(self.grid):
                net = self._chain_move(body, net) if _standing() else self._solid_step(body, net)
            moved = [self._clamped_shift(body, axis, net[axis], n) for axis, n in ((0, H), (1, W))]
            if any(moved):
                self._moves[getattr(body, "node", body).node_id] += 1
            for axis, n in ((0, H), (1, W)):
                size = max(1, round(body.extent[axis] * n))
                if TREE_SIZED_BODIES:                    # move the window's start cell; keep it on the board
                    start = round(body.position[axis] * n - size / 2) + net[axis]
                    start = min(max(start, 0), n - size)
                    body.position[axis] = (start + size / 2) / n
                    continue
                cell = round(body.position[axis] * n) + net[axis]
                cell = min(max(cell, size // 2), n - size + size // 2)
                body.position[axis] = cell / n
            body.velocity = [net[0] / H, net[1] / W]
            if SOLID_FIGURE and SOFT_EDGES and any(moved) and (H, W) == tuple(self.grid) and not _standing():
                self._push_contacts(body, moved)

    def _solid_step(self, body, net: list) -> list:
        """SOLID_FIGURE: the move ``net`` (cells), or the part of it along one axis, that keeps the body's figure out
        of cells the rest of its tree makes figure (above SOLID_THRESHOLD); no move if neither is free."""
        node = body.node if isinstance(body, GroupToken) else body
        moves = getattr(self, "_moves", Counter())
        stamp = sum(moves.values()) - moves[node.node_id]          # moves by anything but this body's instances
        cache = self.__dict__.setdefault("_solid_cache", {})
        hit = cache.get(node.node_id)
        if hit is not None and hit[0] == stamp:    # instances of one body pass through each other: shared maps
            others, own = hit[1]
        else:
            others, own = self._solid_maps(node)
            cache[node.node_id] = (stamp, (others, own))
        self._solid_others = others              # reused by _push_contacts for this move
        H, W = self.grid
        r0, r1, c0, c1 = body.window(H, W)
        mine = np.zeros((H, W), dtype=bool)
        if _soft_body(node):                                   # an agent moves by its core: wherever it leans figure
            q0, q1, p0, p1 = unit_core((r0, r1, c0, c1))
            mine[q0:q1, p0:p1] = own[q0:q1, p0:p1] > 0.5
        else:
            mine[r0:r1, c0:c1] = own[r0:r1, c0:c1] > SOLID_THRESHOLD
        wall = others > SOLID_THRESHOLD
        for step in ((net[0], net[1]), (net[0], 0), (0, net[1])):
            if step == (0, 0):
                continue
            dr, dc = (self._clamped_shift(body, axis, d, n) for axis, d, n in ((0, step[0], H), (1, step[1], W)))
            if (dr, dc) != (0, 0) and not (np.roll(mine, (dr, dc), axis=(0, 1)) & wall).any():
                return [dr, dc]
        return [0, 0]

    def _push_contacts(self, body, d: list) -> None:
        """SOLID_FIGURE with SOFT_EDGES: what a moved body's edge now overlaps is pushed by its move ``d``.

        A rigid unit (a top-level field or body, carrying everything nested in it) moves whole, as far as the board
        lets it: a unit that fills its frame cannot move, so it is a fixed ground. Free mass (substance in no unit)
        is a pile: each particle under the edge moves on its own, so a push breaks it up.
        """
        H, W = self.grid
        node = body.node if isinstance(body, GroupToken) else body
        if not getattr(node, "agency", False):
            return                                # only an agent's edge pushes
        r0, r1, c0, c1 = body.window(H, W)
        ring = np.zeros((H, W), dtype=bool)
        ring[r0:r1, c0:c1] = True
        q0, q1, p0, p1 = unit_core((r0, r1, c0, c1))
        ring[q0:q1, p0:p1] = False
        others = getattr(self, "_solid_others", None)
        if not ring.any() or (others is not None and not (ring & (others > 0.5)).any()):
            return                                # the edge touches nothing: no contact to resolve
        root = self.tree_index()[node.node_id]
        mine = {n.node_id for n in walk(node)}
        parents = self._parents()
        for top in [g for g in self._nodes.values() if isinstance(g, GroupNode) and g.node_id not in mine
                    and self.tree_index().get(g.node_id) == root
                    and not (STATIC_FIELDS and PURE_CLAIMS and isinstance(g, FieldNode))      # fields hold
                    and not any(isinstance(a, GroupNode) for a in self._ancestors(g, parents))]:
            for tok in list(top.instances.values()):
                if tok.position is None or tok.extent is None:
                    continue
                t0, t1, u0, u1 = tok.window(H, W)
                stuff = self._solid_maps(top, only=({n.node_id for n in walk(top)} - mine))[1] > SOLID_THRESHOLD
                inside = np.zeros((H, W), dtype=bool)
                inside[t0:t1, u0:u1] = True
                if (ring & stuff & inside).any():
                    carried = top in self._ancestors(node, parents)          # pushing the unit it is inside
                    for axis, n in ((0, H), (1, W)):
                        size = max(1, round(tok.extent[axis] * n))
                        start = round(tok.position[axis] * n - size / 2)
                        shift = self._clamped_shift(tok, axis, d[axis], n)
                        tok.position[axis] = (start + shift + size / 2) / n
                        if shift and hasattr(self, "_moves"):
                            self._moves[top.node_id] += 1
                        if carried:                   # both move together: the body keeps its place inside
                            body.position[axis] -= shift / n
        if not MOTION_KINDS:
            return
        if STATIC_FIELDS and PURE_CLAIMS:            # contact moves substance outside fields
            S = self._substance(root)
            cells = ring & (S["s"] > 0) & (S["v"] > 0.5) & ~S["F"]
            if cells.any():
                self._shift_substance(S, cells, d[0], d[1])
                if hasattr(self, "_moves"):
                    self._moves["substance"] += 1
            return
        st = self._mass_state(root)
        fy, fn, _ = self._paint(self.grid, root, True, layer="free")
        R, C = _grid_index((H, W))
        where = (R + st["off"][..., 0], C + st["off"][..., 1])
        hit = (fy < 0.5) & ring[where]                                   # substance now under the edge
        st["off"][..., 0] = np.where(hit, np.clip(where[0] + d[0], 0, H - 1) - R, st["off"][..., 0])
        st["off"][..., 1] = np.where(hit, np.clip(where[1] + d[1], 0, W - 1) - C, st["off"][..., 1])

    def _chain_move(self, body, net: list) -> list:
        """STANDING_MOTION with SOLID_FIGURE: the move ``net`` (or its part along one axis) if everything it pushes can
        move with it, else none. A body pushes what its figure would move into: a unit, with everything inside it (a
        body pushing the unit it is inside is carried along), and through it whatever that unit would move into; stuff
        as a run (``_stuff_run``). Nothing may leave the board or enter a field's figure; instances of one body pass
        through each other. A unit's figure here is where its own content is drawn above SOLID_THRESHOLD; an agent of
        uncertain extent's is its core where it leans figure (as in ``_solid_step``: an uncertain edge may overlap, so
        a body has room in its slot), or all of it where it leans figure if it is too small to have a core."""
        H, W = self.grid
        units, groups = self._tokens(), getattr(self, "_unit_groups", {})
        node_of = lambda b: getattr(units[b], "node", units[b])
        root = self.tree_index()[node_of(body.node_id).node_id]
        def figure(u, cells):                    # what of a unit's own content is solid (as in _solid_step)
            v = self._paint_only_values(root, self._own_content(u))[0]
            if not _soft_body(u):
                return (v > SOLID_THRESHOLD) & cells
            rows, cols = np.nonzero(cells)            # an agent of uncertain extent: its core, where it leans figure;
            q0, q1, p0, p1 = unit_core((rows.min(), rows.max() + 1, cols.min(), cols.max() + 1))
            core = np.zeros_like(cells)
            core[q0:q1, p0:p1] = True
            return (v > 0.5) & (core if (core & cells).any() else cells)       # too small for a core: all of it
        own = {b: figure(node_of(b), cells) for b, (_, _, cells) in self._moving_bodies(self.node_views(self.grid)).items()
               if cells.any() and self.tree_index().get(node_of(b).node_id) == root and not isinstance(node_of(b), FieldNode)}
        inside = lambda b, a: a in groups.get(b, ())                          # unit b is inside unit a
        none = np.zeros((H, W), dtype=bool)
        group = lambda a: np.logical_or.reduce([own[b] for b in own if b == a or inside(b, a)] + [none])
        S = self._substance(root)
        stuff = (S["s"] > 0) & (S["v"] > 0.5) & ~S["F"]
        fixed = S["F"] & (0.5 + (S["fe"] - 0.5) * S["fc"] > SOLID_THRESHOLD)             & ~np.logical_or.reduce(list(own.values()) + [none])                  # carried fields move with their unit

        def plan(d):
            moving, push, queue = [], none.copy(), [body.node_id]
            while queue:
                x = queue.pop()
                if x in moving or any(inside(x, m) for m in moving):
                    continue                                                  # carried by a moving unit
                g = group(x)
                foot = _moved_mask(g, *d)
                if foot.sum() < g.sum() or (foot & fixed).any():
                    return None                                               # off the board, or into a field
                if not groups.get(x) and [self._clamped_shift(units[x], a, d[a], n) for a, n in ((0, H), (1, W))] != list(d):
                    return None
                moving.append(x)
                push |= foot & stuff & ~g
                queue += [y for y in own if y not in moving and not inside(y, x) and node_of(y) is not node_of(x)
                          and (foot & own[y] & ~g).any()]
            if push.any() and _stuff_run(S, push, *d)[1].any():
                return None                                                   # stuff in the way cannot move
            return moving, push
        for d in ((net[0], net[1]), (net[0], 0), (0, net[1])):
            found = plan(d) if d != (0, 0) else None
            if found is None:
                continue
            moving, push = found
            for x in moving:
                if x == body.node_id or any(inside(x, m) for m in moving if m != x):
                    continue
                tok = units[x]
                for axis, n in ((0, H), (1, W)):
                    size = max(1, round(tok.extent[axis] * n))
                    tok.position[axis] = (round(tok.position[axis] * n - size / 2) + d[axis] + size / 2) / n
                if hasattr(self, "_moves"):
                    self._moves[node_of(x).node_id] += 1
            if push.any():
                self._shift_substance(S, push, *d)
                if hasattr(self, "_moves"):
                    self._moves["substance"] += 1
            return [0, 0] if any(inside(body.node_id, m) for m in moving) else list(d)   # carried: no move of its own
        return [0, 0]

    @staticmethod
    def _ancestors(node, parents) -> list:
        out = []
        while parents.get(node.node_id):
            node = parents[node.node_id][0]
            out.append(node)
        return out

    @staticmethod
    def _clamped_shift(body, axis: int, d: int, n: int) -> int:
        """How many cells a move of ``d`` along ``axis`` actually shifts the body (it stays on the board)."""
        size = max(1, round(body.extent[axis] * n))
        if TREE_SIZED_BODIES:
            start = round(body.position[axis] * n - size / 2)
            return min(max(start + d, 0), n - size) - start
        cell = round(body.position[axis] * n)
        return min(max(cell + d, size // 2), n - size + size // 2) - cell

    def _solid_maps(self, node, only=None) -> Tuple[np.ndarray, np.ndarray]:
        """The body's tree drawn without the body's subtree, and the body's subtree (or the leaves ``only``) drawn
        alone (where it is). Memoised by what the tree's drawing depends on: a blocked body probes the same state."""
        root = self.tree_index()[node.node_id]
        ids = frozenset(n.node_id for n in walk(node))
        key = (self._draw_key(root), ids, None if only is None else frozenset(only))
        cache = self.__dict__.setdefault("_probes", {})
        if key in cache:
            return cache[key]
        trails = dict(self._trails)               # probes must not extend any trail
        out = []
        try:
            for skip, keep in ((ids, None), (None, ids if only is None else only)):
                self._paint_skip, self._paint_only = skip, keep
                a, b, v = self._paint_root(self.grid, root, True)
                out.append(_compose(a, _covered(a, b, v)))
        finally:
            self._paint_skip = self._paint_only = None
            self._trails = trails
        if len(cache) > 256:
            cache.clear()
        cache[key] = out[0], out[1]
        return cache[key]

    def _pushable(self, unit_id: str, own: tuple) -> bool:
        """Can a leaf inside the groups ``own`` push this unit? Not one of its own groups, and not a nested
        group: a nested group is rigidly part of its parent and moves only with it (or by its own agency)."""
        if unit_id in own:
            return False
        return not getattr(self, "_unit_groups", {}).get(unit_id, ())

    def _drive_bodies(self, views) -> None:
        """Agency: each body instance is pushed by its own switched-on dynamic leaves, every tick.

        A leaf belongs to the innermost group enclosing it; only a body (not a field) is driven. The push is
        the leaf's motion turned into the board by the leaf's frames, then into the body's own frame, at the
        leaf's gate (as orders are). Universal pushes at random; centre and periphery have no net direction.
        """
        import zlib
        if _standing():
            return                                # self-motion is the standing rule acting on the body's own content
        parents = self._parents()
        homes = self._moving_bodies(views, cells=False)
        units = self._tokens()
        for node in self._nodes.values():
            if not (isinstance(node, ObjectNode) and node.is_dynamic and getattr(node, "role", None) == "imagined"):
                continue
            a, anc = node.gate, node
            while parents.get(anc.node_id):
                anc = parents[anc.node_id][0]
                a *= _sg(anc) if not isinstance(anc, ObjectNode) else 1.0
            if not a > ONSET_THRESHOLD:
                continue
            acting = self._resolve_motion(node) if PROBABILISTIC_MOTION else node
            if acting is None:
                continue
            kind = _MOTION_KIND.get(PATTERN_TO_NAME.get(acting.pattern))
            for k, (_, chain, _) in enumerate(views.get(node.node_id, [])):
                groups = self.view_meta(node.node_id)[k][1]
                tok = units.get(groups[-1]) if groups else None
                if tok is None or not getattr(tok.node, "agency", False) or tok.node_id not in homes:
                    continue
                if kind == "random":
                    rng = np.random.default_rng(zlib.crc32(f"{node.node_id}|{tok.node_id}|{self._tick}".encode()))
                    push = (float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)))
                    push = (push[0], 0.0) if node.orientation == Orientation.VERTICAL else (0.0, push[1])
                elif kind in ("bulk", "gradient"):
                    push = _view_turn(chain, BodyNode._template_motion(acting))
                else:
                    continue
                for ax, m in enumerate(_view_unturn(homes[tok.node_id][0], push)):
                    tok.drive[ax] += a * m

    # -- unknown motion (PROBABILISTIC_MOTION) ---------------------------------------------------------

    def unknown_motion_units(self) -> Dict[str, Tuple[str, List[str], "Orientation"]]:
        """Unit key -> (kind, leaf ids, orientation) for every imagined universal* / neutral* leaf.

        A unit is the leaf's innermost field or body node (its template copies share it), else the leaf;
        universal and neutral leaves of one unit are separate units. Keys look like ``"n4:universal"``.
        """
        views = self.node_views(self.grid)
        out: Dict[str, Tuple[str, List[str], Orientation]] = {}
        for node in self._nodes.values():
            kind = _UNKNOWN_MOTION.get(node.pattern) if isinstance(node, ObjectNode) else None
            if kind is None or not node.is_dynamic or getattr(node, "role", None) != "imagined":
                continue
            meta = self.view_meta(node.node_id) if views.get(node.node_id) else []
            groups = meta[0][1] if meta else ()
            unit = groups[-1].split("@")[0] if groups else node.node_id
            key = f"{unit}:{kind}"
            out.setdefault(key, (kind, [], node.orientation))[1].append(node.node_id)
        return out

    def _resolve_motion(self, node: "ObjectNode"):
        """The leaf as it acts in this realization: itself, a directed stand-in, or None (still)."""
        kind = _UNKNOWN_MOTION.get(node.pattern)
        if kind is None:
            return node
        unit = self._motion_unit(node.node_id)
        state = getattr(self, "motion_states", {}).get(f"{unit}:{kind}", "a" if kind == "universal" else "still")
        if state == "still":
            return None
        return _AsMotion(node, TRIGRAM_PATTERNS["top" if state == "a" else "bottom"])

    def _motion_unit(self, nid: str) -> str:
        meta = self.view_meta(nid)
        groups = meta[0][1] if meta else ()
        return groups[-1].split("@")[0] if groups else nid

    def _step_gameboard(self) -> float:
        """One tick: scan, perceive last tick's imagery, issue orders, propagate (for next tick), move, draw.

        Orders are issued before propagation, so they see the gates set last tick, both on motion
        leaves (through this tick's scan) and on their ancestors.
        """
        self.assign_roles()
        result = self._scan()
        views = self.node_views(self.grid)      # unchanged until the bodies move
        self._perceive(views)
        self._issue_orders(views)
        if GROUPING_NODES:
            self._drive_bodies(views)
        self._propagate_delayed()
        self._move_bodies_gameboard(views)
        if STATIC_FIELDS and PURE_CLAIMS:
            for rid in list(getattr(self, "_sub", {})):
                self._move_substance(rid)
                self._reset_fields(rid)
            self._spawn_bodies()
        elif MOTION_KINDS:
            self._move_mass()
        self._spaces, self._imagery = self._render_all()
        self._tick += 1
        return result

    def evaluate_against(self, field: ReceptiveField, at_node_id: Optional[str] = None,
                         modality: str = "retina", detach_after: bool = True) -> float:
        """Show ``field`` on the matching InputNode, ``evaluate()``, then restore the old field."""
        node = self._find_input_node(modality, at_node_id=at_node_id)
        old = node.field
        node.field = field
        result = self.evaluate()
        if detach_after:
            node.field = old
        return result

    # -- input windows -------------------------------------------------

    def _find_input_node(self, node_id_or_modality: str = "retina",
                         at_node_id: Optional[str] = None) -> InputNode:
        """Find an InputNode by id or by modality, optionally only among one ParallelNode's children."""
        if at_node_id is None:
            candidates = [n for n in self._nodes.values() if isinstance(n, InputNode)]
        else:
            parent = self._nodes.get(at_node_id)
            if parent is None:
                raise KeyError(f"No node found with id {at_node_id!r}.")
            if not isinstance(parent, ParallelNode):
                raise TypeError(f"Input lookup target must be a ParallelNode; "
                                f"{at_node_id!r} is a {type(parent).__name__}.")
            candidates = [n for n in parent.children if isinstance(n, InputNode)]

        for node in candidates:
            if node_id_or_modality in (node.node_id, node.modality):
                return node
        where = f" under {at_node_id!r}" if at_node_id is not None else ""
        raise ValueError(f"No explicit InputNode matching id or modality "
                         f"{node_id_or_modality!r}{where}. Add InputNode('I', modality=...) "
                         f"to the graph before evaluation.")

    def get_field_node(self, node_id_or_modality: str) -> InputNode:
        return self._find_input_node(node_id_or_modality)

    def set_field(self, node_id_or_modality: str, field: Optional[ReceptiveField]) -> None:
        """Change what an input window sees. ``None`` silences the window."""
        self._find_input_node(node_id_or_modality).field = field

    set_input = set_field

    def add_field_slot(self, node_id: str, modality: str = "retina",
                       at_node_id: Optional[str] = None, initial_weight: float = 1.0,
                       initial_field: Optional[ReceptiveField] = None) -> InputNode:
        """Add a permanent InputNode under a ParallelNode (default: ``self.root``).

        Example: an initially silent proprio channel that a desire node can gate on::

            g.add_field_slot("proprio", modality="proprio", initial_weight=0.0)
            g.add_dashed_edge(desire, g.get_field_node("proprio"))
        """
        known = {"retina": RetinaNode, "tactile": TactileNode, "proprio": ProprioNode}
        cls = known.get(modality, InputNode)
        node = cls(node_id, field=initial_field,
                   modality=None if modality in known else modality,
                   initial_weight=initial_weight)

        target_id = at_node_id or (self.root.node_id if self.root else None)
        if target_id is None or target_id not in self._nodes:
            raise ValueError(f"No target ParallelNode found (at_node_id={at_node_id!r}).")
        target = self._nodes[target_id]
        if not isinstance(target, ParallelNode):
            raise TypeError(f"Field slots can only be attached to ParallelNodes; "
                            f"'{target_id}' is a {type(target).__name__}.")
        self.add_node(node)
        self.add_solid_edge(target, node)
        return node

    def detach_field_nodes(self) -> None:
        """Remove every InputNode from the graph."""
        for node in [n for n in self._nodes.values() if isinstance(n, InputNode)]:
            self.remove_node(node)

    # -- spatial read-outs ---------------------------------------------

    def render_output(self, shape: Tuple[int, int] = (9, 9),
                      output_root_id: Optional[str] = None,
                      place_bodies: bool = False) -> ReceptiveField:
        """Draw the imagined (output) subgraph as a field.

        This follows the replacement rule spatially: a template's yang bands
        receive the content with weight p, and the whole region receives it
        with weight 1 - p. An ObjectNode paints its bands at ``gate * weight``;
        overlapping paint combines by noisy-OR. How yin bands and undrawn cells
        read is set by ``UNKNOWN_BACKGROUND``.
        The default root is the last root (the imagined one by convention).

        With ``place_bodies=True``, a moving BodyNode is drawn in its own
        window (``BodyNode.window``), which is the region ``evaluate()`` carves
        for it, instead of its parent's whole region. This makes body motion
        visible in the output. The default keeps the original behaviour.
        """
        a, b, v = self._paint_root(shape, output_root_id, place_bodies)
        return ReceptiveField(_compose(a, _covered(a, b, v)))

    def _paint(self, shape: Tuple[int, int], output_root_id: Optional[str],
               place_bodies: bool, layer: str = "all",
               mass_off: Optional[np.ndarray] = None, values: bool = False):
        """The noisy-OR complements ``prod(1 - c)`` of one root's yang and yin paint, and of the emptiness its
        moved bodies left behind (applied by ``_covered`` only where nothing else claims a cell).

        ``layer`` "free" keeps only paint drawn outside bodies (mass), "bodies" only paint inside them.
        """
        H, W = shape
        roots = self.roots
        out_root = self._nodes[output_root_id] if output_root_id is not None else roots[-1]
        # cv: [yang paint (and, in the earlier renderer, yin paint), yin paint, place bodies, frame size].
        # An oriented slot swaps in a canvas for the content's upright frame (render_oriented).
        cv = [np.ones((H, W)), np.ones((H, W)), place_bodies, (H, W), False,   # ..., frame size, inside a body,
              np.ones((H, W)),                                                # emptiness left behind,
              [] if PURE_CLAIMS else None, (0, 0)]       # claims (PURE_CLAIMS), offset from a moved body's home
        skip, only = getattr(self, "_paint_skip", None), getattr(self, "_paint_only", None)   # solidity probes
        frames = [0]                             # turned frames entered (mass moves on the board only)

        def at_home(cells):
            """Cells moved back to where the enclosing body started."""
            dr, dc = cv[7]
            if (dr, dc) == (0, 0):
                return cells
            rs, cs = np.nonzero(cells)
            keep = (rs + dr >= 0) & (rs + dr < cells.shape[0]) & (cs + dc >= 0) & (cs + dc < cells.shape[1])
            out = np.zeros_like(cells)
            out[rs[keep] + dr, cs[keep] + dc] = True
            return out

        def claim(mask, value, w, leaf, home=None, moved=False) -> None:
            """Record a PURE_CLAIMS claim; its home cells (``home``, default its cells) moved back to where the
            enclosing body started. Free mass (``mass_off``, outside bodies, on the board) is drawn where its
            particles are now; a claim whose particles moved counts as moved."""
            home = at_home(mask if home is None else home)
            moved = moved or cv[7] != (0, 0)
            if mass_off is not None and not cv[4] and frames[0] == 0:
                R, C = _grid_index(mask.shape)
                rows = np.clip(R + mass_off[..., 0], 0, mask.shape[0] - 1)
                cols = np.clip(C + mass_off[..., 1], 0, mask.shape[1] - 1)
                moved = moved or bool((mass_off[mask] != 0).any())
                now = np.zeros_like(mask)
                now[rows[mask], cols[mask]] = True
                if not np.isscalar(value):
                    v = np.full(mask.shape, 0.5)
                    v[rows[mask], cols[mask]] = value[mask]
                    value = v
                mask = now
            cv[6].append((mask, value, w, home, leaf, moved))

        def yang_regions(node: Node, region: Region) -> List[Region]:
            if isinstance(node, ObjectNode):
                bands = [band_region(region, i, node.n_bands, node.orientation)
                         for i, bit in enumerate(node.pattern) if bit > 0]
                return [b for b in bands if not is_empty(b)]
            if isinstance(node, SerialNode):
                kids = content_children(node)
                regions: List[Region] = []
                for i, child in enumerate(kids):
                    strip = band_region(region, i, len(kids), node.split_orientation)
                    regions.extend(yang_regions(child, strip) or [strip])
                return regions
            return [region]

        def render_plan(children: List[Node], region: Region, weight: float, ikey: tuple = ()) -> None:
            if not children or weight < 1e-9 or is_empty(region):
                return
            if len(children) == 1:
                render_node(children[0], region, weight, ikey)
                return
            template, rest = children[0], children[1:]
            p = _clamp(template.gate)
            if p > 1e-9:
                slots = orient_slots(template, region) if ORIENTING_TEMPLATES else \
                    [(sub, None) for sub in yang_regions(template, region)]
                for i, (sub, v) in enumerate(slots):
                    if v is None:
                        render_plan(rest, sub, weight * p, ikey + (i,))
                    elif not is_empty(sub):
                        render_oriented(rest, sub, v, weight * p, ikey + (i,))
            if 1.0 - p > 1e-9:
                render_plan(rest, region, weight * (1.0 - p), ikey + ("w",))

        def render_oriented(children: List[Node], slot: Region, v, weight: float, ikey: tuple = ()) -> None:
            """Draw ``children`` upright in their own frame, then show it in ``slot`` facing ``v``."""
            Hc, Wc, rows, cols, _ = _orientation_map(slot, v)
            frames[0] += 1
            saved = cv[:]
            cv[:] = [np.ones((Hc, Wc)), np.ones((Hc, Wc)), saved[2], (Hc, Wc), saved[4],   # bodies live in the frame
                     np.ones((Hc, Wc)), [] if PURE_CLAIMS else None, (0, 0)]
            render_plan(children, (0, Hc, 0, Wc), weight, ikey)
            frame_yang, frame_yin, frame_left, frame_claims = cv[0], cv[1], cv[5], cv[6]
            cv[:] = saved
            frames[0] -= 1
            r0, r1, c0, c1 = slot
            cv[0][r0:r1, c0:c1] *= frame_yang[rows, cols]
            cv[1][r0:r1, c0:c1] *= frame_yin[rows, cols]
            cv[5][r0:r1, c0:c1] *= frame_left[rows, cols]
            for mask, value, w, home, leaf, moved in frame_claims or ():     # into the slot, home cells too
                m, h = np.zeros(cv[0].shape, dtype=bool), np.zeros(cv[0].shape, dtype=bool)
                m[r0:r1, c0:c1], h[r0:r1, c0:c1] = mask[rows, cols], home[rows, cols]
                if not m.any():
                    continue
                if not np.isscalar(value):
                    full = np.full(cv[0].shape, 0.5)
                    full[r0:r1, c0:c1] = value[rows, cols]
                    value = full
                if not np.isscalar(w):           # a strength that varies by cell (an averaged agent)
                    full = np.zeros(cv[0].shape)
                    full[r0:r1, c0:c1] = w[rows, cols]
                    w = full
                claim(m, value, w, leaf, h, moved)

        def draw_moved(draw, home: Region, region: Region, key: tuple) -> None:
            """``draw(region)`` paints a body's content where it is. Under PROBABILISTIC_MOTION, once a body has
            moved, every cell it has claimed so far (at home or anywhere it has been since) but no longer claims
            reads as known empty, as strongly as it was claimed: a moved body leaves a trail of emptiness behind,
            not "no claim", unless other content covers the cell (``_covered``). The trail lasts until
            ``reset_gates()``."""
            key = (H, W, out_root.node_id, layer, cv[0].shape) + key
            trail = self._trails.get(key)
            if cv[6] is not None:                # claims keep the priority of where the body started
                plain, dr0, dc0 = draw, home[0] - region[0], home[2] - region[2]

                def draw(r):
                    old = cv[7]
                    cv[7] = (old[0] + dr0, old[1] + dc0)
                    plain(r)
                    cv[7] = old
            if not PROBABILISTIC_MOTION or (region == home and trail is None):
                draw(region)
                return

            def alone(r):                        # the content's paint on its own, at r
                saved = cv[:]
                cv[0], cv[1], cv[5] = np.ones_like(saved[0]), np.ones_like(saved[1]), np.ones_like(saved[5])
                cv[6] = [] if saved[6] is not None else None
                draw(r)
                out = cv[0], cv[1], cv[5], cv[6]
                cv[:] = saved
                return out
            if trail is None:
                hy, hn, _, _ = alone(home)
                trail = hy * hn
            ny, nn, nv, nc = alone(region)
            trail = self._trails[key] = np.minimum(trail, ny * nn)
            cv[0] *= ny
            cv[1] *= nn
            cv[5] *= nv                          # trails of bodies inside it
            if nc:
                cv[6].extend(nc)
            vacated = (trail < 1) & (ny == 1) & (nn == 1)
            cv[5][vacated] *= trail[vacated]

        def draw_averaged(kids, r: Region, weight: float, ikey: tuple, leaf: str) -> None:
            """SOFT_EDGES with PURE_CLAIMS: the content drawn once per size at the region's corner, composed on its
            own, then placed at each position (``_placements``) and averaged into one claim whose strength is the
            chance some placement claims the cell."""
            r0, r1, c0, c1 = r
            shape = cv[0].shape
            key = (leaf, r1 - r0, c1 - c0, shape, weight, ikey,
                   tuple(n.gate for k in kids for n in walk(k)))        # its look depends on these, not on where
            cache = self.__dict__.setdefault("_averaged", {})
            ids = {n.node_id for k in kids for n in walk(k)}
            if (skip is not None and ids <= skip) or (only is not None and not ids & only):
                return                           # a probe that leaves the whole body out
            if STATIC_FIELDS and getattr(self, "_seeding", False):
                return                           # a body's content is never part of the substance grid
            usable = (skip is None or not ids & skip) and (only is None or ids <= only)   # a probe leaves none out
            hit = cache.get(key) if usable else None
            if hit is None:                      # at the origin: drawn once per size, then placed and averaged
                Q, QV = np.zeros(shape), np.zeros(shape)
                saved = cv[0], cv[1], cv[6], cv[7]
                drawn = {}
                for hs, ws, dy, dx, prob in _placements(r1 - r0, c1 - c0):
                    if (hs, ws) not in drawn:
                        cv[0], cv[1], cv[6], cv[7] = np.ones(shape), np.ones(shape), [], (0, 0)
                        for child in kids:
                            render_node(child, (0, hs, 0, ws), weight, ikey)
                        v, c = _claim_values(cv[6], shape)
                        e = np.where(c > 0, np.clip(0.5 + (v - 0.5) / np.maximum(c, 1e-12), 0.0, 1.0), 0.5)
                        drawn[hs, ws] = (c, c * e)          # how strongly claimed, and the claimed value
                    c, ce = drawn[hs, ws]
                    Q += prob * _shifted(c, dy, dx)
                    QV += prob * _shifted(ce, dy, dx)
                cv[0], cv[1], cv[6], cv[7] = saved
                hit = (Q, QV)
                if usable:
                    if len(cache) > 2000:
                        cache.clear()
                    cache[key] = hit
            Q, QV = (_shifted(m, r0, c0) for m in hit)
            mask = Q > 1e-9
            if not mask.any():
                return
            V = np.where(mask, QV / np.maximum(Q, 1e-12), 0.5)
            cv[0] *= 1.0 - Q * np.clip(2.0 * V - 1.0, 0.0, 1.0)
            cv[1] *= 1.0 - Q * np.clip(1.0 - 2.0 * V, 0.0, 1.0)
            claim(mask, V, Q, leaf)

        def draw_instance(node, region: Region, weight: float, ikey: tuple) -> None:
            """One instance of a field or body, drawn where it is (its region: where the tree places it)."""
            tok = node.instances.get(ikey)
            home = region
            if cv[2] and tok is not None and tok.position is not None:
                region = tok.rect(region, *cv[3])
            soft = _soft_edge(node, home, cv[3])
            averaged = soft and cv[6] is not None and not any(isinstance(n, GroupNode) for n in walk(node)
                                                              if n is not node)

            def draw(r):
                was_inside = cv[4]
                cv[4] = True
                if averaged:                     # an agent of uncertain extent: averaged over its placements
                    draw_averaged(content_children(node), r, weight * _sg(node), ikey, node.node_id)
                else:                            # (with nested units: at its core)
                    for child in content_children(node):
                        render_node(child, unit_core(r) if soft else r, weight * _sg(node), ikey)
                cv[4] = was_inside
            draw_moved(draw, home, region, (node.node_id, ikey))

        def render_node(node: Node, region: Region, weight: float, ikey: tuple = ()) -> None:
            if weight < 1e-9 or is_empty(region):
                return
            if getattr(node, "role", None) == "observer":
                return                           # perceptual patterns are not part of the imagery
            if isinstance(node, InWindowNode):
                if node.field is None or (layer != "all" and cv[4] != (layer == "bodies")):
                    return
                r0, r1, c0, c1 = region          # the field, resized to the window's cells
                f = node.field[np.arange(r1 - r0) * node.field.shape[0] // (r1 - r0)][
                    :, np.arange(c1 - c0) * node.field.shape[1] // (c1 - c0)]
                w = _clamp(weight * node.gate)
                if cv[6] is not None and w > 0:
                    m, value = np.zeros(cv[0].shape, dtype=bool), np.full(cv[0].shape, 0.5)
                    m[r0:r1, c0:c1], value[r0:r1, c0:c1] = True, f
                    claim(m, value, w, node.node_id)
                if UNKNOWN_BACKGROUND:
                    cv[0][r0:r1, c0:c1] *= 1.0 - w * np.clip(2 * f - 1, 0, 1)
                    cv[1][r0:r1, c0:c1] *= 1.0 - w * np.clip(1 - 2 * f, 0, 1)
                else:
                    cv[0][r0:r1, c0:c1] *= 1.0 - w * f
                return
            if isinstance(node, ObjectNode):
                if UNKNOWN_BACKGROUND and node.is_dynamic:
                    return                       # a motion trigram specifies motion, not a static figure
                if layer != "all" and cv[4] != (layer == "bodies"):
                    return                       # the other layer (mass vs bodies)
                if (skip and node.node_id in skip) or (only is not None and node.node_id not in only):
                    return                       # a solidity probe leaves this leaf out
                if STATIC_FIELDS and cv[6] is not None and getattr(self, "_seeding", False) == bool(cv[4]):
                    return                       # bare leaves are drawn once, into the substance grid
                if cv[6] is not None:            # PURE_CLAIMS: centre axis claims every band, others their figure
                    pure, w = len(set(node.pattern)) == 1, _clamp(weight * node.gate)
                    mask = np.zeros(cv[0].shape, dtype=bool)
                    for i, bit in enumerate(node.pattern):
                        if pure or bit > 0.5:
                            r0, r1, c0, c1 = band_region(region, i, node.n_bands, node.orientation)
                            mask[r0:r1, c0:c1] = True
                            cv[0][r0:r1, c0:c1] *= 1.0 - w * _clamp(2 * bit - 1)
                            cv[1][r0:r1, c0:c1] *= 1.0 - w * _clamp(1 - 2 * bit)
                    if w > 0 and mask.any():
                        claim(mask, float(node.pattern[0]) if pure else 1.0, w, node.node_id)
                    return
                for i, bit in enumerate(node.pattern):
                    r0, r1, c0, c1 = band_region(region, i, node.n_bands, node.orientation)
                    w = _clamp(weight * node.gate)
                    if UNKNOWN_BACKGROUND:
                        cv[0][r0:r1, c0:c1] *= 1.0 - w * _clamp(2 * bit - 1)
                        cv[1][r0:r1, c0:c1] *= 1.0 - w * _clamp(1 - 2 * bit)
                    else:
                        cv[0][r0:r1, c0:c1] *= 1.0 - _clamp(w * (1.0 if bit > 0 else 0.5))
            elif isinstance(node, GroupNode):                    # an instance, drawn where it is
                for key in (_spawned(node, ikey, node.instances) if STATIC_FIELDS else [ikey]):
                    draw_instance(node, region, weight, key)
            elif isinstance(node, BodyNode):
                home = region
                if cv[2] and not isinstance(node, PlanNode) and node.position is not None:
                    region = node.window(*cv[3])

                def draw(r):
                    was_inside = cv[4]
                    cv[4] = was_inside or not isinstance(node, PlanNode)
                    render_plan(content_children(node), r, weight * _sg(node), ikey)
                    cv[4] = was_inside
                draw_moved(draw, home, region, (node.node_id, ikey))
            elif isinstance(node, SerialNode):
                kids = content_children(node)
                for i, child in enumerate(kids):
                    render_node(child, band_region(region, i, len(kids), node.split_orientation),
                                weight * _sg(node), ikey)
            elif isinstance(node, ParallelNode):
                for child in content_children(node):
                    render_node(child, region, weight * _sg(node), ikey)

        render_node(out_root, (0, H, 0, W), 1.0)
        if cv[6] is not None:
            if (STATIC_FIELDS and only is None and layer == "all" and not getattr(self, "_seeding", False)
                    and (H, W) == tuple(self.grid)):
                S = self._substance(out_root.node_id)               # the substance grid, as two claims
                for moved in (False, True):
                    cells = (S["s"] > 0) & (S["moved"] == moved)
                    if cells.any():
                        cv[6].append((cells, S["v"], S["s"], cells, "S*" if moved else "S", moved))
            boundary = self._field_leaves(out_root.node_id) if STATIC_FIELDS else frozenset()
            if values:
                return _claim_values(cv[6], (H, W), boundary)
            yang, yin = _compose_claims(cv[6], (H, W), boundary)
            return yang, yin, cv[5]
        return cv[0], cv[1], cv[5]

    # -- STATIC_FIELDS: the substance grid ---------------------------------------------------------------------

    def _substance(self, root_id: str) -> dict:
        """A tree's substance grid, seeded on first use from its bare leaves (the initial condition): per cell a
        claimed value ``v`` (1 figure, 0 ground, 0.5 unknown), its strength ``s``, whether it arrived by motion
        (``moved``); the motion orders on it; each motion leaf's marks (content it has ordered); and the fields'
        claims, re-asserted every tick (``fe`` value, ``fc`` strength, ``F`` cells)."""
        subs = self.__dict__.setdefault("_sub", {})
        S = subs.get(root_id)
        if S is None:
            H, W = self.grid
            trails, self._seeding = dict(self._trails), True
            try:
                v, c = self._paint(self.grid, root_id, True, values=True)
            finally:
                self._seeding, self._trails = False, trails
            S = subs[root_id] = dict(v=_claimed_value(v, c), s=c, moved=np.zeros((H, W), dtype=bool), orders=[],
                                     marks={}, fe=np.full((H, W), 0.5), fc=np.zeros((H, W)),
                                     F=np.zeros((H, W), dtype=bool))
            self._reset_fields(root_id)
        return S

    def _reset_fields(self, root_id: str) -> None:
        """Every field re-asserts its claims where it now is (a body carries the fields inside it): the grid is
        cleared under them, and their claims are kept for drawing and as a source motion can carry away."""
        S = self._sub[root_id]
        ids = self._field_leaves(root_id)
        if not ids:
            return
        v, c = self._paint_only_values(root_id, ids)
        F = c > 0
        S["fe"], S["fc"], S["F"] = _claimed_value(v, c), c, F
        S["src"] = self._field_sources(root_id)
        S["v"][F], S["s"][F], S["moved"][F] = 0.5, 0.0, False
        for key in S["marks"]:                       # what the reset replaced is no longer ordered or marked
            S["marks"][key] &= ~F
        for o in S["orders"]:
            o[0] = o[0] & ~F
        S["orders"] = [o for o in S["orders"] if o[0].any()]

    def _field_units(self, root_id: str) -> list:
        """STATIC_FIELDS: a tree's fields as (field, its own nodes, carried by a body?, holds a motion leaf?)."""
        def build():
            tree, parents, out = self.tree_index(), self._parents(), {}
            for f in [g for g in self._nodes.values() if isinstance(g, FieldNode)]:
                own, stack = set(), [f]
                while stack:
                    n = stack.pop()
                    own.add(n.node_id)
                    stack.extend(c for c in getattr(n, "children", ())
                                 if not getattr(c, "agency", False) and not isinstance(c, FieldNode))
                carried = any(getattr(a, "agency", False) for a in self._ancestors(f, parents))
                moving = any(isinstance(self._nodes[i], ObjectNode) and self._nodes[i].is_dynamic for i in own)
                out.setdefault(tree.get(f.node_id), []).append((f, frozenset(own), carried, moving))
            return out
        return self._meta("field_units", build).get(root_id, [])

    def _field_sources(self, root_id: str) -> dict:
        """STATIC_FIELDS: the substance cells that motion may carry out of fields. An anchored field's (key None):
        any motion moving its region carries copies away, so it re-sows. A field carried by a body: only its own
        motion leaves (key: the field's id); motion from outside moves the body, field and all, and sows nothing."""
        units = self._field_units(root_id)
        groups = {None: frozenset().union(*[own for _, own, carried, _ in units if not carried] or [frozenset()])}
        groups.update({f.node_id: own for f, own, carried, moving in units if carried and moving})
        out = {}
        for key, ids in groups.items():
            if not ids:
                continue
            v, c = self._paint_only_values(root_id, ids)
            out[key] = (c > 0) & (_claimed_value(v, c) > 0.5)
        return out

    def _paint_only_values(self, root_id: str, ids: frozenset) -> Tuple[np.ndarray, np.ndarray]:
        """STATIC_FIELDS: the claimed values and strengths of only the nodes ``ids`` (trails left as they were),
        memoised by what that drawing depends on: fields are re-asserted every tick, mostly where they were."""
        key = (self._draw_key(root_id, substance=False), ids)
        cache = self.__dict__.setdefault("_only_paints", {})
        hit = cache.get(key)
        if hit is None:
            trails, self._paint_only = dict(self._trails), ids
            try:
                hit = self._paint(self.grid, root_id, True, values=True)
            finally:
                self._paint_only, self._trails = None, trails
            if len(cache) > 256:
                cache.clear()
            cache[key] = hit
        return hit

    def _field_leaves(self, root_id: str) -> frozenset:
        """STATIC_FIELDS: the nodes of a tree's fields, not of the bodies they hold (their claims are the boundary
        condition: they outrank everything they cover, so a void field hides whatever enters it)."""
        def build():
            tree, out = self.tree_index(), {}
            for rid in {tree.get(g.node_id) for g in self._nodes.values() if isinstance(g, FieldNode)}:
                ids, stack = set(), [g for g in self._nodes.values()
                                     if isinstance(g, FieldNode) and tree.get(g.node_id) == rid]
                while stack:
                    n = stack.pop()
                    ids.add(n.node_id)
                    stack.extend(c for c in getattr(n, "children", ()) if not getattr(c, "agency", False))
                out[rid] = frozenset(ids)
            return out
        return self._meta("field_leaves", build).get(root_id, frozenset())

    def _spawners(self) -> List["GroupNode"]:
        """STATIC_FIELDS: bodies held by a field (the nearest unit above them is a field)."""
        def build():
            parents, out = self._parents(), []
            for n in self._nodes.values():
                if not getattr(n, "agency", False):
                    continue
                up = next((a for a in self._ancestors(n, parents) if isinstance(a, GroupNode)), None)
                if isinstance(up, FieldNode):
                    out.append(n)
            return out
        return self._meta("spawners", build)

    def _spawn_bodies(self) -> None:
        """STATIC_FIELDS: a field re-asserts the bodies it holds. For each spawning body and template position,
        instances that came to the same place in the same state merge; then, if none is at home, a new instance is
        placed there, as the first one was."""
        same = lambda t: (tuple(round(v, 9) for v in t.position), tuple(tuple(o) for o in t.orders),
                          tuple(round(v, 6) for v in t.jitter + t.drive))
        for node in self._spawners():
            firsts = [k for k in node.instances if not any(k[i] == "s" for i in range(0, len(k) - 1))]
            for base in firsts:
                first = node.instances[base]
                if first.home is None or first.position is None:
                    continue
                keys = _spawned(node, base, node.instances)
                seen = {}
                for k in keys:                                   # merge indistinguishable instances
                    t = node.instances[k]
                    sig = same(t)
                    if sig in seen and k != base:
                        del node.instances[k]
                    else:
                        seen.setdefault(sig, k)
                keys = _spawned(node, base, node.instances)
                if any(np.allclose(node.instances[k].position, first.home) for k in keys):
                    continue
                n = 1 + max((k[-1] for k in keys if k != base), default=0)
                tok = node.token(base + ("s", n))
                tok.position, tok.extent = list(first.home), list(first.extent)
                tok.anchor, tok.home = first.anchor, list(first.home)

    def _move_substance(self, root_id: str) -> None:
        """Each order on the substance grid advances at most one cell per tick once its credit reaches a cell."""
        S = self._sub.get(root_id)
        if S is None:
            return
        keep, fire = [], []
        for o in S["orders"]:                    # [cells, axis, direction, cells left, rate, credit, delay]
            if o[6] >= 1 - 1e-9:
                o[6] -= 1
                keep.append(o)
                continue
            o[5] += o[4]
            k = min(int(o[5]), 1, o[3])
            o[5] -= k
            o[3] -= k
            if k and _standing():
                fire.append(o)                   # moved together below: stuff pressed from both sides holds
            elif k:
                self._shift_substance(S, o[0], o[2] if o[1] == 0 else 0, o[2] if o[1] == 1 else 0)
            if o[3] > 0:
                keep.append(o)
        none = np.zeros(self.grid, dtype=bool)
        for o in fire:
            against = np.logical_or.reduce([q[0] for q in fire if q[1] == o[1] and q[2] == -o[2]] + [none])
            self._shift_substance(S, o[0], o[2] if o[1] == 0 else 0, o[2] if o[1] == 1 else 0, against)
        S["orders"] = keep

    @staticmethod
    def _shift_substance(S: dict, cells: np.ndarray, dy: int, dx: int, against: np.ndarray = None) -> None:
        """Move the contents of ``cells`` by (dy, dx). A field cell gives a copy of its claim (a source); content
        stays at the board's edge; cells left behind read known empty; content arriving on content that also moved
        piles up (noisy-OR strength), elsewhere it replaces what was there. Orders and marks follow their contents.
        STANDING_MOTION: content pushes the content ahead of it instead, and moves only if its run can (``_stuff_run``;
        ``against``: content moving the other way this tick, which it cannot push)."""
        H, W = cells.shape
        src_s = np.where(S["s"] > 0, S["s"], np.where(S["F"], S["fc"], 0.0))
        src_v = np.where(S["s"] > 0, S["v"], S["fe"])
        if _standing():
            run, blocked = _stuff_run(S, cells & (src_s > 0), dy, dx, against)
            cells = run & ~blocked
        rows, cols = np.nonzero(cells & (src_s > 0))
        tr, tc = rows + dy, cols + dx
        ok = (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)
        rows, cols, tr, tc = rows[ok], cols[ok], tr[ok], tc[ok]
        if not rows.size:
            return
        av, aw = src_v[rows, cols].copy(), src_s[rows, cols].copy()
        S["v"][rows, cols], S["s"][rows, cols], S["moved"][rows, cols] = 0.0, aw, False   # left behind: empty
        pv, pw, pm = S["v"][tr, tc], S["s"][tr, tc], S["moved"][tr, tc]
        both = pm & (pw > 0)
        S["v"][tr, tc] = np.where(both, (pw * pv + aw * av) / np.maximum(pw + aw, 1e-12), av)
        S["s"][tr, tc] = np.where(both, 1.0 - (1.0 - pw) * (1.0 - aw), aw)
        S["moved"][tr, tc] = True
        gone = np.zeros((H, W), dtype=bool)
        gone[rows, cols] = True

        def follow(m):
            out = m & ~gone
            out[tr, tc] |= m[rows, cols]
            return out
        for o in S["orders"]:
            o[0] = follow(o[0])
        for key in list(S["marks"]):
            S["marks"][key] = follow(S["marks"][key])

    def derived_fields(self, shape: Tuple[int, int] = (9, 9)) -> Dict[str, ReceptiveField]:
        """One display field per node (used by the visualizer).

        An object node shows its pattern, pulled towards neutral (0.5) as its
        strength falls. Strength is the scan result for observed nodes and
        the gate for imagined ones. Parallel nodes average their children.
        Serial nodes place their children side by side. Plans show their
        content; a dynamic template can set the axis of a serial content.
        """
        H, W = shape
        full = (0, H, 0, W)
        neutral = lambda: np.full((H, W), 0.5, dtype=float)
        parents = {edge.target.node_id: edge.source for edge in self._solid}
        parents_list = self._parents()
        cache: Dict[str, np.ndarray] = {}

        def strength(node: Node) -> float:
            value = node.activation if self._has_input_context(node, parents_list) else node.gate
            return _clamp(value)

        def resize(values: np.ndarray) -> np.ndarray:
            if values.shape == (H, W):
                return values.copy()
            rows = np.linspace(0, values.shape[0] - 1, H).round().astype(int)
            cols = np.linspace(0, values.shape[1] - 1, W).round().astype(int)
            return values[np.ix_(rows, cols)].astype(float, copy=True)

        def pattern_field(node: ObjectNode) -> np.ndarray:
            result = np.zeros((H, W), dtype=float)
            for i, bit in enumerate(node.pattern):
                r0, r1, c0, c1 = band_region(full, i, node.n_bands, node.orientation)
                result[r0:r1, c0:c1] = _clamp(bit)
            return result

        def render(node: Node, orientation: Optional[Orientation] = None) -> np.ndarray:
            if orientation is None and node.node_id in cache:
                return cache[node.node_id]
            kids = content_children(node)
            if isinstance(node, InputNode):
                values = resize(node.field.activation) if node.field is not None else neutral()
            elif isinstance(node, OutputNode):
                parent = parents.get(node.node_id)
                values = render(parent) if parent is not None else neutral()
            elif isinstance(node, ObjectNode):
                values = 0.5 + strength(node) * (pattern_field(node) - 0.5)
            elif isinstance(node, ParallelNode):
                values = sum(render(c) for c in kids) / len(kids) if kids else neutral()
            elif isinstance(node, SerialNode):
                values = neutral()
                axis = orientation or node.split_orientation
                for i, child in enumerate(kids):
                    r0, r1, c0, c1 = band_region(full, i, len(kids), axis)
                    values[r0:r1, c0:c1] = render(child)[r0:r1, c0:c1]
            elif isinstance(node, BodyNode):
                if not kids:
                    values = neutral()
                else:
                    template, content = kids[0], kids[-1]
                    dynamic = isinstance(template, ObjectNode) and template.is_dynamic
                    values = render(content, template.orientation
                                    if dynamic and isinstance(content, SerialNode) else None)
            else:
                values = np.full((H, W), _clamp(node.activation))
            if orientation is None:
                cache[node.node_id] = values
            return values

        return {node.node_id: ReceptiveField(render(node)) for node in self.nodes}

    # -- introspection -------------------------------------------------

    def depth(self) -> int:
        def _d(n: Node) -> int:
            return 1 + max(map(_d, n.children)) if n.children else 0
        return _d(self.root) if self.root else 0

    def size(self) -> int:
        return len(self._nodes)

    def leaf_nodes(self) -> List[ObjectNode]:
        return list(self._meta("leaf_nodes", lambda: [n for n in self._nodes.values() if isinstance(n, ObjectNode)]))

    def activation_summary(self) -> Dict[str, float]:
        return {nid: n.activation for nid, n in self._nodes.items()}

    def body_states(self) -> Dict[str, Dict]:
        """The position, velocity and extent of every placed moving body."""
        return {
            nid: {"position": list(n.position), "velocity": list(n.velocity), "extent": list(n.extent)}
            for nid, n in ((b.node_id, b) for b in self._body_nodes())
            if not isinstance(n, PlanNode) and n.position is not None
        }

    def __repr__(self) -> str:
        return f"ConceptGraph(name={self.name!r}, size={self.size()}, depth={self.depth()})"

    # -- nested-list format --------------------------------------------

    def to_nested_list(self) -> list:
        """Write ``root`` as ``[type, child, ...]`` / ``[trigram_name, orientation]``. Input windows are left out."""
        def convert(node: Node) -> list:
            if isinstance(node, InputNode):
                return []
            if isinstance(node, ObjectNode):
                return [node.name, node.orientation.value]
            return [node.node_type] + [convert(c) for c in node.children if not isinstance(c, InputNode)]
        return convert(self.root) if self.root is not None else []

    # -- backward-compatible names -------------------------------------

    reset_values = reset_gates

    def attach_field(self, field: ReceptiveField, at_node_id: Optional[str] = None,
                     modality: str = "retina", weight: float = 1.0,
                     field_node_id: Optional[str] = None) -> InputNode:
        fid = field_node_id or f"_tmp_{modality}_{len(self._nodes)}"
        return self.add_field_slot(fid, modality=modality, at_node_id=at_node_id,
                                   initial_weight=weight, initial_field=field)


# ============================================================
# 7. Building graphs from nested lists
# ============================================================

_STRUCTURE_MAP = {
    "serial":   SerialNode,
    "parallel": ParallelNode,
    "body":     BodyNode,
    "plan":     PlanNode,
    "template": TemplateNode,
    "field":    FieldNode,
}


def graph_from_nested_list(tree: list, name: str = "") -> ConceptGraph:
    """Build a graph from ``["serial", ["bottom", 1], ["top", 1]]``-style lists.

    A leaf is ``[trigram_name, orientation_int]``. The root is wrapped in a
    ParallelNode if needed, and an InputNode ``"I"`` is added under it.
    """
    graph = ConceptGraph(name=name)
    counter = 0

    def build(spec) -> Node:
        nonlocal counter
        counter += 1
        nid = f"n{counter}"
        if not isinstance(spec, list) or not spec:
            node = ObjectNode(nid, TRIGRAM_PATTERNS["neutral"])
            graph.add_node(node)
        elif spec[0] in _STRUCTURE_MAP:
            node = _STRUCTURE_MAP[spec[0]](nid)
            graph.add_node(node)
            for child_spec in spec[1:]:
                if not isinstance(child_spec, int):     # skip integer annotations
                    graph.add_solid_edge(node, build(child_spec))
        else:
            orientation = Orientation(int(spec[1]) if len(spec) > 1 else 1)
            pattern = ALL_PATTERNS.get(spec[0], TRIGRAM_PATTERNS["neutral"])
            node = ObjectNode(nid, pattern, orientation, name=spec[0])
            graph.add_node(node)
        return node

    inner = build(tree)
    if isinstance(inner, ParallelNode):
        graph.root = inner
    else:
        graph.root = ParallelNode(f"hub_{name}")
        graph.add_node(graph.root)
        graph.add_solid_edge(graph.root, inner)

    if not any(isinstance(c, InputNode) for c in graph.root.children):
        inp = InputNode("I", modality="retina")
        graph.add_node(inp)
        graph.add_solid_edge(graph.root, inp)
    return graph


# Old names kept so that existing code still imports.
ReceptiveFieldNode = InputNode
ReturnNode = OutputNode
