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
from collections import defaultdict, deque
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
_MOTION_KIND = {"top": "bulk", "bottom": "bulk", "upper": "gradient", "lower": "gradient",
                "center": "out", "periphery": "in", "universal": "random"}


def _sg(node) -> float:
    """The gate a structure applies to its composite (1 when GATE_STRUCTURES is off)."""
    return node.gate if GATE_STRUCTURES else 1.0


def _compose(yang_complement: np.ndarray, yin_complement: np.ndarray) -> np.ndarray:
    """A rendered field from the noisy-OR complements of yang and yin paint."""
    if UNKNOWN_BACKGROUND:
        return np.clip(0.5 + 0.5 * (yin_complement - yang_complement), 0.0, 1.0)
    return np.clip(1.0 - yang_complement, 0.0, 1.0)


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
        self.activation = np.clip(activation, 0.0, 1.0).astype(float)
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

    def _tokens(self) -> Dict[str, object]:
        """Every moving unit by id: earlier bodies (the node itself) and group instances."""
        out = {}
        for n in self._nodes.values():
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
                tok = node.token(ikey)
                fh, fw = _frame_shape(chain, (H, W))
                if (place or tok.position is None) and tok.node_id not in placed:   # new instances start in place
                    r0, r1, c0, c1 = rect
                    tok.position = [(r0 + r1) / (2.0 * fh), (c0 + c1) / (2.0 * fw)]
                    tok.extent = [(r1 - r0) / fh, (c1 - c0) / fw]
                    tok.anchor = (r0, c0)
                    placed.add(tok.node_id)
                if tok.position is not None:
                    rect = tok.rect(rect, fh, fw)
                if weight >= 1e-9:
                    views[node.node_id].append((weight, chain, rect))
                    meta[node.node_id].append((ikey, groups))
                for child in content_children(node):
                    visit(child, weight, chain, rect, ikey, groups + (tok.node_id,))
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
        yang, yin, spaces = np.ones(self.grid), np.ones(self.grid), {}
        for r in self.roots:
            key = self._draw_key(r.node_id)
            hit = cache.get(r.node_id)
            if hit is None or hit[0] != key:
                a, b = self._paint_root(self.grid, r.node_id, place_bodies=True)
                hit = cache[r.node_id] = (key, a, b, _compose(a, b))
            _, a, b, spaces[r.node_id] = hit
            yang, yin = yang * a, yin * b
        return spaces, _compose(yang, yin)

    def _draw_key(self, root_id: str) -> tuple:
        """Everything a tree's drawing depends on (besides structure, which is part of the key)."""
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
                MOTION_AWAY_FROM_FIGURE, SEPARATE_TREE_SPACES, self.dynamics, groups, GROUPING_NODES)

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
        fy, fn = self._paint(shape, rid, True, layer="free")
        by, bn = self._paint(shape, rid, True, layer="bodies")
        return _scatter(fy, st["off"]) * by, _scatter(fn, st["off"]) * bn

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
        for b in self._nodes.values():
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
            if MOTION_KINDS:
                self._issue_kinds(node, a, list(zip(keys, views.get(node.node_id, []))), near, parents)
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
        st = self._mass_state(root.node_id)
        H, W = self.grid
        fy, fn = self._paint(self.grid, root.node_id, True, layer="free")
        paint = (fy < 1 - 1e-9) | (fn < 1 - 1e-9)                      # source cells that carry mass
        R, C = _grid_index((H, W))
        where = (R + st["off"][..., 0], C + st["off"][..., 1])          # where each source cell's mass is now
        motion = BodyNode._template_motion(node)
        axis_vec = (1.0, 0.0) if node.orientation == Orientation.VERTICAL else (0.0, 1.0)
        meta = self.view_meta(node.node_id)
        for key, (_, chain, rect) in keyed_views:
            space = _view_mask(self.grid, chain, rect)
            rows, cols = np.nonzero(space)
            if not rows.size:
                continue
            k = int(key.split("#")[1])
            own = meta[k][1] if k < len(meta) else ()                     # a leaf never moves its own groups
            units = getattr(self, "_units", None) or self._tokens()
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

    def motion_state(self) -> tuple:
        """Everything about mass that can change a later frame (for periodicity checks)."""
        out = tuple((rid, st["off"].tobytes(), np.round(st["jit"], 6).tobytes(),
                     tuple((o[0].tobytes(), o[1], o[2], o[3], round(o[4], 6), round(o[5], 6), round(o[6], 6))
                           for o in st["orders"]))
                    for rid, st in sorted(self._mass.items()))
        jit = tuple(tuple(round(v, 6) for v in getattr(n, "jitter", ())) for n in self._nodes.values()
                    if isinstance(n, BodyNode))
        random_now = any(isinstance(n, ObjectNode) and n.is_dynamic and PATTERN_TO_NAME.get(n.pattern) == "universal"
                         for n in self._nodes.values())
        groups = tuple((t.node_id, None if t.position is None else tuple(round(v, 9) for v in t.position),
                        tuple(tuple(o) for o in t.orders), tuple(round(v, 6) for v in t.jitter + t.drive))
                       for t in self._group_tokens())
        return (out, jit, (self._tick if random_now else 0)) + ((groups,) if groups else ())

    def _move_bodies_gameboard(self, views=None) -> None:
        """Each order advances at most one cell per tick once its credit reaches a whole cell; moves add up.

        A body moves on its own frame's cells (the board, or the turned frame it lives in).
        """
        homes = self._moving_bodies(self.node_views(self.grid) if views is None else views, cells=False)
        for body in self._tokens().values():
            if body.position is None:
                continue
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
            kind = _MOTION_KIND.get(PATTERN_TO_NAME.get(node.pattern))
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
                    push = _view_turn(chain, BodyNode._template_motion(node))
                else:
                    continue
                for ax, m in enumerate(_view_unturn(homes[tok.node_id][0], push)):
                    tok.drive[ax] += a * m

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
        if MOTION_KINDS:
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
        return ReceptiveField(_compose(*self._paint_root(shape, output_root_id, place_bodies)))

    def _paint(self, shape: Tuple[int, int], output_root_id: Optional[str],
               place_bodies: bool, layer: str = "all") -> Tuple[np.ndarray, np.ndarray]:
        """The noisy-OR complements ``prod(1 - c)`` of one root's yang and yin paint.

        ``layer`` "free" keeps only paint drawn outside bodies (mass), "bodies" only paint inside them.
        """
        H, W = shape
        roots = self.roots
        out_root = self._nodes[output_root_id] if output_root_id is not None else roots[-1]
        # cv: [yang paint (and, in the earlier renderer, yin paint), yin paint, place bodies, frame size].
        # An oriented slot swaps in a canvas for the content's upright frame (render_oriented).
        cv = [np.ones((H, W)), np.ones((H, W)), place_bodies, (H, W), False]   # ..., frame size, inside a body

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
            saved = cv[:]
            cv[:] = [np.ones((Hc, Wc)), np.ones((Hc, Wc)), saved[2], (Hc, Wc), saved[4]]   # bodies live in the frame
            render_plan(children, (0, Hc, 0, Wc), weight, ikey)
            frame_yang, frame_yin = cv[0], cv[1]
            cv[:] = saved
            r0, r1, c0, c1 = slot
            cv[0][r0:r1, c0:c1] *= frame_yang[rows, cols]
            cv[1][r0:r1, c0:c1] *= frame_yin[rows, cols]

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
                for i, bit in enumerate(node.pattern):
                    r0, r1, c0, c1 = band_region(region, i, node.n_bands, node.orientation)
                    w = _clamp(weight * node.gate)
                    if UNKNOWN_BACKGROUND:
                        cv[0][r0:r1, c0:c1] *= 1.0 - w * _clamp(2 * bit - 1)
                        cv[1][r0:r1, c0:c1] *= 1.0 - w * _clamp(1 - 2 * bit)
                    else:
                        cv[0][r0:r1, c0:c1] *= 1.0 - _clamp(w * (1.0 if bit > 0 else 0.5))
            elif isinstance(node, GroupNode):                    # an instance, drawn where it is
                tok = node.instances.get(ikey)
                if cv[2] and tok is not None and tok.position is not None:
                    region = tok.rect(region, *cv[3])
                was_inside = cv[4]
                cv[4] = True
                for child in content_children(node):
                    render_node(child, region, weight * _sg(node), ikey)
                cv[4] = was_inside
            elif isinstance(node, BodyNode):
                if cv[2] and not isinstance(node, PlanNode) and node.position is not None:
                    region = node.window(*cv[3])
                was_inside = cv[4]
                cv[4] = was_inside or not isinstance(node, PlanNode)
                render_plan(content_children(node), region, weight * _sg(node), ikey)
                cv[4] = was_inside
            elif isinstance(node, SerialNode):
                kids = content_children(node)
                for i, child in enumerate(kids):
                    render_node(child, band_region(region, i, len(kids), node.split_orientation),
                                weight * _sg(node), ikey)
            elif isinstance(node, ParallelNode):
                for child in content_children(node):
                    render_node(child, region, weight * _sg(node), ikey)

        render_node(out_root, (0, H, 0, W), 1.0)
        return cv[0], cv[1]

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
        return [n for n in self._nodes.values() if isinstance(n, ObjectNode)]

    def activation_summary(self) -> Dict[str, float]:
        return {nid: n.activation for nid, n in self._nodes.items()}

    def body_states(self) -> Dict[str, Dict]:
        """The position, velocity and extent of every placed moving body."""
        return {
            nid: {"position": list(n.position), "velocity": list(n.velocity), "extent": list(n.extent)}
            for nid, n in self._nodes.items()
            if isinstance(n, BodyNode) and not isinstance(n, PlanNode) and n.position is not None
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
