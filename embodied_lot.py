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

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
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
        if not (0 <= activation.min() and activation.max() <= 1 + 1e-6):
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
        return ReceptiveField(
            self.activation[row_start:row_end, col_start:col_end],
            sub_mot,
            bounds=(br0 + (br1 - br0) * row_start / h, br0 + (br1 - br0) * row_end / h,
                    bc0 + (bc1 - bc0) * col_start / w, bc0 + (bc1 - bc0) * col_end / w),
            root_activation=self._root_activation,
            root_motion=self._root_motion,
        )

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

class Node(ABC):
    """Base node: an activation in [0, 1], a gate (default 1), and solid children."""

    node_type: str = "node"

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.activation: float = 0.0
        self.gate: float = 1.0
        self.children: List["Node"] = []

    @abstractmethod
    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        """Compute and store this node's activation. ``field`` is None when there is no field context."""

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
        self.gate = float(np.clip(initial_weight, 0.0, 1.0))
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
            means.append(float(values.mean()) if values.size > 0 else 0.0)
        return means

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if field is None:
            self.activation = self.gate
        elif all(v == 0.5 for v in self.pattern):
            self.activation = 0.5 * self.gate
        else:
            sims = [1.0 - abs(obs - exp)
                    for obs, exp in zip(self._band_means(field), self.pattern) if exp > 0]
            self.activation = self.gate * float(np.mean(sims)) if sims else 0.0
        return self.activation

    def __repr__(self) -> str:
        dyn = "dynamic" if self.is_dynamic else "static"
        ori = "V" if self.orientation == Orientation.VERTICAL else "H"
        return f"ObjectNode(id={self.node_id!r}, {trigram_symbol(self.pattern)}:{self.name}:{dyn} {ori})"


# ---- Structure nodes ------------------------------------------------------

class StructureNode(Node, ABC):
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
        scanners = content_children(self)

        pooled = [(n.field, n.gate) for n in inputs if n.field is not None and n.gate > 1e-8]
        if field is not None and self.inherited_weight > 1e-8:
            pooled.append((field, self.inherited_weight))

        if not pooled:
            acts = [a for a in (s.evaluate(None) for s in scanners) if a > 0]
            result = float(np.mean(acts)) if acts else 0.0
        elif not scanners:
            result = float(np.clip(sum(w for _, w in pooled) / len(pooled), 0.0, 1.0))
        else:
            total_w = sum(w for _, w in pooled)
            scores = [sum(w * s.evaluate(f) for f, w in pooled) / total_w for s in scanners]
            result = float(np.clip(np.mean(scores), 0.0, 1.0))

        for out in outputs:
            out.gate, out.activation = 1.0, result
        self.activation = result
        return result


class SerialNode(StructureNode):
    """Splits the field into one strip per child and averages the children's scores.

    The mean is intentional: a partial match still counts. Strips are read
    like bands, so for vertical splits children[0] is the bottom strip and
    for horizontal splits children[0] is the left strip. "A above B" is
    therefore ``[B, A]`` with a vertical split. With no field the activation is 0.
    """

    node_type = "serial"

    def __init__(self, node_id: str,
                 split_orientation: Orientation = Orientation.HORIZONTAL) -> None:
        super().__init__(node_id)
        self.split_orientation = split_orientation

    def _make_strip(self, field: ReceptiveField, i: int, n: int) -> ReceptiveField:
        return field.band(i, n, self.split_orientation)

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if not self.children or field is None:
            self.activation = 0.0
            return 0.0
        n = len(self.children)
        self.activation = float(np.mean(
            [child.evaluate(self._make_strip(field, i, n)) for i, child in enumerate(self.children)]
        ))
        return self.activation


class BodyNode(StructureNode):
    """A plan that occupies a moving window of the field.

    Coordinates are normalised to [0, 1]: position = (y, x) centre,
    extent = (h, w), velocity = (dy, dx). Each evaluation carves the window
    from the root field and applies the replacement rule. Children are
    ``[template, ..., content]``, and for each template with p = template.gate:

        activation = p * mean(content on template's yang bands)
                   + (1 - p) * content on the whole window

    With no field this becomes ``(1 - p) * content.gate``, which is how a
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
        H, W = root.shape
        cy, cx = self.position
        eh, ew = self.extent
        h_px, w_px = max(1, round(eh * H)), max(1, round(ew * W))
        r0 = max(0, round(cy * H) - h_px // 2)
        c0 = max(0, round(cx * W) - w_px // 2)
        r1, c1 = min(H, r0 + h_px), min(W, c0 + w_px)
        if r1 <= r0: r1 = min(H, r0 + 1)
        if c1 <= c0: c1 = min(W, c0 + 1)
        return root.subfield(r0, r1, c0, c1)

    # -- motion --------------------------------------------------------

    @staticmethod
    def _template_motion(template: ObjectNode) -> Tuple[float, float]:
        """(dy, dx) of a dynamic trigram: the mean of its signed half-motions along its axis.

        Opposite half-motions cancel out. Numpy coordinates increase downward
        and to the right.
        """
        directions, magnitudes = TRIGRAM_MOTION.get(
            PATTERN_TO_NAME.get(template.pattern), ((0, 0), (0, 0)))
        motion = float(np.mean(np.asarray(directions) * np.asarray(magnitudes)))
        return (motion, 0.0) if template.orientation == Orientation.VERTICAL else (0.0, motion)

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
            self.position[axis] = float(np.clip(
                self.position[axis] + self.velocity[axis], half, 1.0 - half))

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
            if field is None:
                children[0].activation = children[0].gate
                return children[0].gate
            return children[0].evaluate(field)

        template, rest = children[0], children[1:]
        p = float(np.clip(template.gate, 0.0, 1.0))
        template.evaluate(field)

        if field is None:
            # Imagined: there is no region to scan, so the yang term is 0.
            self.activation = (1.0 - p) * self._plan_activate(rest, None)
            return self.activation

        if not isinstance(template, ObjectNode):
            subs = [field]
        elif template.is_dynamic:
            subs = [self._motion_subfield(template, field)]
        else:
            subs = self._yang_subfields(template, field)

        yang_scan = float(np.mean([self._plan_activate(rest, s) for s in subs])) if subs else 0.0
        full_scan = self._plan_activate(rest, field)
        return p * yang_scan + (1.0 - p) * full_scan

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if not self.children:
            self.activation = 0.0
            return 0.0
        window = self._carve_subfield(field)
        if self._grid is None and window is not None:
            self._grid = np.zeros(window.shape)
        self.activation = self._plan_activate(self.children, window)
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

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: Dict[str, Node] = {}
        self._solid: List[SolidEdge] = []
        self._dashed: List[DashedEdge] = []
        self.root: Optional[Node] = None   # primary root; the default target of add_field_slot()

    # -- construction / access ----------------------------------------

    def add_node(self, node: Node) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.node_id!r}.")
        self._nodes[node.node_id] = node

    def add_nodes(self, nodes: List[Node]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_solid_edge(self, parent: Node, child: Node) -> None:
        self._solid.append(SolidEdge(parent, child))
        if child not in parent.children:
            parent.children.append(child)

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

    def remove_node(self, node: Node) -> None:
        """Delete a node, its edges, and its place in any parent's children."""
        self._solid = [e for e in self._solid if node not in (e.source, e.target)]
        self._dashed = [e for e in self._dashed if node not in (e.source, e.target)]
        for other in self._nodes.values():
            if node in other.children:
                other.children.remove(node)
        del self._nodes[node.node_id]

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
        has_parent = {e.target.node_id for e in self._solid}
        return [n for nid, n in self._nodes.items() if nid not in has_parent]

    def _parents(self) -> Dict[str, List[Node]]:
        parents: Dict[str, List[Node]] = defaultdict(list)
        for edge in self._solid:
            parents[edge.target.node_id].append(edge.source)
        return parents

    def _has_input_context(self, node: Node) -> bool:
        """True if ``node`` is observed: some ancestor ``ParallelNode`` owns an ``InputNode``."""
        parents = self._parents()
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
        """Set every gate back to 1. ``evaluate()`` calls this itself and ``step()`` does not."""
        for node in self._nodes.values():
            node.gate = 1.0
            if isinstance(node, PORTALS):
                node.activation = 1.0

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
        return float(np.clip(last.activation, 0.0, 1.0))

    def _propagate_dashed(self) -> None:
        """Set each dashed target's gate, in topological order.

        Portals and imagined (field-free) object nodes copy the new gate
        into their activation straight away, so imagined chains propagate in
        one pass. Observed nodes wait for the next scan.
        """
        sources: Dict[str, List[Node]] = defaultdict(list)
        for edge in self._dashed:
            sources[edge.target.node_id].append(edge.source)
        for node in self._dashed_topological_order():
            srcs = sources.get(node.node_id)
            if not srcs:
                continue
            node.gate = float(np.clip(np.mean([s.activation for s in srcs]), 0.0, 1.0))
            if isinstance(node, PORTALS) or (
                    isinstance(node, ObjectNode) and not self._has_input_context(node)):
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
        result = self._scan()
        self._propagate_dashed()
        self._update_bodies()
        if restore is not None and detach_after:
            restore[0].field = restore[1]
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
                      output_root_id: Optional[str] = None) -> ReceptiveField:
        """Draw the imagined (output) subgraph as a field.

        This follows the replacement rule spatially: a template's yang bands
        receive the content with weight p, and the whole region receives it
        with weight 1 - p. An ObjectNode paints its yang bands at
        ``gate * weight`` and its yin bands at half that (unknown, not absent).
        Overlapping contributions combine by noisy-OR: ``1 - prod(1 - c)``.
        The default root is the last root (the imagined one by convention).
        """
        H, W = shape
        roots = self.roots
        out_root = self._nodes[output_root_id] if output_root_id is not None else roots[-1]
        complement = np.ones((H, W))

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

        def render_plan(children: List[Node], region: Region, weight: float) -> None:
            if not children or weight < 1e-9 or is_empty(region):
                return
            if len(children) == 1:
                render_node(children[0], region, weight)
                return
            template, rest = children[0], children[1:]
            p = float(np.clip(template.gate, 0.0, 1.0))
            if p > 1e-9:
                for sub in yang_regions(template, region):
                    render_plan(rest, sub, weight * p)
            if 1.0 - p > 1e-9:
                render_plan(rest, region, weight * (1.0 - p))

        def render_node(node: Node, region: Region, weight: float) -> None:
            if weight < 1e-9 or is_empty(region):
                return
            if isinstance(node, ObjectNode):
                for i, bit in enumerate(node.pattern):
                    contrib = float(np.clip(weight * node.gate * (1.0 if bit > 0 else 0.5), 0, 1))
                    if contrib >= 1e-9:
                        r0, r1, c0, c1 = band_region(region, i, node.n_bands, node.orientation)
                        complement[r0:r1, c0:c1] *= 1.0 - contrib
            elif isinstance(node, BodyNode):
                render_plan(content_children(node), region, weight)
            elif isinstance(node, SerialNode):
                kids = content_children(node)
                for i, child in enumerate(kids):
                    render_node(child, band_region(region, i, len(kids), node.split_orientation), weight)
            elif isinstance(node, ParallelNode):
                for child in content_children(node):
                    render_node(child, region, weight)

        render_node(out_root, (0, H, 0, W), 1.0)
        return ReceptiveField(np.clip(1.0 - complement, 0.0, 1.0))

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
        cache: Dict[str, np.ndarray] = {}

        def strength(node: Node) -> float:
            value = node.activation if self._has_input_context(node) else node.gate
            return float(np.clip(value, 0.0, 1.0))

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
                result[r0:r1, c0:c1] = float(np.clip(bit, 0.0, 1.0))
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
                values = np.full((H, W), float(np.clip(node.activation, 0, 1)))
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
        """The position, velocity and extent of every moving body."""
        return {
            nid: {"position": list(n.position), "velocity": list(n.velocity), "extent": list(n.extent)}
            for nid, n in self._nodes.items()
            if isinstance(n, BodyNode) and not isinstance(n, PlanNode)
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
