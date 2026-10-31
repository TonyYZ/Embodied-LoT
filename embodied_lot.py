"""Core graph model for Embodied Language of Thought.

A concept is a ``ConceptGraph`` of perceptual/attentional operators:

* ``ObjectNode``    trigram scanner (an image schema over three bands)
* ``ParallelNode``  attention hub: pools ``InputNode`` fields, averages scanners
* ``SerialNode``    splits the field into strips, averages children on them
* ``PlanNode``      template/content replacement rule
* ``BodyNode``      a plan with a position and extent that moves (``field`` / ``body`` units, ``GroupNode``)
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

# Halves (two bands; index 0 the bottom or left half, as above): static only, a template's half or a half drawn
# filled (no lone-band line). Added 30 Oct 2026.
HALF_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "near": (1, 0),                 # the bottom half (vertical) or the left half (horizontal)
    "far":  (0, 1),                 # the top half or the right half
}
STATIC_ONLY = frozenset(HALF_PATTERNS)   # leaves without a moving version

MONOGRAM_PATTERNS: Dict[str, Tuple[float, ...]] = {
    "yin":  (0,),
    "yang": (1,),
}

ALL_PATTERNS = {**TRIGRAM_PATTERNS, **HALF_PATTERNS, **MONOGRAM_PATTERNS}
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


# The language (gameboard; docs/language_semantics.md has the full account, and legacy/all_languages the engine with
# every earlier language as switches). A program is a tree and nothing else: no numbers in it.
#
# * Drawing. A cell nobody claims is unknown (0.5). The centre-axis patterns (void, neutral, universal) claim the
#   substance in every band (absent, unknown, present); the others claim only their figure, a lone figure band as a
#   salient line (a basket is an outline). Claims are painted general first: the more specific claim wins where it
#   lies inside another; a field's claims outrank everything they cover. A structure's gate weighs its subtree.
# * Groups. ``template`` copies its material into its template's slots (a dynamic template turns it to face its
#   motion); ``field`` is a rigid unit; ``body`` is a field with agency. Bodies fill the region the tree gives them.
#   Separate trees are separate spaces, linked only by dashed edges.
# * Substance. Bare leaves are the initial condition, drawn once into a substance grid that motion moves like sand;
#   a field is a boundary condition re-asserted every tick (a source or a drain) and spawns the bodies it holds; a
#   body carries its claims.
# * Motion. Every motion leaf is a standing rule: each tick it gives a mission ("beyond me") to what lies in its region
#   and has none from it: move until past the region. Kinds: top/bottom bulk, upper/lower in layers, center out,
#   periphery in. Motion leaves claim motion on the content scale (void* still, neutral* unknown, the rest moving):
#   a cell leaning still pins what is in it; leaning moving it drives at each leaf's rate times the share left.
#   universal* / neutral* are latent directions, mixed in the likelihood (``motion_states``). What moves leaves known
#   emptiness behind. The board's edge is open: bodies may leave and come back, solid past the edge.
# * Perception. An observer (a dashed source) sees last tick's drawing of its own tree and its optical flow; the first
#   tick sees the board as drawn. A dashed effect takes one tick, but one into a motion leaf acts in the tick its
#   sources perceive. A tree whose top node is a dashed source measures its own figure.
#
# The one switch left: built-in solidity. With SOLID_FIGURE a moving body cannot enter what its tree makes more likely
# figure than not, and a push is passed along (``_chain_move``); without it, solidity is whatever the program states.
SOLID_FIGURE = True
ONSET_THRESHOLD = 0.5      # a motion leaf is on when its activation passes this (strictly)
# Drawing below the board's resolution. With SUBCELL = s > 1 a tree is drawn on a board s times finer, so a region
# keeps its fractions of a cell, and each claim is pooled back: a cell is claimed as strongly as the claim covers it
# (a third of a cell: a third of the strength). Content smaller than a cell then neither vanishes nor speaks as loud as
# a whole cell. A lone band's line stays one board cell thick (thinner only where its region is). Motion, bodies and
# substance stay on the board. 1: whole cells, as before.
SUBCELL = 1
# An output window's output. Off: its region stretched to the full board (it loses its size and place). On: the board with
# only that region shown, where it is; every other cell unclaimed.
WINDOW_CROP = False
# Moving below the board's resolution. With FINE_MOTION = s > 1 a program runs on a board s times finer, so a tick's move is
# 1/s of a board cell; lines stay a board cell thick, and each output is averaged back to the board.
FINE_MOTION = 1


def _clip(region: "Region", shape: Tuple[int, int]) -> "Region":
    """The part of ``region`` on an H x W canvas (a body's window may reach past the board)."""
    H, W = shape
    r0, r1, c0, c1 = region
    return min(max(r0, 0), H), min(max(r1, 0), H), min(max(c0, 0), W), min(max(c1, 0), W)


def _lone_bands(pattern) -> set:
    """The figure bands with no figure band beside them (each a salient presence, claimed as a line). A half's
    figure band is the half itself, never a line."""
    if len(pattern) == 2:
        return set()
    fig = [b > 0.5 for b in pattern]
    return {i for i, f in enumerate(fig) if f and not (i > 0 and fig[i - 1]) and not (i + 1 < len(fig) and fig[i + 1])}


@lru_cache(maxsize=256)
def _blank(h: int, w: int) -> np.ndarray:
    """A shared all-ones canvas, read-only."""
    out = np.ones((h, w))
    out.flags.writeable = False
    return out


@lru_cache(maxsize=8192)
def _leaf_cells(pattern, n_bands: int, orientation: "Orientation", region: "Region", shape: Tuple[int, int], thick: int):
    """What a still leaf draws in ``region``: its cells (shared, read-only), its bands' (cells, bit), whether it is
    pure, and whether any cell is on the canvas. A pure leaf claims every band (its centre axis), others their figure."""
    pure = len(set(pattern)) == 1
    lone = set() if pure else _lone_bands(pattern)
    mask, bands = np.zeros(shape, dtype=bool), []
    for i, bit in enumerate(pattern):
        if pure or bit > 0.5:
            cells = _clip(_band_line(region, i, n_bands, orientation, thick) if i in lone
                          else band_region(region, i, n_bands, orientation), shape)
            mask[cells[0]:cells[1], cells[2]:cells[3]] = True
            bands.append((cells, bit))
    mask.flags.writeable = False
    return mask, tuple(bands), pure, bool(mask.any())


@lru_cache(maxsize=8192)
def _leaf_home(pattern, n_bands: int, orientation: "Orientation", region: "Region", shape: Tuple[int, int], thick: int,
               dr: int, dc: int) -> np.ndarray:
    """A still leaf's cells moved back by (dr, dc) to where the enclosing body started (shared, read-only)."""
    out = _moved_home(_leaf_cells(pattern, n_bands, orientation, region, shape, thick)[0], dr, dc)
    out.flags.writeable = False
    return out


def _moved_home(cells: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``cells`` moved by (dr, dc); what leaves the canvas is dropped."""
    rs, cs = np.nonzero(cells)
    keep = (rs + dr >= 0) & (rs + dr < cells.shape[0]) & (cs + dc >= 0) & (cs + dc < cells.shape[1])
    out = np.zeros_like(cells)
    out[rs[keep] + dr, cs[keep] + dc] = True
    return out


def _band_line(region: "Region", i: int, n: int, orientation: "Orientation", thick: int = 1) -> "Region":
    """Band ``i``'s line, ``thick`` cells across (no more than the band): along the region's edge for an edge band,
    through the axis for a middle one."""
    r0, r1, c0, c1 = band_region(region, i, n, orientation)
    if r1 <= r0 or c1 <= c0:
        return r0, r1, c0, c1

    def line(a, b, last):                                # within [a, b): the edge at a, the edge at b, or the middle
        t = min(thick, b - a)
        start = a if last == "a" else b - t if last == "b" else (a + b - t) // 2
        return start, start + t
    if orientation == Orientation.VERTICAL:              # vertical bands count from the bottom
        rows = line(r0, r1, "b" if i == 0 else "a" if i == n - 1 else "mid")
        return rows[0], rows[1], c0, c1
    cols = line(c0, c1, "a" if i == 0 else "b" if i == n - 1 else "mid")
    return r0, r1, cols[0], cols[1]


def _pooled(claim: tuple, s: int) -> tuple:
    """A claim on a board ``s`` times finer, back on the board: each cell claimed as strongly as the claim covers it
    (its strength and value averaged over the sub-cells it covers)."""
    mask, value, w, home, leaf, moved = claim
    H, W = mask.shape[0] // s, mask.shape[1] // s
    blocks = lambda a: np.asarray(a, dtype=float).reshape(H, s, W, s).sum(axis=(1, 3))
    n = blocks(mask)
    over = lambda x: x if np.isscalar(x) else np.where(n > 0, blocks(np.where(mask, x, 0.0)) / np.maximum(n, 1), 0.5)
    return n > 0, over(value), (n / (s * s)) * over(w), blocks(home) > 0, leaf, moved


def _spawned(node, ikey: tuple, instances) -> List[tuple]:
    """The keys of a unit's instances at template position ``ikey``: the first, then those a field
    spawned (``ikey + ("s", n)``)."""
    n = len(ikey)
    return [ikey] + sorted((k for k in instances if len(k) == n + 2 and k[:n] == ikey and k[n] == "s"),
                           key=lambda k: k[-1])


_MOTION_KIND = {"top": "bulk", "bottom": "bulk", "upper": "gradient", "lower": "gradient",
                "center": "out", "periphery": "in", "universal": "random"}


def _spawn_index(instances) -> Dict[tuple, List[tuple]]:
    """Template position -> the keys of the instances a field spawned there, in ``_spawned``'s order."""
    out: Dict[tuple, List[tuple]] = {}
    for k in instances:
        if len(k) >= 2 and k[-2] == "s":
            out.setdefault(k[:-2], []).append(k)
    for keys in out.values():
        keys.sort(key=lambda k: k[-1])
    return out


def _sg(node) -> float:
    """The gate a structure applies to its composite."""
    return node.gate


def _compose(yang_complement: np.ndarray, yin_complement: np.ndarray) -> np.ndarray:
    """A rendered field from the noisy-OR complements of yang and yin paint."""
    return np.clip(0.5 + 0.5 * (yin_complement - yang_complement), 0.0, 1.0)


def _covered(yang_complement: np.ndarray, yin_complement: np.ndarray, left: np.ndarray) -> np.ndarray:
    """The yin complement once emptiness left behind by motion (``left``) is added
    where nothing else claims the cell: content that moves into a vacated cell covers it."""
    if left.min() >= 1.0:
        return yin_complement
    free = (yang_complement >= 1.0) & (yin_complement >= 1.0)
    return np.where(free, yin_complement * left, yin_complement)


def _compose_claims(claims: list, shape: Tuple[int, int], boundary=frozenset()) -> Tuple[np.ndarray, np.ndarray]:
    """Yang and yin complements of ``_claim_values``: they compose (``_compose``) to its value and
    count as claimed (``_covered``) as strongly as claims cover the cell."""
    if not claims:
        return np.ones(shape), np.ones(shape)
    v, claimed = _claim_values(claims, shape, boundary)
    u = (1.0 - np.abs(2.0 * v - 1.0)) * claimed
    return 1.0 - np.maximum(0.0, 2.0 * v - 1.0) - u, 1.0 - np.maximum(0.0, 1.0 - 2.0 * v) - u


def _claim_values(claims: list, shape: Tuple[int, int], boundary=frozenset()) -> Tuple[np.ndarray, np.ndarray]:
    """Each cell's value, and how strongly claims cover it, from claims
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
    flat = lambda x: x if np.isscalar(x) else (np.shape(x), np.asarray(x, dtype=float).tobytes())
    key = (tuple(shape), tuple((c[0].tobytes(), flat(c[1]), flat(c[2]), c[3].tobytes(), c[4], c[5], c[4] in boundary)
                               for c in claims))
    hit = _claims_memo.get(key)
    if hit is None:                              # memoised: the same claims recur tick after tick
        if len(_claims_memo) > 4096:
            _claims_memo.clear()
        hit = _claims_memo[key] = _claim_values_of(claims, shape, boundary)
    return hit[0].copy(), hit[1].copy()


_claims_memo: Dict[tuple, tuple] = {}


def _claim_values_in(claims: list, shape: Tuple[int, int], boundary, crop: "Region") -> Tuple[np.ndarray, np.ndarray]:
    """``_claim_values`` read only in ``crop``. A cell's value depends only on the claims covering it and on the
    extents of their leaves, which all the claims make; a claim missing a cell adds 0 to its sums and 1 to its
    products, so leaving out the claims that miss the crop changes no bit of it."""
    r0, r1, c0, c1 = crop
    extent: Dict[str, np.ndarray] = {}
    seen, near = set(), []
    for c in claims:
        k = (c[4], id(c[3]))
        if k not in seen:                        # (copies of a leaf mostly share their home cells)
            seen.add(k)
            extent[c[4]] = extent[c[4]] | c[3] if c[4] in extent else c[3]
        cells = c[0][r0:r1, c0:c1]
        if cells.any():
            near.append((cells, c[1] if np.isscalar(c[1]) else c[1][r0:r1, c0:c1],
                         c[2] if np.isscalar(c[2]) else c[2][r0:r1, c0:c1], c[3], c[4], c[5]))
    if not near:
        return np.full((r1 - r0, c1 - c0), 0.5), np.zeros((r1 - r0, c1 - c0))
    shape = (r1 - r0, c1 - c0)
    if not all(np.isscalar(c[1]) and np.isscalar(c[2]) for c in near):
        return _claim_values_of(near, shape, boundary, extent)
    if len(near) == 1:                           # one claim: nothing to be nested in
        key = (shape, near[0][0].tobytes(), near[0][1], near[0][2])
    else:                                        # all that composing them reads
        key = (shape, tuple((c[0].tobytes(), c[1], c[2], extent[c[4]].tobytes(), c[5], c[4] in boundary) for c in near))
    hit = _near_claims.get(key)
    if hit is None:
        if len(_near_claims) > 4096:
            _near_claims.clear()
        hit = _near_claims[key] = _claim_values_of(near, shape, boundary, extent)
    return hit[0].copy(), hit[1].copy()


_near_claims: Dict[tuple, tuple] = {}


def _claim_values_of(claims: list, shape: Tuple[int, int], boundary, extent: Dict[str, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    M = np.array([c[0] for c in claims])                                     # (n, H, W) cells
    V, w = (_per_claim([c[k] for c in claims], shape) for k in (1, 2))
    if extent is None:
        extent = {}
        for c in claims:
            extent[c[4]] = extent[c[4]] | c[3] if c[4] in extent else c[3]
    if len(claims) == 1:                         # nothing to be nested in
        depth = np.zeros(M.shape, dtype=int)
    else:
        P = np.array([extent[c[4]].ravel() for c in claims])                 # (n, H * W) the leaf's extent
        outside = (P[:, None, :] & ~P[None, :, :]).any(-1)                   # [i, j]: some of i's cells not j's
        moved = np.array([c[5] for c in claims])
        first = moved[:, None] & ~moved[None, :]                             # [i, j]: i moved, j did not
        inside = ((~outside & outside.T) | first) & ~first.T                 # [i, j]: i is nested in j
        if boundary:                             # a field's claims outrank everything they cover
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


def _solid_stuff(S: dict) -> np.ndarray:
    """Where substance more likely than not is: loose stuff, and a field's figure where its claim
    stands (a field's mass is mass like any other; pushed out, it leaves a copy and the field re-asserts behind)."""
    return np.where(S["s"] > 0, (S["v"] > 0.5), S["F"] & (S["fe"] > 0.5) & (S["fc"] > 0))


def _stuff_run(S: dict, cells: np.ndarray, dy: int, dx: int, against: np.ndarray = None,
               drive: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Substance moving by (dy, dx) pushes the substance ahead of it, a field's figure included.
    Returns the run (``cells`` and every substance cell in its way, transitively) and its blocked cells: those whose
    target is off the board or in ``against`` (content pushing the other way), those pinned at least as strongly as
    ``drive``, and those whose target is blocked. Claims of absence are not pushed; they are replaced."""
    against = np.zeros_like(cells) if against is None else against & ~cells
    solid = _solid_stuff(S) & ~against
    run = cells.copy()
    while True:
        ahead = _moved_mask(run, dy, dx) & solid & ~run
        if not ahead.any():
            break
        run |= ahead
    lands = lambda m: _moved_mask(m, -dy, -dx)                # the cells whose target is in m
    blocked = run & (lands(against & ~run) | _held(S, drive))   # (it may leave the board)
    while blocked.any():
        more = run & lands(blocked) & ~blocked
        if not more.any():
            break
        blocked |= more
    return run, blocked


def _held(S: dict, drive: float) -> np.ndarray:
    """The cells pinned at least as strongly as a push of strength ``drive`` (none without pins)."""
    pin = S.get("pin")
    return np.zeros(S["s"].shape, dtype=bool) if pin is None else (pin > 0) & (pin >= drive - 1e-9)


_UNKNOWN_MOTION = {TRIGRAM_PATTERNS["universal"]: "universal", TRIGRAM_PATTERNS["neutral"]: "neutral"}


def unknown_value() -> float:
    """What "not known yet" reads as."""
    return 0.5


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
    H, W = field.activation.shape
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:            # only the region's cells on the board count
        R, C = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        on = (R >= 0) & (R < H) & (C >= 0) & (C < W)
        rows, cols = rows[on], cols[on]
        a = field.activation[R[on], C[on]]
        m = None if field.motion is None else field.motion[R[on], C[on]]
    else:
        a = field.activation[r0:r1, c0:c1]
        m = None if field.motion is None else field.motion[r0:r1, c0:c1]
    total, count = np.zeros((Hc, Wc)), np.zeros((Hc, Wc))
    np.add.at(total, (rows, cols), a)
    np.add.at(count, (rows, cols), 1.0)
    activation = np.where(count > 0, total / np.maximum(count, 1.0), unknown_value())
    motion = None
    if m is not None:
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
    """The board cells a view covers: those whose place in the last frame of ``chain`` falls in ``rect`` (read-only;
    shared)."""
    return _view_mask_of(tuple(shape), chain, tuple(rect))


@lru_cache(maxsize=65536)
def _view_mask_of(shape: Tuple[int, int], chain, rect: Region) -> np.ndarray:
    R, C = _grid_index(shape)
    ok = np.ones(shape, dtype=bool)
    for (r0, r1, c0, c1), v in chain:
        ok &= (R >= r0) & (R < r1) & (C >= c0) & (C < c1)
        _, _, rows, cols, _ = _orientation_map((r0, r1, c0, c1), v)
        R2, C2 = np.zeros_like(R), np.zeros_like(C)
        R2[ok], C2[ok] = rows[R[ok] - r0, C[ok] - c0], cols[R[ok] - r0, C[ok] - c0]
        R, C = R2, C2
    r0, r1, c0, c1 = rect
    out = ok & (R >= r0) & (R < r1) & (C >= c0) & (C < c1)
    out.flags.writeable = False
    return out


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


def _has_dynamic(node: "Node") -> bool:
    return any(isinstance(n, ObjectNode) and n.is_dynamic for n in walk(node))


def orient_slots(template: "Node", region: Region) -> List[Tuple[Region, Optional[Tuple[float, float]]]]:
    """Where a template puts its content, and which way the content faces there.

    Returns ``(slot, direction)`` pairs; direction None means upright. Static parts give the
    usual slots (a leaf's figure, a serial's strips): adjacent yang bands are one figure, so a leaf
    gives one slot per run of them (``1 1 0``: one slot over two thirds; ``1 0 1``: two slots). A dynamic
    leaf, or any non-serial structure, gives one slot over its region facing the sum of its dynamic leaves' motions.
    """
    if isinstance(template, ObjectNode) and not template.is_dynamic:
        runs, run = [], []
        for i, bit in enumerate(template.pattern):
            if bit > 0:
                run.append(band_region(region, i, template.n_bands, template.orientation))
            elif run:
                runs.append(run); run = []
        if run:
            runs.append(run)
        slots = [(min(b[0] for b in r), max(b[1] for b in r), min(b[2] for b in r), max(b[3] for b in r)) for r in runs]
        return [(s, None) for s in slots if not is_empty(s)]
    if isinstance(template, SerialNode):
        kids = content_children(template)
        slots: List[Tuple[Region, Optional[Tuple[float, float]]]] = []
        for i, child in enumerate(kids):
            strip = band_region(region, i, len(kids), template.split_orientation)
            slots.extend(orient_slots(child, strip) or [(strip, None)])
        return slots
    dy, dx = _facing(template)
    return [(region, None if abs(dy) < 1e-12 and abs(dx) < 1e-12 else (dy, dx))]


def _facing(template: "Node", strength: float = 1.0, top: bool = True) -> Tuple[float, float]:
    """Which way a template's content faces: its dynamic leaves' motions, each weighted by how
    strongly it acts (its gate times the gates of the structures between it and the template)."""
    if isinstance(template, ObjectNode):
        if not template.is_dynamic:
            return 0.0, 0.0
        my, mx = BodyNode._template_motion(template)
        w = strength * template.gate
        return w * my, w * mx
    dy = dx = 0.0
    for c in getattr(template, "children", ()):                # the template's own gate is not a turn (it
        cy, cx = _facing(c, strength * (1.0 if top else _sg(template)), False)   # weighs turned against upright)
        dy, dx = dy + cy, dx + cx
    return dy, dx


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
        """Crop a rectangle, keeping track of its bounds in the root field. A rectangle reaching past the field
        (a body past the board) sees unknown there, and no motion."""
        H, W = self.activation.shape
        if row_start < 0 or col_start < 0 or row_end > H or col_end > W:
            q0, q1, p0, p1 = _clip((row_start, row_end, col_start, col_end), (H, W))
            at = (slice(q0 - row_start, q0 - row_start + max(0, q1 - q0)),
                  slice(p0 - col_start, p0 - col_start + max(0, p1 - p0)))
            act = blank_board((max(0, row_end - row_start), max(0, col_end - col_start)))
            act[at] = self.activation[q0:q1, p0:p1]
            mot = None if self.motion is None else np.zeros(act.shape + (2,))
            if mot is not None:
                mot[at] = self.motion[q0:q1, p0:p1]
        else:
            act = self.activation[row_start:row_end, col_start:col_end]
            mot = None if self.motion is None else self.motion[row_start:row_end, col_start:col_end]
        br0, br1, bc0, bc1 = self.bounds
        h, w = max(self.height, 1), max(self.width, 1)
        sub = ReceptiveField.__new__(ReceptiveField)       # the parent is already checked and clipped
        sub.activation = act
        sub.motion = mot
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


_MISSING = object()
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
        means = []
        for i in range(self.n_bands):
            r0, r1, c0, c1 = band_region(field.region, i, self.n_bands, self.orientation)
            values = field.activation[r0:r1, c0:c1]
            means.append(float(values.sum()) / values.size if values.size > 0 else 0.0)   # == values.mean()
        return means

    def _flow_match(self, field: ReceptiveField) -> float:
        """How well the motion in view matches this trigram's motion, band by band and signed (cells
        per tick along its axis): bulk motion everything one way, out away from the centre, in towards it (its centre
        band is not checked), universal any motion, void none, neutral unknown."""
        name = PATTERN_TO_NAME.get(self.pattern)
        if name == "neutral":
            return 0.5
        kind = _MOTION_KIND.get(name)
        axis = 0 if self.orientation == Orientation.VERTICAL else 1
        centre = (field.height if axis == 0 else field.width) / 2.0
        bulk = float(np.sign(BodyNode._template_motion(self)[axis]))
        sims = []
        for i in range(self.n_bands):
            r0, r1, c0, c1 = band_region(field.region, i, self.n_bands, self.orientation)
            m = field.motion[r0:r1, c0:c1, axis]
            if not m.size:
                continue
            seen = float(m.sum()) / m.size
            if kind == "random":
                sims.append(min(1.0, abs(seen)))
                continue
            if name == "void":
                expected = 0.0
            elif kind in ("bulk", "gradient"):
                expected = bulk
            else:
                side = float(np.sign(((r0 + r1) if axis == 0 else (c0 + c1)) / 2.0 - centre))
                if side == 0:
                    continue                             # in / out: the centre band does not move
                expected = side if kind == "out" else -side
            sims.append(1.0 - min(1.0, abs(seen - expected)))
        return _mean(sims) if sims else 0.0

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if field is None:
            self.activation = self.gate
        elif self.is_dynamic:                          # motion: the flow in view (none seen yet: unknown)
            self.activation = self.gate * (self._flow_match(field) if field.motion is not None else 0.5)
        elif len(set(self.pattern)) == 1:              # centre axis: the substance in every band
            target = self.pattern[0]
            sims = [1.0 - 2.0 * abs(obs - 0.5) if target == 0.5 else 1.0 - abs(obs - target)
                    for obs in self._band_means(field)]
            self.activation = self.gate * _mean(sims)
        else:
            lone = _lone_bands(self.pattern)           # a lone band: a salient line anywhere in it will do
            means = [self._line_max(field, i) if i in lone else m for i, m in enumerate(self._band_means(field))]
            sims = [1.0 - abs(obs - exp) for obs, exp in zip(means, self.pattern) if exp > 0]
            self.activation = self.gate * _mean(sims) if sims else 0.0
        return self.activation

    def _line_max(self, field: ReceptiveField, i: int) -> float:
        """The most salient line of band ``i`` (rows of a vertical pattern's band, columns of a
        horizontal one's): the mean of its cells."""
        r0, r1, c0, c1 = band_region(field.region, i, self.n_bands, self.orientation)
        values = field.activation[r0:r1, c0:c1]
        if not values.size:
            return 0.0
        return float(values.mean(axis=1 if self.orientation == Orientation.VERTICAL else 0).max())

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
    evaluated field-free and the activation is their mean.
    """

    node_type = "serial"

    def __init__(self, node_id: str,
                 split_orientation: Orientation = Orientation.HORIZONTAL) -> None:
        super().__init__(node_id)
        self.split_orientation = split_orientation

    def _make_strip(self, field: ReceptiveField, i: int, n: int) -> ReceptiveField:
        return field.band(i, n, self.split_orientation)

    def evaluate(self, field: Optional[ReceptiveField] = None) -> float:
        if field is None and self.children:
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
    """

    node_type = "body"

    def __init__(self, node_id: str,
                 position: Optional[Tuple[float, float]] = None,
                 extent: Optional[Tuple[float, float]] = None,
                 subfield_shape: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(node_id)
        self.position = list(position) if position is not None else None
        self.extent = list(extent) if extent is not None else None
        self.velocity = [0.0, 0.0]               # its last move, per tick (board fractions)
        self.subfield_shape = subfield_shape   # used by PlanNode only
        self.orders: List[list] = []             # gameboard: [axis, direction, cells left, rate, credit, leaf id, delay]
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
        return root.subfield(*_clip(self.window(*root.shape), root.shape))

    def window(self, H: int, W: int) -> Region:
        """The body's rectangle in an H x W root field: its whole window, which may reach past the board (it is cut
        off where it is drawn). Its position is its exact centre."""
        cy, cx = self.position
        eh, ew = self.extent
        h_px, w_px = max(1, round(eh * H)), max(1, round(ew * W))
        r0, c0 = round(cy * H - h_px / 2), round(cx * W - w_px / 2)
        return r0, r0 + h_px, c0, c0 + w_px

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
        return 0.0, -motion                    # away from the figure on both axes

    # -- replacement rule ----------------------------------------------

    def _yang_subfields(self, template: ObjectNode, field: ReceptiveField) -> List[ReceptiveField]:
        """The sub-fields under the template's yang bands. Empty for an all-yin template."""
        n = template.n_bands
        return [field.band(i, n, template.orientation)
                for i, bit in enumerate(template.pattern) if bit > 0]

    def _plan_activate(self, children: List[Node], field: Optional[ReceptiveField]) -> float:
        """Apply the replacement rule recursively: ``children[0]`` is a template and the rest is the plan's content."""
        if len(children) == 1:
            return children[0].evaluate(field)      # imagined leaves return their gate

        template, rest = children[0], children[1:]
        p = _clamp(template.gate)
        template.evaluate(field)
        orienting = _has_dynamic(template)          # a dynamic template turns its content to face its motion

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


class TemplateNode(PlanNode):
    """Copies its material (last child) into the slots of its templates; groups nothing."""

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
        self.velocity = [0.0, 0.0]
        self.orders: List[list] = []
        self.anchor: Optional[Tuple[int, int]] = None   # where its region started when placed (frame cells)
        self.home: Optional[List[float]] = None          # where it was placed (where a field spawns it)

    def rect(self, region: Region, fh: int, fw: int) -> Region:
        """Its window, carried along with the region it receives: a nested instance follows its parent."""
        r0, r1, c0, c1 = self.window(fh, fw)
        if self.anchor is not None:
            dr, dc = region[0] - self.anchor[0], region[2] - self.anchor[1]
            r0, r1, c0, c1 = r0 + dr, r1 + dr, c0 + dc, c1 + dc
        return r0, r1, c0, c1


class GroupNode(BodyNode):
    """A rigid unit over its child. Each copy a template makes of it is a separate instance."""

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
        step():     one tick of the gameboard (``_step_gameboard``; stateful)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._nodes: Dict[str, Node] = {}
        self._solid: List[SolidEdge] = []
        self._dashed: List[DashedEdge] = []
        self.root: Optional[Node] = None   # primary root; the default target of add_field_slot()
        self.grid: Tuple[int, int] = (12, 12)   # the board, in cells
        self._imagery: Optional[np.ndarray] = None
        self._prev_act: Dict[str, float] = {}
        self._inside: Dict[str, set] = {}        # bodies inside each switched-on motion leaf's region
        self._tick = 0
        self._trails: Dict[tuple, np.ndarray] = {}   # probabilistic motion: each moved body's strongest claim per cell so far
        self._version = 0                        # bumped by every structural change (metadata cache key)
        self._meta_key = None
        self._meta_cache: Dict[str, object] = {}

    def _meta(self, name: str, build):
        """Structure-derived metadata (roots, parents, roles ...), rebuilt only after the structure changes."""
        k = self._meta_key
        if (k is None or k[0] != self._version or k[1] != len(self._nodes) or k[2] != len(self._solid)
                or k[3] != len(self._dashed) or k[4] != id(self._solid) or k[5] != id(self._dashed)):
            self._meta_key = (self._version, len(self._nodes), len(self._solid), len(self._dashed), id(self._solid),
                              id(self._dashed))
            self._meta_cache = {}
        cache = self._meta_cache
        hit = cache.get(name, _MISSING)
        if hit is _MISSING:
            hit = cache[name] = build()
        return hit

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

        Every body is first centred in its tree region (``place_bodies``). Dashed targets start unknown (gate
        ``unknown_value()``: not yet conditioned), and pending orders, substance and imagery are cleared.
        """
        self._trails = {}
        for node in self._nodes.values():
            node.gate = 1.0
            if isinstance(node, PORTALS):
                node.activation = 1.0
            if isinstance(node, GroupNode):
                node.instances = {}
        self.place_bodies()
        for edge in self._dashed:
            edge.target.gate = unknown_value()
            if not isinstance(edge.target, PORTALS):
                edge.target.activation = unknown_value()
        for node in self._nodes.values():
            if isinstance(node, BodyNode):
                node.orders = []
        for tok in self._group_tokens():
            tok.orders = []
        self._tick, self._spaces = 0, {}
        self._sub = {}                              # each tree's substance grid
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
        """One tick of the gameboard (``_step_gameboard``). Gates carry over.

        If ``field`` is given, it is shown on the matching InputNode for this
        step only. Call ``reset_gates()`` to replay from t=0.
        """
        restore = None
        if field is not None:
            node = self._find_input_node(modality, at_node_id=at_node_id)
            restore = (node, node.field)
            node.field = field
        result = self._step_gameboard()
        if restore is not None and detach_after:
            restore[0].field = restore[1]
        return result

    # -- gameboard dynamics: a board-game grid world ------------------------

    def assign_roles(self) -> None:
        """Mark every node as observing (perceives, not drawn) or imagined (drawn).

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
        tops = {c.node_id for g in self.roots for c in content_children(g)}   # a whole tree with nothing to
        self._measures = [r for r in roots if r.node_id in tops                # perceive measures itself
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
        """Everything node views depend on besides structure: body placement and template gates."""
        def build():
            bodies = [n for n in self._nodes.values() if isinstance(n, BodyNode) and not isinstance(n, PlanNode)]
            templates = [t for n in self._nodes.values() if isinstance(n, BodyNode)
                         for t in content_children(n)[:-1]]
            turning = [n for t in templates for n in walk(t)     # what ``_facing`` reads: same gates, same facing
                       if (n.is_dynamic if isinstance(n, ObjectNode) else bool(getattr(n, "children", ())))]
            return bodies, templates, turning
        bodies, templates, turning = self._meta("view_inputs", build)
        tokens = tuple((t.node_id, None if t.position is None else tuple(t.position)) for t in self._group_tokens())
        return (tuple((None if b.position is None else tuple(b.position), None if b.extent is None else tuple(b.extent))
                      for b in bodies),
                tuple(t.gate for t in templates), tuple(n.gate for n in turning), tokens)

    def _group_tokens(self) -> List["GroupToken"]:
        """Every instance of every field and body."""
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
                for key in _spawned(node, ikey, node.instances):
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
                if place and node.node_id not in placed:       # the body fills its region
                    r0, r1, c0, c1 = rect
                    node.position = [(r0 + r1) / (2.0 * fh), (c0 + c1) / (2.0 * fw)]
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
            slots = [(i, s, v) for i, (s, v) in enumerate(orient_slots(template, rect)) if not is_empty(s)]
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
        bodies' placement, its substance, trails or input fields. Otherwise the last drawing is reused.
        """
        cache = self.__dict__.setdefault("_draw_cache", {})
        yang, yin, left, spaces = np.ones(self.grid), np.ones(self.grid), np.ones(self.grid), {}
        self._space_claims = {}                  # per tree: how strongly each cell of its space is claimed
        for r in self.roots:
            key = self._draw_key(r.node_id)
            hit = cache.get(r.node_id)
            if hit is None or hit[0] != key:
                a, b, v, c = self._paint(self.grid, r.node_id, place_bodies=True, with_claimed=True)
                hit = cache[r.node_id] = (key, a, b, v, _compose(a, _covered(a, b, v)),
                                          np.where(c > 0, c, 1.0 - v))      # unclaimed: the emptiness left behind
            _, a, b, v, spaces[r.node_id], self._space_claims[r.node_id] = hit
            yang, yin, left = yang * a, yin * b, left * v
        return spaces, _compose(yang, _covered(yang, yin, left))

    def _draw_key(self, root_id: str) -> tuple:
        """Everything a tree's drawing depends on (besides structure, which is part of the key)."""
        def build():
            per = {}
            for r in self.roots:
                ns = list(walk(r))
                per[r.node_id] = (ns, [n for n in ns if isinstance(n, BodyNode) and not isinstance(n, PlanNode)],
                                  [n for n in ns if isinstance(n, GroupNode)],
                                  [n for n in ns if isinstance(n, InWindowNode)])
            return per
        nodes, bodies, groups, inputs = self._meta("draw_nodes", build)[root_id]
        return self._placement_key(nodes, bodies, groups, inputs) + (
            self._substance_key(root_id), tuple((k, t.tobytes()) for k, t in self._trails.items() if k[2] == root_id))

    def _only_key(self, root_id: str, ids: frozenset) -> tuple:
        """The part of ``_draw_key`` the claims of only the nodes ``ids`` depend on: the gates that weigh them (theirs,
        the structures above them and those structures' templates), the placement of the units above or among them,
        and the input fields. Trails and substance are not claims of theirs."""
        return self._placement_key(*self._only_nodes(root_id, ids)[:4])

    def _instance_nodes(self, root_id: str, ids: frozenset) -> Dict[str, list]:
        """In a drawing of only ``ids``: for each field or body none of whose drawn content is another unit or an input,
        the nodes whose gates an instance's claims depend on (units it does not draw are left out)."""
        def build():
            drawn, out = self._only_nodes(root_id, ids)[4], {}
            unit = lambda m: isinstance(m, InWindowNode) or (isinstance(m, BodyNode) and not isinstance(m, PlanNode))

            def inner(n, top):
                ns, simple = [n], True
                for c in n.children:
                    if unit(c) and c.node_id not in drawn:
                        continue                 # not drawn here, nor anything in it
                    sub, ok = inner(c, False)
                    ns += sub
                    simple = simple and ok
                return ns, simple and (top or not (unit(n) and n.node_id in drawn))
            for n in self._nodes.values():
                if isinstance(n, GroupNode) and n.node_id in drawn:
                    ns, simple = inner(n, True)
                    if simple:
                        out[n.node_id] = ns
            return out
        return self._meta(("stamp", root_id, ids), build)

    def _only_nodes(self, root_id: str, ids: frozenset) -> tuple:
        """For a drawing of only ``ids``: the nodes whose gates weigh it, its bodies, groups and input windows, and the
        ids of the nodes on a path to its leaves (nothing elsewhere in the tree draws in it)."""
        def build():
            def visit(n):                        # the nodes on a path to one of ``ids``
                below = [visit(c) for c in n.children]
                if n.node_id in ids or isinstance(n, InWindowNode) or any(below):
                    held.append(n)
                    return True
                return False
            held = []
            visit(self._nodes[root_id])
            return (held + [c for n in held for c in n.children],
                    [n for n in held if isinstance(n, BodyNode) and not isinstance(n, PlanNode)],
                    [n for n in held if isinstance(n, GroupNode)], [n for n in held if isinstance(n, InWindowNode)],
                    frozenset(n.node_id for n in held))
        return self._meta(("only", root_id, ids), build)

    def _placement_key(self, gated, bodies, groups, inputs) -> tuple:
        """A drawing's cache key: the structure, the gates of ``gated``, where the bodies and the groups' instances
        are, and the input fields."""
        toks = [t for n in groups for t in n.instances.values()]
        placed = lambda us: tuple((None if u.position is None else tuple(u.position),
                                   None if u.extent is None else tuple(u.extent)) for u in us)
        return (self._meta("structure", lambda: object()), tuple(self.grid), tuple(n.gate for n in gated),
                placed(bodies), tuple(t.node_id for t in toks), placed(toks), tuple(id(n.field) for n in inputs))

    def tree_index(self) -> Dict[str, str]:
        """Which root (tree) each node belongs to."""
        def build():
            out: Dict[str, str] = {}
            for r in self.roots:
                for n in walk(r):
                    out.setdefault(n.node_id, r.node_id)
            return out
        return self._meta("tree_index", build)

    def _perceive(self, views=None) -> None:
        """Each top-most observer evaluates its subtree against last tick's imagery, through its views.

        A view is a sub-field of the whole board, carried into any turned frames above the
        observer (``node_views``), so bodies inside an observing pattern carve their windows in
        absolute grid coordinates. An observer with several views (it sits in a template's
        content) takes their weighted mean. Observers with an input window keep scanning it.
        """
        flow = getattr(self, "_flow", {})
        if self._imagery is None:                   # the first tick: the board as drawn, nothing moved yet
            self._spaces, self._imagery = self._render_all()
        spaces = getattr(self, "_spaces", None) or {}
        tree = self.tree_index()
        views = self.node_views(self.grid) if views is None else views
        parents = self._parents()
        for node in getattr(self, "_observer_roots", []):
            if isinstance(node, ObjectNode) and self._has_input_context(node, parents):
                continue
            own = spaces.get(tree.get(node.node_id))      # an observer sees only its own tree's space
            board = ReceptiveField(own if own is not None else blank_board(self.grid), flow.get(tree.get(node.node_id)))
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

    def _moving_bodies(self, views, cells: bool = True, rects: bool = False) -> Dict[str, tuple]:
        """Each placed body's home (its first view's chain and frame size) and its board cells (with ``rects``, its
        rect in its home frame instead)."""
        out = {}
        for b in self._body_nodes():
            vs = views.get(b.node_id)
            if isinstance(b, BodyNode) and not isinstance(b, PlanNode) and b.position is not None \
                    and b.extent is not None and vs:
                mask = vs[0][2] if rects else np.logical_or.reduce([_view_mask(self.grid, chain, rect) for _, chain, rect in vs]) \
                    if cells else None
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
                                            rect if rects else _view_mask(self.grid, chain, rect) if cells else None)
        return out

    def _issue_orders(self, views=None) -> None:
        """Every switched-on imagined motion leaf gives its missions this tick (``_issue_kinds``), in its own tree,
        through each of its views (``node_views``): its space is its own region on the board. Leaves in templates
        give orientation, not orders."""
        parents = self._parents()
        views = self.node_views(self.grid) if views is None else views
        bodies_at = self._moving_bodies(views)
        self._units = self._tokens()
        tree, by_tree = self.tree_index(), defaultdict(dict)    # a motion leaf moves only its own tree's bodies
        for b, v in bodies_at.items():
            by_tree[tree.get(getattr(self._units[b], "node", self._units[b]).node_id)][b] = v

        def build_templates():                   # dynamic leaves in templates give orientation, not orders
            out = set()
            for b in self._nodes.values():
                if isinstance(b, BodyNode) and not isinstance(b, GroupNode):
                    for t in content_children(b)[:-1]:
                        out.update(n.node_id for n in walk(t))
            return out
        in_templates = self._meta("in_templates", build_templates)
        self._motion_field(views, in_templates)
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
            acting = self._resolve_motion(node)
            if acting is None:                            # unknown motion, still in this realization
                for key in [k for k in self._inside if k.split("#")[0] == node.node_id]:
                    del self._inside[key]
                continue
            self._issue_kinds(acting, a, list(zip(keys, views.get(node.node_id, []))), by_tree[tree.get(node.node_id)],
                              parents)

    # -- motion kinds ------------------------------------------------

    def _issue_kinds(self, node: "ObjectNode", a: float, keyed_views, bodies_at, parents) -> None:
        """One switched-on motion leaf's motion this tick, by its trigram's kind (``_MOTION_KIND``)."""
        kind = _MOTION_KIND.get(PATTERN_TO_NAME.get(node.pattern))
        if kind is None:
            return
        root = node
        while parents.get(root.node_id):
            root = parents[root.node_id][0]
        meta = self.view_meta(node.node_id)
        units = getattr(self, "_units", None) or self._tokens()
        if keyed_views:
            self._issue_substance(node, kind, a, keyed_views, bodies_at, root.node_id, meta, units)

    def _unit_figures(self, root: str) -> Dict[str, np.ndarray]:
        """Every unit of a tree on the board: where it more likely than not is (``_unit_crops``)."""
        return {b: fig for b, (_, _, fig, _) in self._unit_crops(root).items()}

    def _unit_crops(self, root: str, whole: bool = False) -> Dict[str, tuple]:
        """Every unit of a tree (bodies, not fields) as (row, column, fig, cover): the board cell its arrays start at and,
        over its cells, where it more likely than not is (else where it most likely is) and how likely it is at each
        cell. With ``whole``, a unit past the edge whole, row and column off the board: it is solid there too. Units can
        be far past the edge, so nothing here spans the space between them."""
        def figure(cells, v, c):
            solid = (_claimed_value(v, c) > 0.5) & cells
            likely = solid & (c > 0.5)
            fig = likely if likely.any() or not solid.any() else solid & (c >= c[solid].max() - 1e-9)
            return fig, np.where(solid, c, 0.0)
        views = self.node_views(self.grid)                   # (views first: they may place new copies)
        units = self._tokens()
        node_of = lambda b: getattr(units[b], "node", units[b])
        rects = self._moving_bodies(views, rects=True) if whole else {}
        out, bases = {}, {}
        for b, (chain, _, cells) in self._moving_bodies(views).items():
            if b not in units or not (whole or cells.any()):      # (a copy spawned this tick is not a unit yet)
                continue
            u = node_of(b)
            if self.tree_index().get(u.node_id) != root or isinstance(u, FieldNode):
                continue
            if whole and not chain:
                (r, c0), (v, c) = self._whole_cells(root, units[b], rects[b][2], bases)
                cells = np.ones(v.shape, dtype=bool)
            else:
                (r, c0), (v, c) = (0, 0), self._paint_only_values(root, self._own_content(u))
            if cells.any():
                out[b] = (r, c0) + figure(cells, v, c)
        return out

    def _whole_cells(self, root: str, tok, rect: Region, bases: dict = None):
        """A unit's own claims over its cells, the part past the edge included, and the board cell they start at: drawn
        with the unit moved onto the board (as far as it fits), then moved back. ``bases``: drawings with every unit
        where it is, to reuse: moving one unit changes that unit's claims only, and they are drawn again in place."""
        H, W = self.grid
        u, (r0, r1, c0, c1) = getattr(tok, "node", tok), rect
        dr, dc = (-r0 if r0 < 0 else min(0, H - r1)), (-c0 if c0 < 0 else min(0, W - c1))
        q0, q1, p0, p1 = _clip((r0 + dr, r1 + dr, c0 + dc, c1 + dc), (H, W))
        ids, shifted = self._own_content(u), None
        if bases is not None and SUBCELL == 1 and (dr or dc):
            if ids not in bases:
                record = {}
                claims = self._paint_claims(root, ids, record)
                bases[ids] = claims, record, self._paint_redraw
            claims, record, redraw = bases[ids]
            drawings = record.get(tok.node_id)

            def shifted():
                if drawings is None:
                    return claims                # not drawn (a template, say): where it is changes nothing
                if None in drawings:
                    return None                  # drawn in a turned frame: draw it all
                out, at = [], 0
                for d in drawings:
                    out += claims[at:d[0]]
                    out += redraw(d)
                    at = d[1]
                return out + claims[at:]
        was = list(tok.position)
        tok.position[0], tok.position[1] = was[0] + dr / H, was[1] + dc / W
        try:
            values = self._paint_only_values(root, ids, (q0, q1, p0, p1), shifted)
        finally:
            tok.position[:] = was
        return (q0 - dr, p0 - dc), values

    def _drive_past_edge(self, node, tok, rect: Region, a: float, root_id: str, bodies_at) -> None:
        """A body's own leaf whose body is wholly past the edge (in the leaf's region) still drives the body
        where the body is (its figure there), at the leaf's own rate: nothing on the board reaches it there to dilute or
        pin it. Its mission: until its window is past the region."""
        holder = getattr(tok, "node", tok)
        if tok is None or isinstance(holder, FieldNode) or tok.node_id not in bodies_at \
                or any(o[5] == node.node_id for o in tok.orders):
            return
        home = self._moving_bodies(self.node_views(self.grid), rects=True).get(tok.node_id)
        if home is None or home[0]:                      # not placed in the board's own frame
            return
        start, values = self._whole_cells(root_id, tok, home[2])
        rows, cols = (start[k] + np.arange(values[0].shape[k]) for k in (0, 1))
        r0, r1, c0, c1 = rect
        there = ((rows >= r0) & (rows < r1))[:, None] & ((cols >= c0) & (cols < c1))[None, :]
        fig = there & (values[0] > 0) & (_claimed_value(*values) > 0.5)
        if not fig.any():
            return
        motion = BodyNode._template_motion(node)
        norm = float(np.hypot(*motion))
        if norm < 1e-12:
            return
        for axis, comp in enumerate((motion[0] / norm, motion[1] / norm)):
            if abs(comp) > 1e-9:
                at, lo, hi = (rows, cols)[axis], rect[2 * axis], rect[2 * axis + 1] - 1
                beyond = int(hi - at.min() + 1) if comp > 0 else int(at.max() - lo + 1)
                self._push_body(tok, bodies_at, axis, a * comp, node_id=node.node_id, beyond=beyond)

    def _home_clear(self, node, first) -> bool:
        """SOLID_FIGURE: may a field spawn a copy of ``node`` at ``first``'s home? Only where nothing
        solid is: no unit (copies of the body included) and no substance more likely there than not (unknown is clear;
        the fields' own claims are not in the way of what they hold)."""
        H, W = self.grid
        root = self.tree_index()[node.node_id]
        was, first.position = first.position, list(first.home)
        r0, r1, c0, c1 = _clip(first.window(H, W), (H, W))
        first.position = was
        home = np.zeros((H, W), dtype=bool)
        home[r0:r1, c0:c1] = True
        S = self._substance(root)
        if (home & (S["s"] > 0) & (S["v"] > 0.5) & ~S["F"]).any():
            return False
        return not any((home & fig).any() for fig in self._unit_figures(root).values())

    def _own_figure(self, root_id: str, node) -> np.ndarray:
        """Where a body's own content is figure."""
        v, c = self._paint_only_values(root_id, self._own_content(node))
        return (c > 0) & (_claimed_value(v, c) > 0.5)

    def _own_content(self, node) -> frozenset:
        """A body's own content: its nodes and its fields', not its nested bodies'."""
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
        """``_issue_kinds`` on the substance grid: a standing rule. Everything whose substance lies in a view's space
        and has no order from this leaf under way gets one (a body's leaf acts where the body's figure is), as strongly
        as the composed motion claims drive it there; a mission runs until its content is past the region. "In" moves
        each row (or column) only as far as the centre, so the halves pile up there."""
        S = self._substance(root_id)
        H, W = self.grid
        R, C = _grid_index((H, W))
        none = np.zeros((H, W), dtype=bool)
        src = S.get("src", {})
        drive = S.get("drive")                                 # the composed motion claims' rates
        loose = (S["s"] > 0) & (S["v"] > 0.5)
        motion = BodyNode._template_motion(node)
        axis_vec = (1.0, 0.0) if node.orientation == Orientation.VERTICAL else (0.0, 1.0)
        some = (isinstance(node, _AsMotion)                  # universal*: "some motion here" (along its
                and _UNKNOWN_MOTION.get(node._node.pattern) == "universal")   # axis), which moving content satisfies
        for key, (_, chain, rect) in keyed_views:
            k = int(key.split("#")[1])
            own = meta[k][1] if k < len(meta) else ()
            space = _view_mask(self.grid, chain, rect)
            rows, cols = np.nonzero(space)
            if not rows.size:
                if own and kind in ("bulk", "gradient") and not chain:
                    self._drive_past_edge(node, units.get(own[-1]), rect, a, root_id, bodies_at)
                continue
            mine = getattr(getattr(units.get(own[-1]), "node", None), "node_id", None) if own else None
            stuff = loose | src.get(None, none) | src.get(mine, none)    # a carried field: only its own motion
            n_orders = len(S["orders"])
            turned = _view_turn(chain, axis_vec)
            ax = 0 if abs(turned[0]) > abs(turned[1]) else 1         # the leaf's axis on the board
            holder = getattr(units.get(own[-1]), "node", units.get(own[-1])) if own else None
            if holder is not None and not isinstance(holder, FieldNode):
                space = space & self._own_figure(root_id, holder)   # a body's leaf acts where the body is
                stuff = stuff & space                                 # (its figure), not on its unknown frame
                if not space.any():                                 # its figure is past the edge
                    if kind in ("bulk", "gradient") and not chain:
                        self._drive_past_edge(node, units.get(own[-1]), rect, a, root_id, bodies_at)
                    continue
            bodies = [units[b] for b, (_, _, cells) in sorted(bodies_at.items())
                      if not isinstance(getattr(units[b], "node", None), FieldNode) and (space & cells).any()
                      and (self._own_figure(root_id, getattr(units[b], "node", units[b])) & cells & space).any()]
            self._inside[key] = ids = {b.node_id for b in bodies}
            groups = getattr(self, "_unit_groups", {})              # a unit carried by one this leaf drives
            fresh = [b for b in bodies if not any(o[5] == node.node_id for o in b.orders)    # moves with it,
                     and not ids & set(groups.get(b.node_id, ()))                              # not twice
                     and not (some and any(o[0] == ax for o in b.orders))]
            under_way = np.logical_or.reduce([o[0] for o in S["orders"]
                                              if (some and o[1] == ax) or o[7:] == [key]] + [none])
            fresh_m = stuff & space & ~under_way
            scale = {}
            if drive is not None:                # as strongly as the composed claims drive it there
                scale = {b.node_id: float(drive[space & bodies_at[b.node_id][2]].max(initial=0.0)) for b in fresh}
                fresh = [b for b in fresh if scale[b.node_id] > 1e-9]
                fresh_m = fresh_m & (drive > 1e-9)
            k = lambda b: scale.get(b.node_id, 1.0)
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
                centre = (lo + hi + 1) / 2.0
                coord = (R, C)[axis]
                sign = 1 if comp > 0 else -1
                rate = a * abs(comp)
                beyond = lambda m, s: (int(hi - coord[m].min() + 1) if s > 0      # as far as it takes the
                                       else int(coord[m].max() - lo + 1))            # content past the region
                past = lambda b, s: beyond(bodies_at[b.node_id][2], s)          # a body: until its window is past
                if kind == "bulk":
                    if fresh_m.any():
                        S["orders"].append([fresh_m.copy(), axis, sign, beyond(fresh_m, sign), rate, 0.0, 0.0])
                    for b in fresh:
                        self._push_body(b, bodies_at, axis, sign * rate * k(b), node_id=node.node_id, beyond=past(b, sign))
                elif kind == "gradient":                                # layers from the leading edge
                    levels = sorted(set(coord[fresh_m].tolist()), reverse=sign > 0)
                    for j, v in enumerate(levels):
                        layer = fresh_m & (coord == v)
                        S["orders"].append([layer, axis, sign, beyond(layer, sign), rate, 0.0, j / a])
                    lead = lambda b: -sign * float(np.nonzero(bodies_at[b.node_id][2])[axis].mean())
                    for j, b in enumerate(sorted(fresh, key=lead)):
                        self._push_body(b, bodies_at, axis, sign * rate * k(b), node_id=node.node_id, delay=j / a,
                                        beyond=past(b, sign))
                else:                                                   # out / in: the two halves
                    for half in (-1, 1):
                        part = fresh_m & (np.sign(coord + 0.5 - centre) == half)
                        go = half if kind == "out" else -half
                        if not part.any():
                            continue
                        if kind == "out":
                            S["orders"].append([part, axis, go, beyond(part, go), rate, 0.0, 0.0])
                            continue
                        for v in sorted(set(coord[part].tolist())):      # in: as far as the centre, no further
                            dist = int(round(abs(v + 0.5 - centre) - 0.5))
                            if dist > 0:
                                S["orders"].append([part & (coord == v), axis, go, dist, rate, 0.0, 0.0])
                    for b in fresh:
                        side = np.sign(float(np.nonzero(bodies_at[b.node_id][2])[axis].mean()) + 0.5 - centre)
                        if side:
                            go = side if kind == "out" else -side
                            self._push_body(b, bodies_at, axis, go * rate * k(b), node_id=node.node_id, half=True,
                                            beyond=past(b, go) if kind == "out" else 0)
            if drive is not None:                # the substance orders, split by how strongly each cell is driven
                level, new = np.round(drive, 3), S["orders"][n_orders:]
                S["orders"][n_orders:] = [[o[0] & (level == lv), *o[1:4], o[4] * lv, *o[5:]]
                                          for o in new for lv in np.unique(level[o[0]]) if lv > 0]
            for o in S["orders"][n_orders:]:
                o.append(key)

    def _motion_field(self, views, in_templates=frozenset()) -> None:
        """Each tree's motion claims this tick, composed as content claims are (strength: the leaf's
        gate; a body's leaf claims where the body is): void* none (0), neutral* unknown (0.5), any other motion leaf
        some (1). A cell's pin is how far it leans still (1 - 2v): what is in it resists a push of no more strength
        (``_held``), whatever came there. Its drive is how far it leans moving (2v - 1), as a share of what the moving
        claims alone would make it: each leaf there moves its content at its own rate times that share
        (``_issue_substance``), so void* and neutral* dilute motion and leaves of different strength keep theirs."""
        parents, units, tree = self._parents(), self._tokens(), self.tree_index()
        claims, fields = defaultdict(list), set()
        for node in self._nodes.values():
            name = PATTERN_TO_NAME.get(getattr(node, "pattern", None))
            if not (isinstance(node, ObjectNode) and node.is_dynamic and getattr(node, "role", None) == "imagined"
                    and node.node_id not in in_templates):
                continue
            a = self._acting(node, parents)
            if not a > ONSET_THRESHOLD:
                continue
            root, meta = tree[node.node_id], self.view_meta(node.node_id)
            for k, (_, chain, rect) in enumerate(views.get(node.node_id, [])):
                space = _view_mask(self.grid, chain, rect)
                own = meta[k][1] if k < len(meta) else ()
                holder = getattr(units.get(own[-1]), "node", units.get(own[-1])) if own else None
                if holder is not None and not isinstance(holder, FieldNode):
                    space = space & self._own_figure(root, holder)
                elif holder is not None:
                    fields.add(node.node_id)              # a field's claims outrank what they cover, as in content
                if space.any():
                    claims[root].append((space, {"void": 0.0, "neutral": 0.5}.get(name, 1.0), a, space,
                                         node.node_id, False))
        for S in getattr(self, "_sub", {}).values():
            S["pin"] = S["drive"] = None
        for rid, cs in claims.items():
            v, _ = _claim_values(cs, tuple(self.grid), fields)
            moving = [c for c in cs if c[1] == 1.0]          # what the moving claims alone would drive
            m = (2.0 * _claim_values(moving, tuple(self.grid), fields)[0] - 1.0 if moving
                 else np.zeros(tuple(self.grid)))
            S = self._substance(rid)
            S["pin"] = np.maximum(0.0, 1.0 - 2.0 * v)
            S["drive"] = np.where(m > 1e-9, np.maximum(0.0, 2.0 * v - 1.0) / np.maximum(m, 1e-9), 0.0)

    def _acting(self, node, parents) -> float:
        """How strongly a leaf acts: its gate, times the gates of the structures above it."""
        a, anc = node.gate, node
        while parents.get(anc.node_id):
            anc = parents[anc.node_id][0]
            a *= _sg(anc) if not isinstance(anc, ObjectNode) else 1.0
        return a

    def _push_body(self, body, bodies_at, axis: int, signed_rate: float, node_id: str = "",
                   delay: float = 0.0, half: bool = False, beyond: int = 0) -> None:
        """Order a body along a board axis, expressed in the body's own frame; with ``beyond`` (a body on the board's
        frame), a mission for that many cells."""
        home, frame, _ = bodies_at[body.node_id]
        vec = [0.0, 0.0]
        vec[axis] = signed_rate
        for ax, m in enumerate(_view_unturn(home, tuple(vec))):
            if abs(m) < 1e-9:
                continue
            cells = max(1, round(body.extent[ax] * frame[ax]))
            if beyond and not home:          # a destination, where its window starts once past
                sign = 1 if m > 0 else -1
                at = round(body.position[ax] * frame[ax] - cells / 2)
                a = self._acting(self._nodes[node_id], self._parents()) if node_id in self._nodes else 1.0
                body.orders.append([ax, sign, beyond, abs(m), 0.0, node_id, delay, at + sign * beyond, a, False])
                continue
            body.orders.append([ax, 1 if m > 0 else -1, max(1, round(cells / 2)) if half else cells, abs(m), 0.0,
                                node_id, delay])

    def _substance_key(self, root_id: str):
        S = getattr(self, "_sub", {}).get(root_id)
        return None if S is None else (S["v"].tobytes(), S["s"].tobytes(), S["moved"].tobytes())

    def motion_state(self) -> tuple:
        """Everything about motion and substance that can change a later frame (for periodicity checks).

        Not the tick: nothing reads it (universal* is a latent direction, fixed per realization, not drawn per tick).
        """
        groups = tuple((t.node_id, None if t.position is None else tuple(round(v, 9) for v in t.position),
                        tuple(tuple(o) for o in t.orders))
                       for t in self._group_tokens())
        subs = tuple((rid, self._substance_key(rid),
                      tuple((o[0].tobytes(), o[1], o[2], o[3], round(o[4], 6), round(o[5], 6), round(o[6], 6))
                            for o in S["orders"]), S["passed"].tobytes() if "passed" in S else b"")
                     for rid, S in sorted(getattr(self, "_sub", {}).items()))
        flow = tuple(sorted((k, v.tobytes()) for k, v in getattr(self, "_flow", {}).items()))
        return ((groups,) if groups else ()) + ((subs,) if subs else ()) + ((flow,) if flow else ())

    def _move_bodies_gameboard(self, views=None) -> None:
        """Each order advances at most one cell per tick once its credit reaches a whole cell; moves add up.

        A body moves on its own frame's cells (the board, or the turned frame it lives in).
        """
        homes = self._moving_bodies(self.node_views(self.grid) if views is None else views, cells=False)
        bodies = [b for b in self._tokens().values() if b.position is not None
                  and not isinstance(getattr(b, "node", None), FieldNode)]       # a field never moves itself
        self._pending, self._strength, self._stepped = {}, {}, {}   # this tick's own moves (and how strongly driven,
                                                                    # and the missions that stepped), all taken first
        parents = self._parents()
        for body in bodies:
            net = self._pending[body.node_id] = [0, 0]
            frame = homes[body.node_id][1] if body.node_id in homes else self.grid
            left = [o[2] for o in body.orders]                   # what each had left as this tick began
            now = [self._acting(self._nodes[o[5]], parents) if len(o) > 7 and o[5] in self._nodes else 1.0
                   for o in body.orders]                         # how strongly each mission's leaf acts now
            off = [len(o) > 7 and not a > ONSET_THRESHOLD for o, a in zip(body.orders, now)]   # a trip under way
            stepped = self._stepped[body.node_id] = []
            for i, order in enumerate(body.orders):
                if len(order) > 6 and order[6] >= 1 - 1e-9:     # queued: waits its turn
                    order[6] -= 1
                    continue
                if len(order) > 8:
                    if off[i] and not order[9]:                  # its leaf went off before it moved the body:
                        order[2] = 0                             # a push held back, never begun; it lapses
                        continue
                    if not off[i] and order[8] > 0:              # while its leaf is on, as strong as the leaf now
                        order[3], order[8] = order[3] * now[i] / order[8], now[i]
                if len(order) > 7:                               # a mission: against one under way, it waits;
                    if any(o[0] == order[0] and o[1] != order[1] and n > 0 and f
                           for o, n, f in zip(body.orders[:i], left, off)):
                        continue
                    size = max(1, round(body.extent[order[0]] * frame[order[0]]))     # how far it still is from
                    order[2] = order[1] * (order[7] - round(body.position[order[0]] * frame[order[0]] - size / 2))
                    if order[2] <= 0:                            # its destination, wherever the body now is
                        continue
                order[4] += order[3]
                k = min(int(order[4]), 1, order[2])
                order[4] -= k
                order[2] -= k
                net[order[0]] += order[1] * k
                if k:
                    self._strength[body.node_id] = max(self._strength.get(body.node_id, 0.0), min(1.0, order[3]))
                    stepped.append(order)
            body.orders = [o for o in body.orders if o[2] > 0]
        for body in bodies:
            H, W = homes[body.node_id][1] if body.node_id in homes else self.grid
            net = self._pending[body.node_id]
            if SOLID_FIGURE and any(net) and (H, W) == tuple(self.grid):
                net = self._chain_move(body, net)
            elif any(net) and (H, W) == tuple(self.grid) and self._pinned(body):
                net = [0, 0]                             # a pin holds a body without built-in solidity too
            for o in self._stepped.get(body.node_id, ()):
                if len(o) > 9 and net[o[0]] * o[1] > 0:
                    o[9] = True                                  # it has moved the body: committed
            for axis, n in ((0, H), (1, W)):                     # move the window's start cell
                size = max(1, round(body.extent[axis] * n))
                body.position[axis] = (round(body.position[axis] * n - size / 2) + net[axis] + size / 2) / n
            body.velocity = [net[0] / H, net[1] / W]

    def _pinned(self, body) -> bool:
        """Is the body's figure (on the board) where a pin holds it against its own drive?"""
        root = self.tree_index()[getattr(body, "node", body).node_id]
        fig = self._unit_figures(root).get(body.node_id)
        held = _held(self._substance(root), getattr(self, "_strength", {}).get(body.node_id, 1.0))
        return fig is not None and bool((fig & held).any())

    def _chain_move(self, body, net: list) -> list:
        """SOLID_FIGURE: the move ``net`` (or its part along one axis) if everything it pushes can
        move with it, else none. Something blocks where it more likely than not is: substance and fields where they
        claim figure with probability above one half, a unit where it more likely than not covers the cell (its own
        content's coverage above one half: for an agent of uncertain extent, all but its corners; a body too small for any
        cell to pass one half, where it most likely is). A body pushes what
        it would move into: a unit, with everything inside it (a body pushing the unit it is inside is carried along),
        and through it whatever that unit would move into; stuff as a run (``_stuff_run``), a field's figure included (a
        copy is pushed out; the field re-asserts behind once the cell is clear). Nothing may leave the board: what holds
        is what would be pushed off it. Copies of one body are solid to each other, as any two bodies are.

        A move resolves what made it possible: the uncertain stuff
        it passes through (unknown, not substance more likely than not) is resolved by Bayes when it leaves a cell:
        covering the cell meant it held no substance, so its chance of substance becomes p x (1 - c), c the most likely
        the unit covered it while there (fields re-assert their claims, so a fog closes again behind)."""
        H, W = self.grid
        units, groups = self._tokens(), getattr(self, "_unit_groups", {})
        node_of = lambda b: getattr(units[b], "node", units[b])
        root = self.tree_index()[node_of(body.node_id).node_id]
        crops = self._unit_crops(root, whole=True)                            # every unit, past the edge too
        units = self._tokens()                                                # (with any copies the views placed)
        inside = lambda b, a: a in groups.get(b, ())                          # unit b is inside unit a
        known = {}

        def cells(b):
            """Where unit b more likely than not is, as (row, column) cells: units may be anywhere past the board."""
            if b not in known:
                r, c, fig, _ = crops[b]
                rows, cols = np.nonzero(fig)
                known[b] = set(zip((rows + r).tolist(), (cols + c).tolist()))
            return known[b]

        def on_board(where):
            found = [(r, c) for r, c in where if 0 <= r < H and 0 <= c < W]
            return tuple(np.array(found).T) if found else None

        def times(a, r, c, cover):
            """``a *= 1 - cover``, with ``cover`` starting at board cell (r, c)."""
            q0, q1, p0, p1 = _clip((r, r + cover.shape[0], c, c + cover.shape[1]), (H, W))
            a[q0:q1, p0:p1] *= 1.0 - cover[q0 - r:q1 - r, p0 - c:p1 - c]

        group = lambda a: set().union(*[cells(b) for b in crops if b == a or inside(b, a)])
        S = self._substance(root)
        stuff = _solid_stuff(S).copy()                                        # a unit's own: pushed whole
        for r, c, fig, _ in crops.values():
            q0, q1, p0, p1 = _clip((r, r + fig.shape[0], c, c + fig.shape[1]), (H, W))
            stuff[q0:q1, p0:p1] &= ~fig[q0 - r:q1 - r, p0 - c:p1 - c]
        drive = getattr(self, "_strength", {}).get(body.node_id, 1.0)
        held = _held(S, drive)                                                # pinned too strongly to be pushed

        def plan(d):
            moving, push, queue = [], np.zeros((H, W), dtype=bool), [body.node_id]
            while queue:
                x = queue.pop()
                if x in moving or any(inside(x, m) for m in moving):
                    continue                                                  # carried by a moving unit
                g = group(x)
                at = on_board(g)
                if at is not None and held[at].any():
                    return None                                               # pinned
                moving.append(x)
                ahead = {(r + d[0], c + d[1]) for r, c in g} - g              # what it would move into
                at = on_board(ahead)
                if at is not None:
                    push[at] |= stuff[at]
                queue += [y for y in crops if y not in moving and not inside(y, x) and not ahead.isdisjoint(cells(y))]
            if push.any() and _stuff_run(S, push, *d, drive=drive)[1].any():
                return None                                                   # stuff in the way cannot move
            return moving, push

        for d in ((net[0], net[1]), (net[0], 0), (0, net[1])):
            if d == (0, 0):
                continue
            found = plan(d)
            if found is None:
                continue
            moving, push = found
            movers = [x for x in moving if not any(inside(x, m) for m in moving if m != x)]
            for x in movers:
                if x == body.node_id:
                    continue                                                  # the caller moves the body itself
                tok = units[x]
                own_move = getattr(self, "_pending", {}).get(tok.node_id)
                for axis in (0, 1):            # a pushed unit already moving this way this tick: the push is its move
                    if own_move and own_move[axis] * d[axis] > 0:
                        own_move[axis] -= d[axis] if abs(own_move[axis]) >= abs(d[axis]) else own_move[axis]
                for axis, n in ((0, H), (1, W)):
                    size = max(1, round(tok.extent[axis] * n))
                    tok.position[axis] = (round(tok.position[axis] * n - size / 2) + d[axis] + size / 2) / n
            if push.any():
                self._shift_substance(S, push, *d, drive=drive)
            was, now = np.ones((H, W)), np.ones((H, W))       # the chance no unit covers a cell, before / after
            for b, (r, c, _, cover) in crops.items():
                moved = any(b == x or inside(b, x) for x in movers)
                times(was, r, c, cover)
                times(now, r + d[0] if moved else r, c + d[1] if moved else c, cover)
            passed = S.setdefault("passed", np.zeros((H, W)))   # the most likely it covered a cell it is still on
            np.maximum(passed, 1.0 - np.minimum(was, now), out=passed)
            left = (now >= 1.0) & (passed > 0)
            seen = left & (S["s"] > 0) & ~S["F"] & ~(S["v"] > 0.5)             # uncertain stuff it went through:
            p = 0.5 + (S["v"] - 0.5) * S["s"]                                 # substance only where it did not go
            S["v"][seen], S["s"][seen], S["moved"][seen] = (p * (1.0 - passed))[seen], 1.0, False
            passed[left] = 0.0
            return [0, 0] if any(inside(body.node_id, m) for m in moving) else list(d)   # carried: no move of its own
        return [0, 0]

    @staticmethod
    def _ancestors(node, parents) -> list:
        out = []
        while parents.get(node.node_id):
            node = parents[node.node_id][0]
            out.append(node)
        return out

    def unknown_motion_units(self) -> Dict[str, Tuple[str, List[str], "Orientation"]]:
        """Unit key -> (kind, leaf ids, orientation) for every imagined universal* leaf (neutral* is a claim that
        motion is unknown, ``_motion_field``, not a latent direction).

        A unit is the leaf's innermost field or body node (its template copies share it), else the leaf. Keys look
        like ``"n4:universal"``.
        """
        views = self.node_views(self.grid)
        out: Dict[str, Tuple[str, List[str], Orientation]] = {}
        for node in self._nodes.values():
            kind = _UNKNOWN_MOTION.get(node.pattern) if isinstance(node, ObjectNode) else None
            if kind != "universal" or not node.is_dynamic or getattr(node, "role", None) != "imagined":
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
        if kind == "neutral":
            return None                          # it orders nothing; it only dilutes (_motion_field)
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
        self._gate_motion_now()
        self._issue_orders(views)
        self._propagate_delayed()
        before = self._flow_begin()
        self._move_bodies_gameboard(views)
        for rid in list(getattr(self, "_sub", {})):
            self._move_substance(rid)
            self._reset_fields(rid)
        self._spawn_bodies()
        self._flow_end(before)
        self._spaces, self._imagery = self._render_all()
        self._tick += 1
        return result

    def _gate_motion_now(self) -> None:
        """A dashed edge into a motion leaf sets its gate in the tick its sources perceive (the mean of their
        activations), so the motion it switches on is ordered before the next move."""
        sources: Dict[str, List[Node]] = defaultdict(list)
        for edge in self._dashed:
            if isinstance(edge.target, ObjectNode) and edge.target.is_dynamic:
                sources[edge.target.node_id].append(edge.source)
        for nid, srcs in sources.items():
            self._nodes[nid].gate = _clamp(_mean([s.activation for s in srcs]))

    def _unit_cells(self) -> Dict[str, np.ndarray]:
        return {b: cells for b, (_, _, cells) in self._moving_bodies(self.node_views(self.grid)).items()}

    def _flow_begin(self):
        """Before a tick's moves, where every unit is (and a clean slate for the substance's flow)."""
        for S in getattr(self, "_sub", {}).values():
            S["flow"] = np.zeros(tuple(self.grid) + (2,))
        return self._unit_cells()

    def _flow_end(self, before) -> None:
        """This tick's optical flow per tree, (H, W, 2) cells moved per cell: a moved unit's cells get its shift,
        moved substance its own; observers see it next tick (``_perceive``)."""
        H, W = self.grid
        now = self._unit_cells()                       # (views first: they may place new copies)
        tree, units, flow = self.tree_index(), self._tokens(), {}
        for b, cells in now.items():
            old = before.get(b)
            if old is None or old is cells or not old.any() or not cells.any():
                continue                               # (the same view mask: it did not move)
            d = np.argwhere(cells).min(axis=0) - np.argwhere(old).min(axis=0)
            if d.any():
                flow.setdefault(tree.get(getattr(units[b], "node", units[b]).node_id), np.zeros((H, W, 2)))[cells] = d
        for rid, S in getattr(self, "_sub", {}).items():
            if S.get("flow") is not None and S["flow"].any():
                flow[rid] = flow.get(rid, np.zeros((H, W, 2))) + S["flow"]
        self._flow = flow

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
        overlapping paint combines by noisy-OR.
        The default root is the last root (the imagined one by convention).

        With ``place_bodies=True``, a moving BodyNode is drawn in its own
        window (``BodyNode.window``), which is the region ``evaluate()`` carves
        for it, instead of its parent's whole region. This makes body motion
        visible in the output. The default keeps the original behaviour.
        """
        a, b, v = self._paint(shape, output_root_id, place_bodies)
        return ReceptiveField(_compose(a, _covered(a, b, v)))

    def _paint(self, shape: Tuple[int, int], output_root_id: Optional[str], place_bodies: bool, values: bool = False,
               raw: bool = False, with_claimed: bool = False):
        """The noisy-OR complements ``prod(1 - c)`` of one root's yang and yin paint, from its composed claims, and of
        the emptiness its moved bodies left behind (applied by ``_covered`` only where nothing else claims a cell).
        With ``values``, the claims' values and strengths instead (``_claim_values``); with ``raw``, the claims and the
        emptiness left behind, uncomposed. With SUBCELL > 1 the board is drawn that much finer and pooled back. ``self._paint_only``: only
        those leaves are drawn (and no substance); ``self._seeding``: only the bare leaves, into the substance grid."""
        H, W = shape
        out_root = self._nodes[output_root_id] if output_root_id is not None else self.roots[-1]
        # cv: [yang paint, yin paint, place bodies, frame size, inside a body, emptiness left behind, claims,
        # offset from a moved body's home]. An oriented slot swaps in a canvas for its upright frame (render_oriented).
        # The yang and yin paint only measure what a moved body still covers (its trail); the drawing is the claims'.
        blank = _blank if values else (lambda h, w: np.ones((h, w)))     # (values: claims only, nothing paints these)
        cv = [blank(H, W), blank(H, W), place_bodies, (H, W), False, blank(H, W), [], (0, 0)]
        only = getattr(self, "_paint_only", None)
        keep = None if only is None else self._only_nodes(out_root.node_id, only)[4]
        spawn_index = {}                         # per group: ``_spawned``'s answers (a drawing adds no instances)
        stamps = self.__dict__.setdefault("_stamps", {}) if only is not None and values else None
        stampable = None if stamps is None else self._instance_nodes(out_root.node_id, only)
        record = self.__dict__.pop("_paint_record", None)     # per unit: where its claims are, how to redraw them
                                                 # (None for a drawing in a turned frame)
        turned = [0]                             # inside a turned frame (those claims are carried into their slot)
        seeding = getattr(self, "_seeding", False)
        fine = SUBCELL > 1 and (H, W) == tuple(self.grid)      # drawn SUBCELL times finer, then pooled back
        thick = FINE_MOTION if FINE_MOTION > 1 else (
            SUBCELL if (H, W) == (self.grid[0] * SUBCELL, self.grid[1] * SUBCELL) and SUBCELL > 1 else 1)

        def at_home(cells):
            """Cells moved back to where the enclosing body started."""
            dr, dc = cv[7]
            return cells if dr == 0 and dc == 0 else _moved_home(cells, dr, dc)

        def claim(mask, value, w, leaf, home=None, moved=False) -> None:
            """Record a claim; its home cells (``home``, default its cells) moved back to where the enclosing body
            started."""
            cv[6].append((mask, value, w, at_home(mask if home is None else home), leaf, moved or cv[7] != (0, 0)))

        def render_plan(children: List[Node], region: Region, weight: float, ikey: tuple = ()) -> None:
            if not children or weight < 1e-9 or is_empty(region):
                return
            if len(children) == 1:
                render_node(children[0], region, weight, ikey)
                return
            template, rest = children[0], children[1:]
            if keep is not None and not any(c.node_id in keep for c in rest):
                return
            p = _clamp(template.gate)
            if p > 1e-9:
                for i, (sub, v) in enumerate(orient_slots(template, region)):
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
            turned[0] += 1
            cv[:] = [blank(Hc, Wc), blank(Hc, Wc), saved[2], (Hc, Wc), saved[4],   # bodies live in the frame
                     blank(Hc, Wc), [], (0, 0)]
            render_plan(children, (0, Hc, 0, Wc), weight, ikey)
            frame_yang, frame_yin, frame_left, frame_claims = cv[0], cv[1], cv[5], cv[6]
            cv[:] = saved
            turned[0] -= 1
            r0, r1, c0, c1 = _clip(slot, cv[0].shape)                    # the part of the slot on the canvas
            rows, cols = (a[r0 - slot[0]:r1 - slot[0], c0 - slot[2]:c1 - slot[2]] for a in (rows, cols))
            if not values:
                cv[0][r0:r1, c0:c1] *= frame_yang[rows, cols]
                cv[1][r0:r1, c0:c1] *= frame_yin[rows, cols]
                cv[5][r0:r1, c0:c1] *= frame_left[rows, cols]
            for mask, value, w, home, leaf, moved in frame_claims:          # into the slot, home cells too
                m, h = np.zeros(cv[0].shape, dtype=bool), np.zeros(cv[0].shape, dtype=bool)
                m[r0:r1, c0:c1], h[r0:r1, c0:c1] = mask[rows, cols], home[rows, cols]
                if not m.any():
                    continue
                if not np.isscalar(value):
                    full = np.full(cv[0].shape, 0.5)
                    full[r0:r1, c0:c1] = value[rows, cols]
                    value = full
                claim(m, value, w, leaf, h, moved)

        def draw_moved(draw, home: Region, region: Region, key: tuple) -> None:
            """``draw(region)`` paints a body's content where it is; its claims keep the priority of where it started.
            Once a body has moved, every cell it has claimed so far (at home or anywhere it has been since) but no
            longer claims reads as known empty, as strongly as it was claimed: a moved body leaves a trail of
            emptiness behind, not "no claim", unless other content covers the cell (``_covered``). The trail lasts
            until ``reset_gates()``."""
            key = (H, W, out_root.node_id, cv[0].shape) + key
            trail = self._trails.get(key)
            plain, dr0, dc0 = draw, home[0] - region[0], home[2] - region[2]

            def draw(r):
                old = cv[7]
                cv[7] = (old[0] + dr0, old[1] + dc0)
                plain(r)
                cv[7] = old
            if values or (region == home and trail is None):
                draw(region)                     # (values: claims only, which trails do not touch)
                return

            def alone(r):                        # the content's paint on its own, at r
                saved = cv[:]
                cv[0], cv[1], cv[5], cv[6] = np.ones_like(saved[0]), np.ones_like(saved[1]), np.ones_like(saved[5]), []
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
            cv[6].extend(nc)
            vacated = (trail < 1) & (ny == 1) & (nn == 1)
            cv[5][vacated] *= trail[vacated]

        def draw_instance(node, region: Region, weight: float, ikey: tuple) -> None:
            """One instance of a field or body, drawn where it is (its region: where the tree places it)."""
            tok = node.instances.get(ikey)
            if record is None or tok is None:
                place(node, tok, region, weight, ikey)
                return
            drawing = None if turned[0] else [len(cv[6]), None, node, region, weight, ikey, cv[4], cv[7]]
            record.setdefault(tok.node_id, []).append(drawing)     # (a nested copy may be drawn from two parents)
            place(node, tok, region, weight, ikey)
            if drawing is not None:
                drawing[1] = len(cv[6])

        def place(node, tok, region: Region, weight: float, ikey: tuple) -> None:
            home = region
            if cv[2] and tok is not None and tok.position is not None:
                region = tok.rect(region, *cv[3])
            if stampable is not None:            # a partial drawing's claims of an instance holding no other unit
                inner = stampable.get(node.node_id)
                if inner is not None:
                    key = (only, node.node_id, ikey, home, region, weight, cv[0].shape, cv[3], cv[4], cv[7],
                           tuple([n.gate for n in inner]))
                    hit = stamps.get(key)
                    if hit is None:
                        start = len(cv[6])
                        drawn(node, home, region, weight, ikey)
                        if len(stamps) > 4096:
                            stamps.clear()
                        stamps[key] = cv[6][start:]
                    else:
                        cv[6].extend(hit)
                    return
            drawn(node, home, region, weight, ikey)

        def drawn(node, home: Region, region: Region, weight: float, ikey: tuple) -> None:
            def draw(r):
                was_inside = cv[4]
                cv[4] = True
                for child in content_children(node):
                    render_node(child, r, weight * _sg(node), ikey)
                cv[4] = was_inside
            draw_moved(draw, home, region, (node.node_id, ikey))

        def render_node(node: Node, region: Region, weight: float, ikey: tuple = ()) -> None:
            if weight < 1e-9 or is_empty(region):
                return
            if keep is not None and node.node_id not in keep:
                return
            if getattr(node, "role", None) == "observer":
                return                           # perceptual patterns are not part of the imagery
            if isinstance(node, InWindowNode):
                if node.field is None:
                    return
                r0, r1, c0, c1 = region          # the field, resized to the window's cells
                f = node.field[np.arange(r1 - r0) * node.field.shape[0] // (r1 - r0)][
                    :, np.arange(c1 - c0) * node.field.shape[1] // (c1 - c0)]
                q0, q1, p0, p1 = _clip(region, cv[0].shape)          # only its part on the board
                f, (r0, r1, c0, c1) = f[q0 - r0:q1 - r0, p0 - c0:p1 - c0], (q0, q1, p0, p1)
                w = _clamp(weight * node.gate)
                if w > 0:
                    m, value = np.zeros(cv[0].shape, dtype=bool), np.full(cv[0].shape, 0.5)
                    m[r0:r1, c0:c1], value[r0:r1, c0:c1] = True, f
                    claim(m, value, w, node.node_id)
                if not values:
                    cv[0][r0:r1, c0:c1] *= 1.0 - w * np.clip(2 * f - 1, 0, 1)
                    cv[1][r0:r1, c0:c1] *= 1.0 - w * np.clip(1 - 2 * f, 0, 1)
                return
            if isinstance(node, ObjectNode):
                if node.is_dynamic:
                    return                       # a motion trigram specifies motion, not a static figure
                if only is not None and node.node_id not in only:
                    return                       # a drawing of only some leaves
                if seeding == bool(cv[4]):
                    return                       # bare leaves are drawn once, into the substance grid
                w = _clamp(weight * node.gate)
                mask, bands, pure, some = _leaf_cells(node.pattern, node.n_bands, node.orientation, region, cv[0].shape,
                                                      thick)
                if not values:
                    for (r0, r1, c0, c1), bit in bands:
                        cv[0][r0:r1, c0:c1] *= 1.0 - w * _clamp(2 * bit - 1)
                        cv[1][r0:r1, c0:c1] *= 1.0 - w * _clamp(1 - 2 * bit)
                if w > 0 and some:
                    dr, dc = cv[7]
                    home = mask if dr == 0 and dc == 0 else _leaf_home(node.pattern, node.n_bands, node.orientation,
                                                                       region, cv[0].shape, thick, dr, dc)
                    cv[6].append((mask, float(node.pattern[0]) if pure else 1.0, w, home, node.node_id,
                                  dr != 0 or dc != 0))
            elif isinstance(node, GroupNode):                    # an instance, drawn where it is
                index = spawn_index.get(node.node_id)
                if index is None:
                    index = spawn_index[node.node_id] = _spawn_index(node.instances)
                draw_instance(node, region, weight, ikey)
                for key in index.get(ikey, ()):
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

        if fine:
            claims, left = self._paint((H * SUBCELL, W * SUBCELL), output_root_id, place_bodies, raw=True)
            cv[5] = left.reshape(H, SUBCELL, W, SUBCELL).mean(axis=(1, 3))
            cv[6] = [_pooled(c, SUBCELL) for c in claims]
        else:
            render_node(out_root, (0, H, 0, W), 1.0)
        if raw:
            if record is not None:
                def redraw(drawing):             # one instance's claims, drawn again where it now is
                    _, _, node, region, weight, ikey, inside, offset = drawing
                    saved = cv[:]
                    cv[4], cv[6], cv[7] = inside, [], offset
                    place(node, node.instances.get(ikey), region, weight, ikey)
                    claims = cv[6]
                    cv[:] = saved
                    return claims
                self._paint_redraw = redraw
            return cv[6], cv[5]
        if only is None and not seeding and (H, W) == tuple(self.grid):
            S = self._substance(out_root.node_id)               # the substance grid, as two claims
            for moved in (False, True):
                cells = (S["s"] > 0) & (S["moved"] == moved)
                if cells.any():
                    cv[6].append((cells, S["v"], S["s"], cells, "S*" if moved else "S", moved))
        boundary = self._field_leaves(out_root.node_id)
        if values:
            return _claim_values(cv[6], (H, W), boundary)
        yang, yin = _compose_claims(cv[6], (H, W), boundary)
        if with_claimed:                         # also how strongly claims cover each cell (``_claim_values``)
            return yang, yin, cv[5], _claim_values(cv[6], (H, W), boundary)[1]
        return yang, yin, cv[5]

    # -- the substance grid ---------------------------------------------------------------------

    def _substance(self, root_id: str) -> dict:
        """A tree's substance grid, seeded on first use from its bare leaves (the initial condition): per cell a
        claimed value ``v`` (1 figure, 0 ground, 0.5 unknown), its strength ``s``, whether it arrived by motion
        (``moved``); the motion orders on it; and the fields'
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
                                     fe=np.full((H, W), 0.5), fc=np.zeros((H, W)),
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
        for o in S["orders"]:
            o[0] = o[0] & ~F
        S["orders"] = [o for o in S["orders"] if o[0].any()]

    def _field_units(self, root_id: str) -> list:
        """A tree's fields as (field, its own nodes, carried by a body?, holds a motion leaf?)."""
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
        """The substance cells that motion may carry out of fields. An anchored field's (key None):
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

    def _paint_claims(self, root_id: str, ids: frozenset, record: dict = None) -> list:
        """The claims of a drawing of only the nodes ``ids``, trails left as they were; ``record``: see ``_paint``."""
        trails, self._paint_only = dict(self._trails), ids
        if record is not None:
            self._paint_record = record
        try:
            return self._paint(self.grid, root_id, True, values=True, raw=True)[0]
        finally:
            self._paint_only, self._trails = None, trails

    def _paint_only_values(self, root_id: str, ids: frozenset, crop: Region = None, shifted=None) -> Tuple[np.ndarray, np.ndarray]:
        """The claimed values and strengths of only the nodes ``ids`` (trails left as they were), in ``crop`` if given,
        memoised by what that drawing depends on: fields are re-asserted every tick, mostly where they were.
        ``shifted``: a callable giving the same drawing's claims more cheaply than drawing it."""
        key = (self._only_key(root_id, ids), ids)
        cache = self.__dict__.setdefault("_only_paints", {})
        hit = cache.get(key)
        if hit is None:
            claims = shifted() if shifted is not None else None
            if claims is None:
                claims = self._paint_claims(root_id, ids)
            if len(cache) > 256:
                cache.clear()
            hit = cache[key] = [claims, None]
        boundary = self._field_leaves(root_id)
        if crop is not None:
            return _claim_values_in(hit[0], tuple(self.grid), boundary, crop)
        if hit[1] is None:
            hit[1] = _claim_values(hit[0], tuple(self.grid), boundary)
        return hit[1]

    def _field_leaves(self, root_id: str) -> frozenset:
        """The nodes of a tree's fields, not of the bodies they hold (their claims are the boundary
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
        """Bodies held by a field (the nearest unit above them is a field)."""
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
        """A field re-asserts the bodies it holds. For each spawning body and template position,
        instances that came to the same place in the same state merge; then, if none is at home, a new instance is
        placed there, as the first one was. With SOLID_FIGURE: only once its home is clear
        (``_home_clear``), so copies never overlap; a spawner whose home is blocked produces nothing until it clears."""
        same = lambda t: (tuple(round(v, 9) for v in t.position), tuple(tuple(o) for o in t.orders))
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
                if SOLID_FIGURE and not self._home_clear(node, first):
                    continue                     # its home is not clear: nothing new appears inside something solid
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
            if k:
                fire.append(o)                   # moved together below: stuff pressed from both sides holds
            if o[3] > 0:
                keep.append(o)
        none = np.zeros(self.grid, dtype=bool)
        for o in fire:
            against = np.logical_or.reduce([q[0] for q in fire if q[1] == o[1] and q[2] == -o[2]] + [none])
            self._shift_substance(S, o[0], o[2] if o[1] == 0 else 0, o[2] if o[1] == 1 else 0, against, min(1.0, o[4]))
        S["orders"] = keep

    @staticmethod
    def _shift_substance(S: dict, cells: np.ndarray, dy: int, dx: int, against: np.ndarray = None,
                         drive: float = 1.0) -> None:
        """Move the contents of ``cells`` by (dy, dx). A field cell gives a copy of its claim (a source); content
        stays at the board's edge; cells left behind read known empty; content arriving on content that also moved
        piles up (noisy-OR strength), elsewhere it replaces what was there. Orders follow their contents. Content
        pushes the content ahead of it, and moves only if its run can (``_stuff_run``; ``against``: content moving the
        other way this tick, which it cannot push); what leaves the board is gone."""
        H, W = cells.shape
        src_s = np.where(S["s"] > 0, S["s"], np.where(S["F"], S["fc"], 0.0))
        src_v = np.where(S["s"] > 0, S["v"], S["fe"])
        run, blocked = _stuff_run(S, cells & (src_s > 0), dy, dx, against, drive)
        cells = run & ~blocked
        rows, cols = np.nonzero(cells & (src_s > 0))
        tr, tc = rows + dy, cols + dx
        ok = (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)
        lr, lc = rows[~ok], cols[~ok]
        rows, cols, tr, tc = rows[ok], cols[ok], tr[ok], tc[ok]
        if not rows.size and not lr.size:
            return
        S["v"][lr, lc], S["s"][lr, lc], S["moved"][lr, lc] = 0.0, src_s[lr, lc], False      # left behind: empty
        if S.get("flow") is not None:            # this tick's optical flow of the substance
            S["flow"][tr, tc] = (dy, dx)
        av, aw = src_v[rows, cols].copy(), src_s[rows, cols].copy()
        S["v"][rows, cols], S["s"][rows, cols], S["moved"][rows, cols] = 0.0, aw, False   # left behind: empty
        pv, pw, pm = S["v"][tr, tc], S["s"][tr, tc], S["moved"][tr, tc]
        both = pm & (pw > 0)
        S["v"][tr, tc] = np.where(both, (pw * pv + aw * av) / np.maximum(pw + aw, 1e-12), av)
        S["s"][tr, tc] = np.where(both, 1.0 - (1.0 - pw) * (1.0 - aw), aw)
        S["moved"][tr, tc] = True
        stayed = np.ones((H, W), dtype=bool)
        stayed[rows, cols] = stayed[lr, lc] = False
        if S["orders"]:                          # all at once: orders can be many
            was = np.array([o[0] for o in S["orders"]])
            now = was & stayed
            now[:, tr, tc] |= was[:, rows, cols]
            for o, m in zip(S["orders"], now):
                o[0] = m

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
