"""Readable aliases for the Embodied LoT trigram vocabulary."""

from embodied_lot import Orientation, TRIGRAM_PATTERNS


# Orientations
V = Orientation.VERTICAL
H = Orientation.HORIZONTAL

# Vertical pattern names. The same tuples can be interpreted horizontally.
VOID = TRIGRAM_PATTERNS["void"]
BOTTOM = TRIGRAM_PATTERNS["bottom"]
CENTER_V = TRIGRAM_PATTERNS["center"]
MAJORITY_LOWER = TRIGRAM_PATTERNS["lower"]
UNIVERSAL = TRIGRAM_PATTERNS["universal"]
PERIPHERY_V = TRIGRAM_PATTERNS["periphery"]
MAJORITY_UPPER = TRIGRAM_PATTERNS["upper"]
TOP = TRIGRAM_PATTERNS["top"]
NEUTRAL = TRIGRAM_PATTERNS["neutral"]

# Horizontal aliases. They are intentionally the same pattern tuples.
LEFT = BOTTOM
CENTER_H = CENTER_V
MAJORITY_LEFT = MAJORITY_LOWER
RIGHT = TOP
PERIPHERY_H = PERIPHERY_V
MAJORITY_RIGHT = MAJORITY_UPPER


__all__ = [
    "V",
    "H",
    "VOID",
    "BOTTOM",
    "CENTER_V",
    "MAJORITY_LOWER",
    "UNIVERSAL",
    "PERIPHERY_V",
    "MAJORITY_UPPER",
    "TOP",
    "NEUTRAL",
    "LEFT",
    "CENTER_H",
    "MAJORITY_LEFT",
    "RIGHT",
    "PERIPHERY_H",
    "MAJORITY_RIGHT",
]
