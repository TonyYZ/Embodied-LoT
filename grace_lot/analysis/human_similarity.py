"""The human 26 x 26 similarity matrix from GRACE Norming Study 3 (ratings 1-7)."""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grace_lot.grace.dataset import manner_of

Rating = Tuple[str, str, str, float]     # (subject, manner_a, manner_b, rating)


def load_ratings(path) -> List[Rating]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [(r["sub_number"], manner_of(r["left_action_videos"]),
                 manner_of(r["right_action_videos"]), float(r["rating"])) for r in csv.DictReader(f)]


def similarity_matrix(ratings: Sequence[Rating], manners: Sequence[str],
                      subjects: Optional[Sequence[str]] = None) -> np.ndarray:
    """Mean rating per unordered pair. ``subjects`` may repeat (bootstrap). Unrated pairs are NaN; the diagonal is NaN."""
    by_subject: Dict[str, List[Rating]] = defaultdict(list)
    for r in ratings:
        by_subject[r[0]].append(r)
    idx = {m: i for i, m in enumerate(manners)}
    total = np.zeros((len(manners),) * 2)
    count = np.zeros_like(total)
    for s in (subjects if subjects is not None else by_subject):
        for _, a, b, v in by_subject[s]:
            i, j = idx[a], idx[b]
            total[i, j] += v; total[j, i] += v
            count[i, j] += 1; count[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        m = total / count
    np.fill_diagonal(m, np.nan)
    return m


def subjects(ratings: Sequence[Rating]) -> List[str]:
    return sorted({r[0] for r in ratings})
