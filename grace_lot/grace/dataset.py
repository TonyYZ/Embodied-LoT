"""Index the GRACE release: actors, manners and the Study-3 exemplar videos.

Filenames are ``<actor><F|M>_<manner>.mp4``. Actor 00 is the experimenter
performing *gestures*, so ``00F_*`` is not locomotion and is excluded.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_NAME = re.compile(r"^(\d\d)([FM])_([a-z]+)\.mp4$")


@dataclass(frozen=True)
class Video:
    path: Path
    actor: str        # e.g. "05F"
    manner: str       # e.g. "hopping"

    @property
    def stem(self) -> str:
        return self.path.stem


def parse(path: Path) -> Optional[Video]:
    m = _NAME.match(path.name)
    return Video(path, m.group(1) + m.group(2), m.group(3)) if m else None


def locomotion_videos(raw_dir: Path) -> List[Video]:
    """Every locomotion video in the release: actors 01-13, gesture videos (00F) left out."""
    videos = [parse(p) for p in sorted(Path(raw_dir).glob("*.mp4"))]
    return [v for v in videos if v is not None and v.actor != "00F"]


def manner_of(filename: str) -> str:
    return _NAME.match(filename).group(3)


def study3_exemplars(similarity_csv: Path, raw_dir: Path,
                     substitutes: Dict[str, str]) -> Dict[str, Video]:
    """manner -> the exemplar video rated in Norming Study 3.

    A missing file is replaced by ``substitutes[filename]``. That video gets
    ``substituted_for`` set so later results can flag it.
    """
    names: Dict[str, str] = {}
    with open(similarity_csv, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            for key in ("left_action_videos", "right_action_videos"):
                names[manner_of(row[key])] = row[key]
    exemplars = {}
    for manner, name in sorted(names.items()):
        path = Path(raw_dir) / name
        if not path.exists():
            if name not in substitutes:
                raise FileNotFoundError(f"{name} is missing and has no substitute in the config.")
            path = Path(raw_dir) / substitutes[name]
        exemplars[manner] = parse(path)
    return exemplars


def substituted(exemplars: Dict[str, Video], similarity_csv: Path) -> Dict[str, str]:
    """manner -> the original exemplar filename, for manners whose exemplar was substituted."""
    original = {}
    with open(similarity_csv, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            for key in ("left_action_videos", "right_action_videos"):
                original[manner_of(row[key])] = row[key]
    return {m: original[m] for m, v in exemplars.items() if v.path.name != original[m]}


def split_videos(raw_dir: Path, split_cfg: dict, which: str) -> Dict[str, List[Video]]:
    """manner -> videos for one split: "train", "val", "test" or "trainval" (train + val)."""
    test, val = set(split_cfg["test_actors"]), set(split_cfg["val_actors"])
    keep = {"train": lambda a: a not in test and a not in val,
            "val": lambda a: a in val,
            "test": lambda a: a in test,
            "trainval": lambda a: a not in test}[which]
    out: Dict[str, List[Video]] = {}
    for v in locomotion_videos(raw_dir):
        if keep(v.actor):
            out.setdefault(v.manner, []).append(v)
    return out
