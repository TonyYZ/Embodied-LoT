"""Data and scoring shared by the two learners.

Both the MCMC learner (``concept_learning``) and the DreamCoder-style learner
(``dreamcoder_lot``) score a ``ConceptGraph`` on the same data:

* ``Observation``  one static scene with a graded response, scored with a
  stateless ``evaluate_against()``
* ``Trajectory``   a sequence of scenes, replayed with ``step()`` after a
  gate reset, so that dashed gates carry from frame t to frame t+1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from embodied_lot import ConceptGraph, Orientation, ReceptiveField


@dataclass
class Observation:
    """One scene paired with a graded response in [0, 1].

    Examples of a response: a generalisation judgement, a similarity
    rating, or a choice proportion.
    """
    field: ReceptiveField
    response: float
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.response <= 1.0:
            raise ValueError("response must be in [0, 1].")


@dataclass
class Trajectory:
    """A sequence of scenes and one response per frame."""
    frames: List[ReceptiveField]
    responses: List[float]
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.responses):
            raise ValueError(f"frames and responses must have the same length; "
                             f"got {len(self.frames)} frames and {len(self.responses)} responses.")
        for i, r in enumerate(self.responses):
            if not 0.0 <= r <= 1.0:
                raise ValueError(f"response[{i}] = {r} is not in [0, 1].")

    @property
    def length(self) -> int:
        return len(self.frames)


Item = Union[Observation, Trajectory]


def predict(graph: ConceptGraph, item: Item) -> List[float]:
    """The graph's prediction for every frame of ``item``."""
    if isinstance(item, Trajectory):
        graph.reset_gates()
        return [graph.step(f) for f in item.frames]
    return [graph.evaluate_against(item.field)]


def responses(item: Item) -> List[float]:
    return list(item.responses) if isinstance(item, Trajectory) else [item.response]


def targets_and_weights(data: Sequence[Item]) -> Tuple[List[float], List[float]]:
    """Return targets and weights flattened over all frames. Each frame takes its item's weight."""
    targets, weights = [], []
    for item in data:
        r = responses(item)
        targets += r
        weights += [item.weight] * len(r)
    return targets, weights


def frame_scores(graph: ConceptGraph, data: Sequence[Item]
                 ) -> Tuple[List[float], List[float], List[float]]:
    """Return predictions, targets and weights flattened over all frames."""
    preds = [p for item in data for p in predict(graph, item)]
    return (preds, *targets_and_weights(data))


def weighted_mse(preds: Sequence[float], targets: Sequence[float],
                 weights: Sequence[float]) -> float:
    total_weight = sum(weights) or 1.0
    return float(sum(w * (p - t) ** 2 for p, t, w in zip(preds, targets, weights)) / total_weight)


def evaluate_on_dataset(graph: ConceptGraph, data: Sequence[Item]) -> Dict[str, object]:
    """Summary fit statistics: mean_error, rmse, correlation, predictions, targets."""
    preds, targets, _ = frame_scores(graph, data)
    p, t = np.array(preds), np.array(targets)
    corr = (float(np.corrcoef(p, t)[0, 1])
            if len(preds) > 1 and p.std() > 0 and t.std() > 0 else float("nan"))
    return {
        "mean_error": float(np.mean(np.abs(p - t))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "correlation": corr,
        "predictions": preds,
        "targets": targets,
    }


def make_trigram_field(pattern: Sequence[float],
                       orientation: Orientation = Orientation.VERTICAL,
                       *, size: int = 9) -> ReceptiveField:
    """A size x size field whose thirds follow ``pattern``, read bottom-up (vertical) or left-right."""
    arr = np.zeros((size, size), dtype=float)
    parts = np.array_split(np.arange(size), len(pattern))
    if orientation == Orientation.VERTICAL:
        for value, rows in zip(pattern, reversed(parts)):
            arr[rows, :] = value
    else:
        for value, cols in zip(pattern, parts):
            arr[:, cols] = value
    return ReceptiveField(arr)
