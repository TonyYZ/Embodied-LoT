"""Held-out recognition: do manner programs inferred from some actors fit new actors?

For each held-out video v and each manner m:

    score(v, m) = E_{r ~ p(r | training videos of m)} [ D(v, lot_trajectory(r)) ]

v is assigned to the manner with the lowest expected distance. This criterion
does not depend on beta, so settings with different beta can be compared
on equal terms. D is the run's own distance: with ``inference.temporal_weight`` w it
includes w x the frame-to-frame-change term, and with a body-part interface it is the
matched part distance (one output-to-part mapping per video). It reports top-1 and top-5 accuracy and the mean
reciprocal rank (chance: 1/26 and 5/26).

Kinematic baselines on the same split:
  kin_prototype  nearest manner mean of the training field sequences
  kin_1nn        nearest single training video (field sequences)
  kin_joints_1nn nearest single training video (joint DTW); slower, optional
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from grace_lot.grace.dataset import Video
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.grace.trajectories import resample
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.distance import DISTANCES, dtw
from grace_lot.lot_adapter.program import parse_program


def load_posteriors(post_dir: Path, manners: Sequence[str]):
    return {m: [(parse_program(s), w) for s, w in json.loads((Path(post_dir) / f"{m}.json").read_text())["posterior"]]
            for m in manners}


def predictive_fields(cfg: dict, posteriors, n_samples: int = 300, seed: int = 0) -> Dict[str, List[np.ndarray]]:
    """For each manner, rendered fields of ``n_samples`` posterior draws (with replacement, by weight)."""
    pre = cfg["preprocess"]
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    rng = random.Random(seed)
    out = {}
    for m, post in posteriors.items():
        progs, w = zip(*post)
        out[m] = np.stack([renderer(p) for p in rng.choices(progs, w, k=n_samples)])
    return out


def _parts_scores(cfg: dict, posteriors, held_t: dict, w: float, n_samples: int = 300, seed: int = 0):
    """scores[video][manner] = E_post[matched part distance] for the body-part interfaces."""
    from grace_lot.grace.parts import part_fields
    from grace_lot.inference.likelihood import MatchedPartsLikelihood, MatchedScanLikelihood
    pre, itf = cfg["preprocess"], cfg.get("interface", {})
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    stems = list(held_t)
    if itf.get("representation") == "parts_scan":
        from grace_lot.grace.scanner import scan_parts
        obs = [scan_parts(held_t[s], pre, itf["scan_softness"], itf["edge_samples"]) for s in stems]
        lik = MatchedScanLikelihood(obs, renderer, 1.0, itf["scan_softness"], itf["lot_salience"], w)
    else:
        lik = MatchedPartsLikelihood([part_fields(held_t[s], pre) for s in stems], renderer, 1.0, w)
    rng = random.Random(seed)
    scores = {s: {} for s in stems}
    for m, post in posteriors.items():
        progs, weights = zip(*post)
        draws = rng.choices(progs, weights, k=n_samples)
        cache = {}
        for p in draws:
            if p not in cache:
                cache[p] = np.array([d for _, _, d in lik.assignments(p)])
        mean = np.mean([cache[p] for p in draws], axis=0)            # (videos,)
        for s, d in zip(stems, mean):
            scores[s][m] = float(d)
    return scores


def _rank_summary(scores: Dict[str, Dict[str, float]], truth: Dict[str, str]) -> Dict[str, float]:
    """scores[video][manner] (lower is better); truth[video] = its manner."""
    ranks = []
    for v, s in scores.items():
        order = sorted(s, key=s.get)
        ranks.append(order.index(truth[v]) + 1)
    ranks = np.array(ranks)
    return {"top1": float(np.mean(ranks == 1)), "top5": float(np.mean(ranks <= 5)),
            "mrr": float(np.mean(1.0 / ranks)), "mean_rank": float(ranks.mean()), "n": int(len(ranks))}


def evaluate(cfg: dict, post_dir: Path, train: Dict[str, List[Video]], held: Dict[str, List[Video]],
             joints: bool = False, unsealed: bool = False) -> Dict[str, object]:
    sealed = set(cfg.get("split", {}).get("test_actors", []))
    if not unsealed and any(v.actor in sealed for vs in held.values() for v in vs):
        raise PermissionError("held-out set contains sealed test actors; pass unsealed=True only after freezing")
    if any(v.actor in sealed for vs in train.values() for v in vs):
        raise PermissionError("training set contains sealed test actors")
    manners = sorted(train)
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    dist = DISTANCES[cfg["inference"]["distance"]]
    preds = predictive_fields(cfg, load_posteriors(post_dir, manners))
    held_t = {v.stem: load_trajectory(cfg, v) for vs in held.values() for v in vs}
    truth = {v.stem: v.manner for vs in held.values() for v in vs}
    train_t = {m: [load_trajectory(cfg, v) for v in vs] for m, vs in train.items()}

    # E_post[D(v, render(r))]; vectorised over posterior draws when D is the mean squared difference
    w = cfg["inference"].get("temporal_weight", 0.0)
    representation = cfg.get("interface", {}).get("representation", "occupancy")
    if representation != "occupancy":
        lot = _parts_scores(cfg, load_posteriors(post_dir, manners), held_t, w)
    elif cfg["inference"]["distance"] == "euclid":
        lot = {s: {m: float(((preds[m] - t.fields) ** 2).mean()
                            + (w * ((np.diff(preds[m], axis=1) - np.diff(t.fields, axis=0)) ** 2).mean() if w else 0.0))
                   for m in manners} for s, t in held_t.items()}
    else:
        lot = {s: {m: float(np.mean([dist(t.fields, y) for y in preds[m]])) for m in manners}
               for s, t in held_t.items()}
    proto = {m: np.mean([t.fields for t in ts], axis=0) for m, ts in train_t.items()}
    kin_proto = {s: {m: dist(t.fields, proto[m]) for m in manners} for s, t in held_t.items()}
    kin_1nn = {s: {m: min(dist(t.fields, u.fields) for u in train_t[m]) for m in manners}
               for s, t in held_t.items()}
    results = {"lot": _rank_summary(lot, truth), "kin_prototype": _rank_summary(kin_proto, truth),
               "kin_1nn": _rank_summary(kin_1nn, truth)}
    if joints:
        J = lambda t: resample(t.joints_full, 48)
        tj = {m: [J(u) for u in ts] for m, ts in train_t.items()}
        kin_j = {s: {m: min(dtw(J(t), u) for u in tj[m]) for m in manners} for s, t in held_t.items()}
        results["kin_joints_1nn"] = _rank_summary(kin_j, truth)
    results["per_video_lot_rank"] = {s: sorted(lot[s], key=lot[s].get).index(truth[s]) + 1 for s in lot}
    by_manner = {}
    for s, r in results["per_video_lot_rank"].items():
        by_manner.setdefault(truth[s], []).append(r)
    results["per_video_lot_rank_mean"] = {m: float(np.mean(r)) for m, r in sorted(by_manner.items())}
    return results
