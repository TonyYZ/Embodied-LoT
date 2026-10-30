"""Do locomotion programs explain the gestures that mime them? Training actors and gesture videos only.

    python -m grace_lot.analysis.gestures --run results/grace_v7_parts/all26_current [--draws 1000]

Each GRACE manner has a gesture video (``00F_<manner>.mp4``): the experimenter, seated, mimes the manner with
her hands. The gestures become six-channel boards (``grace.hands``: hand-centred, three channels per hand) and are
scored under each manner's locomotion posterior, fitted to the training actors' bodies, exactly as held-out
locomotion videos are (``inference.predictive``): with that manner's observation noise and the outputs matched
to the channels anonymously, so a program's "leg" may become a finger.

    lpd[m, g] = log mean_{r ~ posterior of manner m} p(gesture g | r)

Gestures differ in how easy they are for any program, and manners in how broad their posteriors are, so the
matrix is double-centred (row and column means removed) before the match is read off:
  * per gesture: the rank of its own manner among the 26 (1 = its own manner explains it best), and per manner the
    rank of its own gesture;
  * the mean of the matched (diagonal) entries, against 10,000 permutations of the gesture labels.
The same is done for a theory-light baseline on the same features: the negative squared distance between the
gesture and the manner's mean training features, under the best channel matching and phase.
Against people (GRACE Norming Study 1, read only after the scores are computed): for each gesture, participants
rated its match with its own manner and with one foil manner. Reported: how often the model also prefers the own
manner to the foil, and whether its margin follows the participants' margin across gestures (ratings of
sealed test actors' videos are left out).
For each manner, the sampled program that best explains its own gesture is shown on the gesture (``gifs/``), and
each of its outputs is traced to the body part it explains on the medoid training actor and the hand channel it
explains in the gesture; ``part_channel_pairs`` counts these pairs over the manners (outputs drawn identically to
another output of the same program are left out: the matching among them is arbitrary).
With motion scoring (``inference.likelihood = "descriptors"``) the gestures' hand channels are scored like body parts
on their swing amplitudes alone, with the manner's noise pooled over parts (so the matching stays anonymous); limb-pair
couplings are left out, as no hand channel pair stands for a limb pair. The baseline is then the distance of the
gesture's amplitudes to the manner's mean amplitudes, in noise units, under the best channel matching.
Nothing here was tuned on these results. Writes ``<run>/gestures/gestures.json`` and ``<run>/gestures/gifs/``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.hands import CHANNEL_NAMES, gesture_boards, gesture_features, gesture_path
from grace_lot.grace.parts import PART_NAMES
from grace_lot.inference.predictive import log_predictive
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot


def double_centred(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def match_stats(M: np.ndarray, n_perm: int = 10000, seed: int = 0) -> dict:
    """Ranks of the matched entries of a (manners x gestures) score matrix, same order on both axes."""
    D = double_centred(M)
    n = len(D)
    own_gesture = [int((D[:, g] > D[g, g]).sum()) + 1 for g in range(n)]       # rank of g's manner, per gesture
    own_manner = [int((D[m] > D[m, m]).sum()) + 1 for m in range(n)]            # rank of m's gesture, per manner
    diag = float(np.trace(D) / n)
    rng = np.random.default_rng(seed)
    null = np.array([D[np.arange(n), rng.permutation(n)].mean() for _ in range(n_perm)])
    return {"rank_of_own_manner": own_gesture, "rank_of_own_gesture": own_manner,
            "mean_rank_own_manner": float(np.mean(own_gesture)), "top1": float(np.mean(np.array(own_gesture) == 1)),
            "top5": float(np.mean(np.array(own_gesture) <= 5)), "diagonal_mean": diag,
            "p_permutation": float((1 + (null >= diag).sum()) / (1 + n_perm)), "chance_mean_rank": (n + 1) / 2}


def human_matches(csv_path, manners: list, sealed: set) -> dict:
    """GRACE Norming Study 1: participants rated how well each gesture matches an actor's locomotion (1-7), for
    the gesture's own manner and for one foil manner. Per gesture: the mean rating of its own manner and of its
    foil, over the rated videos of actors outside the sealed test set."""
    import csv
    from collections import defaultdict
    r = defaultdict(list)
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["action_videos"][:3] not in sealed:
                r[(row["gesture_videos"][4:-4], row["action_videos"][4:-4])].append(float(row["rating"]))
    out = {}
    for g in manners:
        foils = [a for (gg, a) in r if gg == g and a != g]
        if (g, g) in r and len(foils) == 1:
            out[g] = {"foil": foils[0], "own": float(np.mean(r[(g, g)])), "foil_rating": float(np.mean(r[(g, foils[0])]))}
    return out


def versus_humans(M: np.ndarray, manners: list, human: dict) -> dict:
    """The Study-1 choice for each gesture, own manner vs its foil, made by a score matrix (double-centred, rows =
    manners, columns = gestures), and how the model's margin follows the humans' margin (Spearman, over gestures)."""
    from scipy.stats import spearmanr
    D = double_centred(M)
    gs = [g for g in manners if g in human and human[g]["foil"] in manners]
    margin = [D[manners.index(g), manners.index(g)] - D[manners.index(human[g]["foil"]), manners.index(g)] for g in gs]
    hmargin = [human[g]["own"] - human[g]["foil_rating"] for g in gs]
    rho, p = spearmanr(margin, hmargin)
    return {"gestures": gs, "own_beats_foil": float(np.mean(np.array(margin) > 0)), "margin": margin,
            "human_margin": hmargin, "spearman_with_humans": float(rho), "p_spearman": float(p)}


def descriptor_baseline(gest: np.ndarray, lik) -> float:
    """-min over channel matching of the squared distance, in noise units, of the gesture's amplitudes to the
    manner's mean training amplitudes (motion scoring)."""
    from grace_lot.inference.descriptor_likelihood import video_descriptors
    a = video_descriptors(gest, lik.salience)[0]                                   # (channels, A)
    C = (((a[:, None] - lik.amps.mean(0)[None]) ** 2) / lik.s2.mean(0)).sum(-1)   # (channel, part)
    r, c = linear_sum_assignment(C)
    return -float(C[r, c].sum())


def lik_train_render(cfg: dict, prog) -> np.ndarray:
    from grace_lot.lot_adapter.adapter import rollout
    y = rollout(prog, cfg["preprocess"]["frames"], cfg["preprocess"]["field_size"])
    return y if y.ndim == 4 else y[:, None]


def feature_baseline(gest: np.ndarray, train_feats: list, phases: int) -> float:
    """-min over phase and channel matching of the squared distance to the manner's mean features."""
    mean = np.mean(train_feats, axis=0)                                           # (T, P, F)
    best = np.inf
    for s in range(phases):
        g = np.roll(gest, -s, axis=0)
        C = ((g[:, :, None] - mean[:, None]) ** 2).sum(axis=(0, 3))             # (gesture channel, part)
        r, c = linear_sum_assignment(C)
        best = min(best, C[r, c].sum())
    return -float(best)


CHANNEL_COLOURS = [(0.95, 0.95, 0.95), (1.0, 0.15, 0.2), (0.2, 0.45, 1.0),       # hand 1: palm, walking, other
                   (0.6, 0.6, 0.6), (0.1, 0.85, 0.9), (1.0, 0.55, 0.1)]       # hand 2


def tint(chans, colours) -> np.ndarray:
    """(C, H, W) -> RGB, each channel in its colour (max-composited); None: purple (matched to nothing)."""
    out = np.zeros(chans.shape[1:] + (3,))
    for c, col in zip(chans, colours):
        out = np.maximum(out, np.clip(c, 0, 1)[..., None] * np.array(col or (0.75, 0.3, 0.9)))
    return out


def gesture_gif(cfg: dict, manner: str, y: np.ndarray, channel_of: dict, path: Path) -> None:
    """The gesture video | its hand channels | a locomotion program's outputs, tinted by the channel each matches."""
    from grace_lot.analysis.gallery import CELL, board_px, gif, video_frames
    from grace_lot.analysis.inspect_parts import figure
    boards, idx = gesture_boards(cfg, manner)
    T = len(boards)
    out_cols = [CHANNEL_COLOURS[channel_of[k]] if k in channel_of else None for k in range(y.shape[1])]
    gif([video_frames(gesture_path(cfg["data"]["raw_dir"], manner), idx, 12 * CELL),
         [board_px(tint(boards[t], CHANNEL_COLOURS)) for t in range(T)],
         [board_px(tint(figure(y[t]), out_cols)) for t in range(T)]], path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    from grace_lot.inference.run import build, observations_for
    run = Path(args.run)
    cfg = load_config(args.config)
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    cfg["lot"] = saved_lot(rc["lot"])
    apply_semantics(cfg["lot"])
    manners = sorted(f.stem for f in (run / "posteriors").glob("*.json"))
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    assert not any(v.actor in set(cfg["split"]["test_actors"]) for m in manners for v in train[m])
    motion = cfg["inference"].get("likelihood") == "descriptors"
    G = [gesture_boards(cfg, m)[0].astype(float) if motion else gesture_features(cfg, m) for m in manners]
    rng = random.Random(0)
    lpd = np.empty((len(manners), len(manners)))
    base = np.empty_like(lpd)
    best = {}
    out = run / "gestures"
    (out / "gifs").mkdir(parents=True, exist_ok=True)
    pairs = np.zeros((len(PART_NAMES), len(CHANNEL_NAMES)), int)       # body part x hand channel, same output
    for i, m in enumerate(manners):
        obs = observations_for(cfg, train[m])
        lik_train = build(cfg, obs)[3]
        noise = dict(s2=lik_train.s2, s2d=lik_train.s2d)
        if motion:                               # pooled over parts; couplings carry no weight (see above)
            noise = dict(s2=np.broadcast_to(lik_train.s2.mean(0), lik_train.s2.shape).copy(),
                         s2d=np.full_like(lik_train.s2d, 1e6))
        lik = build(cfg, G, noise=noise)[3]
        post = json.loads((run / "posteriors" / f"{m}.json").read_text())["posterior"]
        progs, weights = zip(*post)
        sample = rng.choices(progs, weights, k=args.draws)
        cache = {s: lik.per_video_scores(parse_program(s)) for s in set(sample)}
        lpd[i] = log_predictive(np.array([cache[s] for s in sample]))
        top = max(cache, key=lambda s: cache[s][i])                       # the sampled program best for its gesture
        prog = parse_program(top)
        y = lik_train_render(cfg, prog)
        rows, cols, _ = lik.assignments(prog)[i]
        channel_of = {int(r): int(c) for r, c in zip(rows, cols) if r < y.shape[1]}
        O = np.stack(obs)
        med = int(np.argmin(((O - O.mean(0)) ** 2).mean(axis=(1, 2, 3))))
        rows, cols, _ = lik_train.assignments(prog)[med]
        part_of = {int(r): int(c) for r, c in zip(rows, cols) if r < y.shape[1]}
        same = lambda k: any(np.array_equal(y[:, k], y[:, j]) for j in range(y.shape[1]) if j != k)
        for k in set(channel_of) & set(part_of):
            if not same(k):                          # identical outputs are matched arbitrarily among themselves
                pairs[part_of[k], channel_of[k]] += 1
        best[m] = {"program": top, "lpd_own_gesture": float(cache[top][i]),
                   "outputs": {k: {"body_part": PART_NAMES[part_of[k]] if k in part_of else None,
                                   "hand_channel": CHANNEL_NAMES[channel_of[k]] if k in channel_of else None}
                               for k in range(y.shape[1])}, "gif": f"gifs/{m}.gif"}
        gesture_gif(cfg, m, y, channel_of, out / "gifs" / f"{m}.gif")
        base[i] = [descriptor_baseline(g, lik_train) if motion else feature_baseline(g, obs, cfg["inference"].get("phases", 1))
                   for g in G]
        print(f"{m}: {len(cache)} programs; own gesture lpd {lpd[i, i]:.1f}, best {manners[int(lpd[i].argmax())]}",
              flush=True)
    res = {"manners": manners, "lpd": lpd.tolist(), "baseline": base.tolist(), "best_for_own_gesture": best,
           "lot": match_stats(lpd), "features": match_stats(base), "draws": args.draws,
           "part_channel_pairs": {"parts": PART_NAMES, "channels": CHANNEL_NAMES, "counts": pairs.tolist()}}
    study1 = Path(cfg["data"]["raw_dir"]) / "Norming_Study_1.csv"            # human data enter only here
    if study1.exists():
        human = human_matches(study1, manners, set(cfg["split"]["test_actors"]))
        res["humans"] = {"ratings": human, "lot": versus_humans(lpd, manners, human),
                         "features": versus_humans(base, manners, human)}
        for k in ("lot", "features"):
            h = res["humans"][k]
            print(f"Study 1, {k}: own manner beats the foil for {h['own_beats_foil']:.2f} of {len(h['gestures'])} "
                  f"gestures (humans: {np.mean([v['own'] > v['foil_rating'] for v in human.values()]):.2f}); "
                  f"margin vs human margin rho {h['spearman_with_humans']:.2f} (p {h['p_spearman']:.3f})")
    (out / "gestures.json").write_text(json.dumps(res, indent=1))
    for k in ("lot", "features"):
        s = res[k]
        print(f"{k}: mean rank of own manner {s['mean_rank_own_manner']:.1f} (chance {s['chance_mean_rank']:.1f}), "
              f"top-1 {s['top1']:.2f}, top-5 {s['top5']:.2f}, diagonal {s['diagonal_mean']:.2f}, "
              f"p {s['p_permutation']:.4f}")


if __name__ == "__main__":
    main()
