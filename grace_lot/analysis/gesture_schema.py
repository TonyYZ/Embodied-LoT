"""Do gestures share their manner's motion schema? Rhythm, direction and alternation, not shape or size.

    python -m grace_lot.analysis.gesture_schema --run results/grace_v7_parts/desc_fix_5

The gesture boards (``grace.hands``) look at the hands in the image plane, but the experimenter mimes on her lap,
facing the camera, so a "walking" hand mostly travels toward the camera, and a hand is a few cells on the board.
Here a gesture is compared with the actors at the level of its schema instead, by a rule fixed (30 Oct 2026) before
any gesture was scored:

  * 3D: each hand's position from the body pose model's wrist (MediaPipe Pose world landmarks: depth from a body
    model), its shape from the hand model's world landmarks, both in metres; which hand is which from the pose
    model's left and right wrist. Channels as in ``grace.hands``: palm, walking fingers, other fingers, per hand.
  * window: the gesture proper (``hands.active``), two cycles (``preprocess.sampling``), the model's frame count.
  * directions: up is up (the camera is upright); along is the horizontal direction the channels move most
    (pooled). Locomotion parts: along is board x (the side view), up is minus board y.
  * per part, scale-free: its rhythm and direction profile (amplitudes at every rhythm from once per clip to twice
    per cycle, along and up, over their sum) and its engagement (that sum over the largest part's); and the four
    limb-pair couplings (``grace.descriptors``), at once per cycle, along and up.
  * a manner: its training actors' mean and spread per feature (floored at a tenth of the spread pooled over manners).
    A gesture's density under a manner: the Gaussian density, averaged over all 720 assignments of the six channels
    to the six body parts (no channel is assumed to be a leg).
Read off as ``analysis.gestures``: double-centred matrix, rank of each gesture's own manner, permutation test; then,
only after, Norming Study 1 (own manner vs foil). No programs, no fitting: if gestures do not share their actors'
schema, programs fitted to those actors cannot explain them either. Writes ``<run>/gesture_schema.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from grace_lot.analysis.gestures import human_matches, match_stats, versus_humans
from grace_lot.config import load_config
from grace_lot.grace import descriptors as D
from grace_lot.grace import hands as Hd
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.parts import PART_NAMES

KS = (1, 2, 3, 4)                     # once per clip (two cycles) .. twice per cycle
ONCE = 2
PAIRS = [(PART_NAMES.index(a), PART_NAMES.index(b)) for a, b in D.PAIRS]
CHANNEL_POINTS = {c: sorted({i for e in edges for i in e}) for c, edges in Hd.CHANNELS.items()}


def detect3d(cfg: dict, manner: str) -> dict:
    """Per frame: the hand model's image and world landmarks (2 slots) and the pose model's wrists (image, world).
    Cached in ``<processed>/hands3d/``."""
    d = Path(cfg["data"]["processed_dir"]) / "hands3d"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"00F_{manner}.npz"
    if path.exists():
        return dict(np.load(path))
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    models = Path(cfg["data"]["model_path"]).parent
    hopts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models / "hand_landmarker.task")),
                                         running_mode=vision.RunningMode.VIDEO, num_hands=2)
    popts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(models / "pose_landmarker_heavy.task")),
                                         running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(str(Hd.gesture_path(cfg["data"]["raw_dir"], manner)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = {k: [] for k in ("img", "world", "wrist_img", "wrist_world")}
    with vision.HandLandmarker.create_from_options(hopts) as hl, vision.PoseLandmarker.create_from_options(popts) as pl:
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            t = int(round(n * 1000 / fps))
            n += 1
            hr, pr = hl.detect_for_video(im, t), pl.detect_for_video(im, t)
            a, b = np.full((2, 21, 2), np.nan), np.full((2, 21, 3), np.nan)
            for i, (pts, wp) in enumerate(zip(hr.hand_landmarks[:2], hr.hand_world_landmarks[:2])):
                a[i] = [[p.x * w, p.y * h] for p in pts]
                b[i] = [[p.x, p.y, p.z] for p in wp]
            wi, ww = np.full((2, 2), np.nan), np.full((2, 3), np.nan)
            if pr.pose_landmarks:
                for s, j in enumerate((15, 16)):                        # left, right wrist
                    p, q = pr.pose_landmarks[0][j], pr.pose_world_landmarks[0][j]
                    wi[s], ww[s] = [p.x * w, p.y * h], [q.x, q.y, q.z]
            for k, v in zip(out, (a, b, wi, ww)):
                out[k].append(v)
    cap.release()
    res = {k: np.array(v) for k, v in out.items()}
    np.savez_compressed(path, **res)
    return res


def hands_3d(r: dict) -> np.ndarray:
    """(F, 2, 21, 3) landmarks in metres (x right, y down, z away), slot 0 the left wrist's hand, 1 the right's: each
    detected hand goes to the nearer pose wrist (in the image) and is placed with its wrist at that wrist."""
    F = len(r["img"])
    out = np.full((F, 2, 21, 3), np.nan)
    for f in range(F):
        dets = [i for i in range(2) if not np.isnan(r["img"][f, i, 0, 0])]
        if not dets or np.isnan(r["wrist_img"][f]).all():
            continue
        dist = np.array([[np.linalg.norm(r["img"][f, i, 0] - r["wrist_img"][f, s]) for s in range(2)] for i in dets])
        if len(dets) == 2:
            slots = [0, 1] if dist[0, 0] + dist[1, 1] <= dist[0, 1] + dist[1, 0] else [1, 0]
        else:
            slots = [int(np.nanargmin(dist[0]))]
        for i, s in zip(dets, slots):
            shape = r["world"][f, i] - r["world"][f, i, 0]                 # about the hand's own wrist
            out[f, s] = r["wrist_world"][f, s] + shape
    t = np.arange(F)                                                       # gaps: linear interpolation
    for s in range(2):
        ok = ~np.isnan(out[:, s, 0, 0])
        if ok.mean() < 0.2:
            out[:, s] = np.nan
        elif ok.any():
            flat = out[:, s].reshape(F, -1)
            for k in range(flat.shape[1]):
                flat[~ok, k] = np.interp(t[~ok], t[ok], flat[ok, k])
    return out


def gesture_signals(cfg: dict, manner: str, frames: int) -> np.ndarray:
    """(T, 6, 2) channel centres over the gesture's two cycles: along and up (metres)."""
    from grace_lot.grace.temporal import sample_window
    P = hands_3d(detect3d(cfg, manner))
    img, a0 = Hd.active(P[..., :2] * 1000.0)                               # its window, as the 2D boards' (y down)
    P = P[a0:a0 + len(img)]
    P[np.isnan(img)[..., 0]] = np.nan                                      # a resting hand dropped by ``active``
    filled = np.where(np.isnan(P), 0.0, P)
    a, b = sample_window(filled[..., :2].reshape(len(filled), 42, 2), cfg["preprocess"].get("sampling", "segment"))
    P = P[np.round(np.linspace(a, b - 1, frames)).astype(int)]
    C = np.stack([np.nanmean(P[:, h, pts], axis=1) for h in range(2) for pts in CHANNEL_POINTS.values()], axis=1)
    path = (C[..., [0, 2]] - np.nanmean(C[..., [0, 2]], axis=0)).reshape(-1, 2)
    path = path[~np.isnan(path).any(1)]
    d = np.linalg.svd(path, full_matrices=False)[2][0]                     # the horizontal direction moved most
    return np.stack([C[..., 0] * d[0] + C[..., 2] * d[1], -C[..., 1]], axis=-1)   # NaN: a hand never seen


def couplings(z: np.ndarray) -> np.ndarray:
    """(P, 2) once-per-cycle along/up components -> (4, 2) limb-pair couplings."""
    return np.array([2 * np.real(z[a] * np.conj(z[b])) / np.maximum(np.abs(z[a]) ** 2 + np.abs(z[b]) ** 2, 1e-12)
                     for a, b in PAIRS])


def features(sig: np.ndarray):
    """(T, P, 2) along/up signals -> per part (P, 9): the rhythm-direction profile (8) and engagement; the parts'
    once-per-cycle components (P, 2), for the couplings."""
    Z = np.fft.rfft(np.nan_to_num(sig - np.nanmean(sig, axis=0)), axis=0) / len(sig)
    amp = np.abs(Z[list(KS)]).transpose(1, 0, 2).reshape(sig.shape[1], -1)   # (P, 4 rhythms x 2 directions)
    total = amp.sum(1)
    prof = amp / np.maximum(total, 1e-12)[:, None]
    eng = total / max(total.max(), 1e-12)
    return np.concatenate([prof, eng[:, None]], axis=1), Z[ONCE]


def actor_features(cfg: dict, videos) -> list:
    from grace_lot.inference.run import observations_for
    out = []
    for boards in observations_for(cfg, videos):
        s = D.board_signals(boards.astype(float), cfg["interface"]["lot_salience"])    # (T, P, 4)
        out.append(features(np.stack([s[..., 0], -s[..., 1]], axis=-1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = load_config(args.config)
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    cfg["inference"]["likelihood"] = "descriptors"                          # observations as part boards
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    assert not any(v.actor in set(cfg["split"]["test_actors"]) for vs in train.values() for v in vs)
    manners = sorted(train)
    T = cfg["preprocess"]["frames"]
    stats = {}
    for m in manners:
        fs = actor_features(cfg, train[m])
        F, C = np.stack([f for f, _ in fs]), np.stack([couplings(z) for _, z in fs])
        stats[m] = (F.mean(0), F.var(0), C.mean(0), C.var(0))
        print(f"{m}: {len(fs)} actors", flush=True)
    vf = np.mean([s[1] for s in stats.values()], axis=0).mean(0)           # pooled spread per feature
    vc = np.mean([s[3] for s in stats.values()], axis=0)
    gest = {}
    for m in manners:
        gest[m] = features(gesture_signals(cfg, m, T))
        print(f"gesture {m}: engagement {np.round(gest[m][0][:, -1], 2)}", flush=True)
    perms = np.array(list(itertools.permutations(range(6))))                # body part q is channel p[q]
    per = {g: (gest[g][0][perms], np.stack([couplings(gest[g][1][p]) for p in perms])) for g in manners}
    M = np.empty((len(manners),) * 2)
    for i, m in enumerate(manners):
        mf, sf, mc, sc = stats[m]
        sf, sc = np.maximum(sf, 0.1 * vf), np.maximum(sc, 0.1 * vc)
        for j, g in enumerate(manners):
            f, c = per[g]                                                   # (720, 6, 9), (720, 4, 2)
            ll = -0.5 * ((((f - mf) ** 2) / sf + np.log(2 * np.pi * sf)).sum(axis=(1, 2))
                         + (((c - mc) ** 2) / sc + np.log(2 * np.pi * sc)).sum(axis=(1, 2)))
            M[i, j] = logsumexp(ll) - np.log(len(perms))
    res = {"manners": manners, "lpd": M.tolist(), "schema": match_stats(M)}
    s = res["schema"]
    print(f"schema: mean rank of own manner {s['mean_rank_own_manner']:.1f} (chance {s['chance_mean_rank']:.1f}), "
          f"top-1 {s['top1']:.2f}, top-5 {s['top5']:.2f}, p {s['p_permutation']:.4f}")
    study1 = Path(cfg["data"]["raw_dir"]) / "Norming_Study_1.csv"           # human data enter only here
    if study1.exists():
        human = human_matches(study1, manners, set(cfg["split"]["test_actors"]))
        res["humans"] = {"ratings": human, "schema": versus_humans(M, manners, human)}
        h = res["humans"]["schema"]
        print(f"Study 1: own manner beats the foil for {h['own_beats_foil']:.2f} of {len(h['gestures'])} gestures "
              f"(humans: {np.mean([v['own'] > v['foil_rating'] for v in human.values()]):.2f}); margin vs human margin "
              f"rho {h['spearman_with_humans']:.2f} (p {h['p_spearman']:.3f})")
    (run / "gesture_schema.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
