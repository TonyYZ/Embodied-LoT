"""The GRACE gesture videos (``00F_<manner>.mp4``) as part boards, so locomotion programs can be scored on them.

    python -m grace_lot.grace.hands              # detect hands in every gesture video (cached), print a summary

In each gesture the experimenter, seated, mimes a manner of locomotion with her hands (a hand as the walker, its
fingers as legs; often the other forearm as the ground). Hands come from MediaPipe Hand Landmarker
(``data/models/hand_landmarker.task``, 21 landmarks per hand, VIDEO mode, up to two hands), cached in
``data/processed/hands/<stem>.npz``.

The gesture interface, fixed before any gesture was scored (and never chosen by results):
  * frame: hand-centred. All hand landmarks of the video's sampled window are scaled together so that their
    larger extent is one body height, centred in the board window as a standing body is (a locomotion body is one
    body height tall in ``preprocess.window_y``); image y points down, as on the boards.
  * time: two cycles of the hands' motion (``preprocess.sampling``, ``grace.temporal``), resampled to the frames.
  * parts: six channels, as many as a body has, three per hand (the hand image-left first): palm (wrist and the
    knuckles), the index and middle fingers (the legs of a "walking" hand), and the thumb, ring and little fingers.
    A missing hand leaves its channels empty. Locomotion outputs are matched to these channels anonymously, as
    they are to body parts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
CHANNELS = {"palm": [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)],
            "walking fingers": [(5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12)],
            "other fingers": [(0, 1), (1, 2), (2, 3), (3, 4), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
                              (19, 20)]}
CHANNEL_NAMES = [f"{side} {c}" for side in ("hand 1", "hand 2") for c in CHANNELS]


def gesture_path(raw_dir, manner: str) -> Path:
    return Path(raw_dir) / f"00F_{manner}.mp4"


def detect_hands(video: Path, model: Path) -> dict:
    """Per frame, up to two hands' 21 image landmarks in pixels: ``image`` (F, 2, 21, 2), NaN where absent."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=2)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(round(n * 1000 / fps)))
            n += 1
            hands = np.full((2, 21, 2), np.nan)
            for i, pts in enumerate(res.hand_landmarks[:2]):
                hands[i] = [[p.x * w, p.y * h] for p in pts]
            frames.append(hands)
    cap.release()
    return {"image": np.array(frames), "fps": fps, "size": np.array([w, h])}


def load_hands(cfg: dict, manner: str) -> dict:
    """Detected hands of one gesture video (cached), with the two hands put in a consistent order: by image x."""
    d = Path(cfg["data"]["processed_dir"]) / "hands"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"00F_{manner}.npz"
    if not path.exists():
        np.savez_compressed(path, **detect_hands(gesture_path(cfg["data"]["raw_dir"], manner),
                                                 Path(cfg["data"]["model_path"]).with_name("hand_landmarker.task")))
    h = dict(np.load(path))
    h["image"] = track(h["image"])
    return h


def track(img: np.ndarray) -> np.ndarray:
    """Keep each hand in its slot: every detection goes to the slot whose last position (mean landmark) is nearest,
    slots start in image-x order, and frames where a hand was lost are filled by linear interpolation between its
    neighbouring detections (held at the ends). A hand detected in under a fifth of the frames is dropped (NaN)."""
    F = len(img)
    out = np.full_like(img, np.nan)
    last = [None, None]
    for f in range(F):
        dets = [img[f, i] for i in range(2) if not np.isnan(img[f, i, 0, 0])]
        if not dets:
            continue
        cents = [d.mean(axis=0) for d in dets]
        if last[0] is None and last[1] is None:
            order = np.argsort([c[0] for c in cents])
            slots = list(order) if len(dets) == 2 else [0]
            for s, i in zip(range(2), slots):
                out[f, s], last[s] = dets[i], cents[i]
            continue
        dist = lambda s, c: np.inf if last[s] is None else np.linalg.norm(last[s] - c)
        if len(dets) == 2:
            keep = dist(0, cents[0]) + dist(1, cents[1]) <= dist(0, cents[1]) + dist(1, cents[0])
            pairs = [(0, 0), (1, 1)] if keep else [(0, 1), (1, 0)]
        else:
            seen = [s for s in range(2) if last[s] is not None]
            hand_size = np.ptp(dets[0], axis=0).max()
            if len(seen) == 1 and dist(seen[0], cents[0]) > hand_size:     # far from the one hand seen: the other
                s = 1 - seen[0]
            else:
                s = int(np.argmin([dist(0, cents[0]), dist(1, cents[0])]))
            pairs = [(s, 0)]
        for s, i in pairs:
            out[f, s], last[s] = dets[i], cents[i]
    t = np.arange(F)
    for s in range(2):
        ok = ~np.isnan(out[:, s, 0, 0])
        if ok.mean() < 0.2:                       # seen in under a fifth of the frames: a stray detection
            out[:, s] = np.nan
        elif ok.any():
            flat = out[:, s].reshape(F, -1)
            for k in range(flat.shape[1]):
                flat[~ok, k] = np.interp(t[~ok], t[ok], flat[ok, k])
            out[:, s] = flat.reshape(F, 21, 2)
    return out


def active(img: np.ndarray, still: float = 0.25):
    """The gesture proper: the frames from the first to the last in which the hands move faster than ``still`` x
    their 95th-percentile speed (smoothed over 5 frames), so the rests before and after are dropped. A hand that
    barely moves (a path under a tenth of the other's) is kept only near the moving one (median positions within
    two hand sizes, e.g. a forearm held as the ground); resting elsewhere, e.g. in the lap, it is dropped (NaN).
    Returns the kept frames and the index of the first."""
    c = np.nanmean(img, axis=2)                                              # (F, 2, 2) hand centres
    speed = np.nan_to_num(np.linalg.norm(np.diff(c, axis=0), axis=-1)).sum(axis=1)
    speed = np.convolve(speed, np.ones(5) / 5, mode="same")
    moving = np.flatnonzero(speed > still * np.percentile(speed, 95))
    a, b = moving[0], moving[-1] + 2
    img = img[a:b].copy()
    path = [np.nansum(np.linalg.norm(np.diff(np.nanmean(img[:, s], axis=1), axis=0), axis=-1)) for s in range(2)]
    main, other = int(np.argmax(path)), 1 - int(np.argmax(path))
    if path[other] < 0.1 * path[main]:                                      # a hand that barely moves
        size = np.nanmedian(np.ptp(img[:, main], axis=1).max(axis=-1))
        here = lambda s: np.nanmedian(np.nanmean(img[:, s], axis=1), axis=0)
        if np.linalg.norm(here(other) - here(main)) > 2 * size:
            img[:, other] = np.nan
    return img, a


def gesture_boards(cfg: dict, manner: str, edge_samples: int = None):
    """(T, 6, H, W) binary channel boards of one gesture, and the source frame of each board frame."""
    from grace_lot.grace.temporal import sample_window
    pre, itf = cfg["preprocess"], cfg["interface"]
    k = itf["edge_samples"] if edge_samples is None else edge_samples
    img, a0 = active(load_hands(cfg, manner)["image"])
    filled = np.where(np.isnan(img), 0.0, img)                              # a dropped hand: constant, for the period
    a, b = sample_window(filled.reshape(len(filled), 42, 2), pre.get("sampling", "segment"))
    T = pre["frames"]
    idx = np.round(np.linspace(a, b - 1, T)).astype(int)
    pts = img[idx]                                                           # (T, 2, 21, 2), NaN where absent
    lo, hi = np.nanmin(pts.reshape(-1, 2), axis=0), np.nanmax(pts.reshape(-1, 2), axis=0)
    scale = (hi - lo).max()
    (x0, x1), (y0, y1) = pre["window_x"], pre["window_y"]
    centre = (lo + hi) / 2
    # canonical units: one body height = the hands' larger extent; board x in [0, 1) over window_x, y down
    u = (pts[..., 0] - centre[0]) / scale
    v = (pts[..., 1] - centre[1]) / scale + (y0 + y1) / 2                  # a body's middle sits at the window's
    bx, by = (u - x0) / (x1 - x0), (v - y0) / (y1 - y0)
    n = pre["field_size"]
    out = np.zeros((T, 6, n, n))
    s = np.linspace(0, 1, k + 2)
    for hand in range(2):
        for c, edges in enumerate(CHANNELS.values()):
            for i, j in edges:
                for f in s:
                    px = bx[:, hand, i] * (1 - f) + bx[:, hand, j] * f
                    py = by[:, hand, i] * (1 - f) + by[:, hand, j] * f
                    ok = ~np.isnan(px)
                    t = np.arange(T)[ok]
                    out[t, 3 * hand + c, np.clip((py[ok] * n).astype(int), 0, n - 1),
                        np.clip((px[ok] * n).astype(int), 0, n - 1)] = 1.0
    return out, a0 + idx


def gesture_features(cfg: dict, manner: str) -> np.ndarray:
    """(T, 6, F) detector features of a gesture's channels, as the locomotion parts' (``scan_part_cells``)."""
    from grace_lot.grace.scanner import field_features
    itf = cfg["interface"]
    return field_features(gesture_boards(cfg, manner)[0], itf["scan_softness"], itf["lot_salience"],
                          itf.get("features", "max6"))


def main() -> None:
    from grace_lot.config import load_config
    from grace_lot.grace.dataset import split_videos
    cfg = load_config("configs/grace_v7_parts.toml")
    for m in sorted(split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")):
        img = load_hands(cfg, m)["image"]
        n = (~np.isnan(img[..., 0, 0])).sum(axis=1)
        print(f"{m:14s} frames {len(img):4d}  no hand {np.mean(n == 0):.2f}  one {np.mean(n == 1):.2f}  "
              f"two {np.mean(n == 2):.2f}", flush=True)


if __name__ == "__main__":
    main()
