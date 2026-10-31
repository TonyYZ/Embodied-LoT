"""A gallery of fitted programs beside the videos they explain. Diagnostic only; training actors only.

    python -m grace_lot.analysis.gallery --run results/grace_v7_parts/w32_l1.0_current_cap32 [--top 3] [--candidates 150]

For each manner of a run (Gaussian scan likelihood), the ``candidates`` most frequent posterior programs and the
MAP are scored on the manner's training videos (log prior + log likelihood, the run's own settings), and the best
``top`` programs with distinct renderings are shown, with their trees, for the medoid training actor:

    the video (at the frames the model sees) | its body parts on the board | the top program's outputs,
    each tinted by the part it is matched to

Writes ``<run>/gallery/index.html``, its GIFs in ``<run>/gallery/gifs/`` and ``<run>/gallery/gallery.json``
(programs, scores and trees).
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

from grace_lot.analysis.inspect_parts import COLORS, colour, figure
from grace_lot.analysis.inspect_programs import tree_svg
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.parts import PART_NAMES
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.grace.scanner import part_boards
from grace_lot.grace.temporal import sample_window
from grace_lot.lot_adapter.adapter import rollout
from grace_lot.lot_adapter.program import parse_program, saved_lot, size

CELL = 16                                  # pixels per board cell (12 x 12; a board is 12 x CELL pixels tall)


def sampled_frames(traj, frames: int) -> np.ndarray:
    """The source video frames the model's frames come from (the sampled window, resampled)."""
    a, b = sample_window(traj.joints_full, "cycle2")
    return traj.segment[0] + a + np.round(np.linspace(0, b - a - 1, frames)).astype(int)


def video_frames(path: Path, idx, height: int, centres=None, width: float = 0.8) -> list:
    """Frames ``idx`` of a video as RGB arrays ``height`` tall; with ``centres`` (pixel x per frame), each is cropped
    to ``width`` x its height around the centre, following a walker across the room."""
    import cv2
    cap, out = cv2.VideoCapture(str(path)), []
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if centres is not None:
            w = int(width * f.shape[0])                                      # padded, so the walker stays centred
            x0 = int(centres[k])
            f = cv2.copyMakeBorder(f, 0, 0, w, w, cv2.BORDER_REPLICATE)[:, x0 + w - w // 2:x0 + w - w // 2 + w]
        out.append(cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height), interpolation=cv2.INTER_AREA))
    return out


def walker_centres(cfg: dict, video, idx) -> np.ndarray:
    """The walker's pixel x at frames ``idx`` (mean of the detected landmarks, from the pose cache)."""
    from grace_lot.grace.pose import PoseSequence
    from grace_lot.grace.preprocess import processed_paths
    x = np.nanmean(PoseSequence.load(processed_paths(cfg["data"]["processed_dir"], video)["pose"]).image[..., 0], axis=1)
    ok = ~np.isnan(x)
    return np.interp(idx, np.flatnonzero(ok), x[ok])


def program_board(y: np.ndarray, labels: list) -> np.ndarray:
    """One frame of a program's outputs: figure tinted by the matched part, cells no output claims (0.5 in every
    output) dark grey, cells claimed empty black."""
    img = colour(figure(y), labels)
    unclaimed = np.all(np.abs(y - 0.5) < 1e-6, axis=0)
    img[unclaimed] = np.maximum(img[unclaimed], 0.18)
    return img


def cell_px(pre: dict) -> tuple:
    """(height, width) of a board cell in pixels: boards are 12 x CELL pixels tall, cells as wide as they are in
    body heights (a world-frame board's columns span twice what its rows do)."""
    (x0, x1), (y0, y1) = pre["window_x"], pre["window_y"]
    h = 12 * CELL // pre["field_size"]
    return h, round(h * (x1 - x0) / (y1 - y0))


def board_px(img: np.ndarray, cell: tuple = (CELL, CELL)) -> np.ndarray:
    return (np.kron(img, np.ones(cell + (1,))) * 255).astype(np.uint8)


def gif(panels: list, path: Path) -> None:
    """Frames of side-by-side panels (each a list of (h, w, 3) uint8 images), written as an animated GIF."""
    from PIL import Image
    ims = []
    for row in zip(*panels):
        h = max(p.shape[0] for p in row)
        gap = np.full((h, 6, 3), 255, np.uint8)
        parts = []
        for p in row:
            pad = np.full((h - p.shape[0], p.shape[1], 3), 255, np.uint8)
            parts += [np.concatenate([p, pad]), gap]
        ims.append(Image.fromarray(np.concatenate(parts[:-1], axis=1)))
    ims[0].save(path, format="GIF", save_all=True, append_images=ims[1:], duration=200, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=150)
    ap.add_argument("--manners", nargs="*")
    args = ap.parse_args()
    from grace_lot.inference.run import build, observations_for
    from grace_lot.lot_adapter.program import apply_semantics
    run = Path(args.run)
    cfg = load_config(args.config)
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    cfg["lot"] = saved_lot(rc["lot"])
    apply_semantics(cfg["lot"])
    pre, itf = cfg["preprocess"], cfg["interface"]
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    sealed = set(cfg["split"]["test_actors"])
    out_dir = run / "gallery"
    (out_dir / "gifs").mkdir(parents=True, exist_ok=True)
    manners = args.manners or sorted(f.stem for f in (run / "posteriors").glob("*.json"))
    sections, record = [], {}
    for manner in manners:
        videos = train[manner]
        assert not any(v.actor in sealed for v in videos)
        post = json.loads((run / "posteriors" / f"{manner}.json").read_text())
        mass = Counter()
        for s, w in post["posterior"]:
            mass[s] += w
        cands = [s for s, _ in mass.most_common(args.candidates)]
        if post["map"] not in cands:
            cands.append(post["map"])
        obs = observations_for(cfg, videos)
        _, renderer, prior, lik, _ = build(cfg, obs)
        scored = sorted(((prior(p) + lik(p), prior(p), p) for p in map(parse_program, cands)), key=lambda x: -x[0])
        top, seen = [], set()
        for lp, lpr, p in scored:
            key = renderer(p).tobytes()
            if key not in seen:
                seen.add(key); top.append((lp, lpr, p))
            if len(top) == args.top:
                break
        # the medoid training actor: closest to the manner's mean features
        O = np.stack([o[0] if o.ndim == 4 else o for o in obs])                       # phase 0 of per-phase copies
        med = int(np.argmin(((O - O.mean(0)) ** 2).mean(axis=(1, 2, 3))))
        traj = load_trajectory(cfg, videos[med])
        boards = part_boards(traj, pre, itf["edge_samples"])
        lp, lpr, best = top[0]
        warmup = getattr(lik, "warmup", 0)
        y = rollout(best, lik.frames + warmup, lik.size)[warmup:] if warmup else renderer(best)     # the scored frames
        y = y if y.ndim == 4 else y[:, None]
        rows, cols, _ = lik.assignments(best)[med]
        label = {int(k): PART_NAMES[int(c)] for k, c in zip(rows, cols) if k < y.shape[1]}
        T = len(boards)
        idx = sampled_frames(traj, T)
        anim = f"gifs/{manner}.gif"
        cell, centres = cell_px(pre), walker_centres(cfg, videos[med], idx)
        if pre.get("frame", "body") == "world":                     # the board stays put; so does the crop
            centres = np.full_like(centres, centres.mean())
        aspect = (pre["window_x"][1] - pre["window_x"][0]) / (pre["window_y"][1] - pre["window_y"][0])
        gif([video_frames(videos[med].path, idx, 12 * CELL, centres, 0.8 * aspect),
             [board_px(colour(boards[t], PART_NAMES), cell) for t in range(T)],
             [board_px(program_board(y[t], [label.get(k) for k in range(y.shape[1])]), cell) for t in range(T)]],
            out_dir / anim)
        cards = []
        for r, (lp_, lpr_, p) in enumerate(top, 1):
            cards.append(f"<div class=prog><h4>#{r} &nbsp; log posterior {lp_:.1f} &nbsp;(prior {lpr_:.1f}, "
                         f"{size(p.tree)} nodes, {mass.get(str(p), 0) * 100:.2f}% of samples)</h4>"
                         f"<pre>{html.escape(str(p))}</pre><div class=tree>{tree_svg(p)}</div></div>")
        mapping = ", ".join(f"out{k} &rarr; {lab}" for k, lab in sorted(label.items())) or "none"
        sections.append(
            f"<section id={manner}><h2>{manner}</h2><p class=meta>{len(videos)} training actors; shown: "
            f"{videos[med].stem} (medoid). {post['unique_programs']} distinct programs sampled; R-hat "
            f"{post['r_hat_log_post']:.2f}.</p><figure><img src='{anim}' alt='{manner}: video, parts, program'>"
            f"<figcaption>video &nbsp;|&nbsp; body parts on the board &nbsp;|&nbsp; best program's outputs "
            f"(matching: {mapping})</figcaption></figure>{''.join(cards)}</section>")
        record[manner] = {"medoid": videos[med].stem, "mapping": label, "gif": anim,
                          "unique_programs": post["unique_programs"], "r_hat": post["r_hat_log_post"],
                          "top": [{"program": str(p), "log_post": lp_, "log_prior": lpr_,
                                   "sample_share": mass.get(str(p), 0), "nodes": size(p.tree), "svg": tree_svg(p)}
                                  for lp_, lpr_, p in top]}
        print(f"{manner}: best {top[0][0]:.1f}  {top[0][2]}", flush=True)
    legend = " ".join(f"<span style='color:rgb{tuple(int(255 * x) for x in COLORS[p])}'>&#9632;</span> {p.replace('_', ' ')}"
                      for p in PART_NAMES)
    css = ("body{font:15px/1.5 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1100px;color:#1d2330;"
           "background:#fbfaf7}h2{margin:0 0 4px;text-transform:capitalize}section{border-top:1px solid #ddd;"
           "padding:20px 0}.meta,figcaption{color:#5b6170;font-size:13px}img{max-width:100%;image-rendering:pixelated}"
           "pre{white-space:pre-wrap;background:#f1efe9;padding:8px;font-size:12px;border-radius:4px}"
           ".tree{overflow-x:auto}.prog h4{margin:14px 0 4px;font-weight:600;font-size:14px}")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Fitted programs</title><style>{css}</style>"
            f"</head><body><h1>Fitted programs and the videos they explain</h1><p>Run <code>{html.escape(run.name)}</code>"
            f", fitted to the training actors only. Part colours: {legend}; purple: an output matched to no part. In the program panel, dark grey cells are ones the program makes no claim about (they are predicted from real bodies), black cells ones it claims empty.</p>"
            + "".join(sections) + "</body></html>")
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / "gallery.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
