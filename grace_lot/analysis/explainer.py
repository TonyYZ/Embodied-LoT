"""Figures for a step-by-step page: one video from raw frames to limb-motion descriptors, and a program from its
tree to the same descriptors. Training actors only.

    python -m grace_lot.analysis.explainer --run <descriptor run> --manner jumping --out results/explainer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grace_lot.analysis.gallery import sampled_frames, video_frames
from grace_lot.analysis.inspect_parts import COLORS
from grace_lot.analysis.inspect_programs import tree_svg
from grace_lot.config import load_config
from grace_lot.grace import descriptors as D
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.parts import PART_NAMES, part_connections
from grace_lot.grace.pose import PoseSequence
from grace_lot.grace.preprocess import load_trajectory, processed_paths
from grace_lot.grace.scanner import part_boards
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot

INK, MUTED = "#22252b", "#7a7f88"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
PART_RGB = {p: COLORS[p] if p != "torso" else (0.55, 0.55, 0.6) for p in PART_NAMES}   # torso visible on white


def tinted(board: np.ndarray, rgb) -> np.ndarray:
    """A board as an image: claimed figure in the part's colour, unknown grey, empty dark."""
    dark = np.full(board.shape + (3,), 0.12)
    return np.where(board[..., None] >= 0.75, np.array(rgb), np.where(board[..., None] > 0.25, 0.35, dark))


def frame_height(video) -> int:
    import cv2
    return int(cv2.VideoCapture(str(video.path)).get(cv2.CAP_PROP_FRAME_HEIGHT))


def board_grid(rows, labels, path: Path) -> None:
    """rows: list of (16 boards (H, W), rgb); one row per board sequence."""
    fig, axes = plt.subplots(len(rows), 16, figsize=(12, 0.8 * len(rows) + 0.3), squeeze=False)
    for i, (boards, rgb) in enumerate(rows):
        for t in range(16):
            ax = axes[i, t]
            ax.imshow(tinted(boards[t], rgb), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(str(t + 1), fontsize=7, color=MUTED)
        axes[i, 0].set_ylabel(labels[i], rotation=0, ha="right", va="center", fontsize=8)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manner", default="jumping")
    ap.add_argument("--compare", nargs="*", default=["swinging", "scurrying"])
    ap.add_argument("--out", default="results/explainer")
    args = ap.parse_args()
    from grace_lot.inference.run import build, observations_for
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    run, out, m = Path(args.run), Path(args.out), args.manner
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/grace_v7_parts.toml")
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    cfg["lot"] = saved_lot(rc["lot"])
    apply_semantics(cfg["lot"])
    pre, es = cfg["preprocess"], cfg["interface"]["edge_samples"]
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    video = train[m][0]
    traj = load_trajectory(cfg, video)
    idx = sampled_frames(traj, pre["frames"])
    conns = part_connections(traj.connections)
    info = {"video": video.stem, "frames": [int(i) for i in idx]}

    # 1. raw frames with the detected landmarks
    pose = PoseSequence.load(processed_paths(cfg["data"]["processed_dir"], video)["pose"])
    frames = video_frames(video.path, idx, 360)
    scale = 360 / frame_height(video)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
    for ax, t in zip(axes, (0, 2, 4, 6)):
        ax.imshow(frames[t]); ax.axis("off")
        xy = pose.image[idx[t]] * scale
        for part, edges in conns.items():
            for a, b in edges:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], "-", color=PART_RGB[part], lw=1.6)
        ax.set_title(f"model frame {t + 1} (video frame {idx[t]})", color=MUTED, fontsize=8)
    fig.savefig(out / "1_landmarks.png", dpi=110, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 2. the body frame: 16 resampled frames over two cycles
    j = traj.joints
    fig, axes = plt.subplots(1, 16, figsize=(12, 2.2), sharey=True)
    for t, ax in enumerate(axes):
        for part, edges in conns.items():
            for a, b in edges:
                ax.plot(j[t, [a, b], 0], j[t, [a, b], 1], "-", color=PART_RGB[part], lw=1.4)
        ax.set_xlim(*pre["window_x"]); ax.set_ylim(pre["window_y"][1], pre["window_y"][0])
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(str(t + 1), fontsize=7, color=MUTED)
    fig.savefig(out / "2_body_frame.png", dpi=110, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 3. six part boards per frame (what the model sees)
    B = part_boards(traj, pre, es)                                         # (T, P, H, W)
    board_grid([(B[:, p], PART_RGB[part]) for p, part in enumerate(PART_NAMES)],
               [p.replace("_", " ") for p in PART_NAMES], out / "3_part_boards.png")

    # 4. signals and their cycle harmonics
    sig = D.board_signals(B)                                               # (T, P, 4)
    t = np.arange(16)
    Z = np.fft.rfft(sig, axis=0) / 16
    wave = lambda z, k: 2 * np.real(z * np.exp(2j * np.pi * k * t / 16))
    li, ri, to = (PART_NAMES.index(p) for p in ("left_leg", "right_leg", "torso"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for p in (li, ri):
        x = sig[:, p, 0] * 12
        col = PART_RGB[PART_NAMES[p]]
        axes[0].plot(t + 1, x, "o", color=col, ms=4, label=PART_NAMES[p].replace("_", " "))
        axes[0].plot(t + 1, x.mean() + 12 * wave(Z[2, p, 0], 2), "-", color=col, lw=1.3)
    axes[0].set_title("leg centroids, x (cells), and their once-per-cycle wave", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    y = sig[:, to, 1] * 12
    axes[1].plot(t + 1, y, "o", color=PART_RGB["torso"], ms=4)
    axes[1].plot(t + 1, y.mean() + 12 * (wave(Z[2, to, 1], 2) + wave(Z[4, to, 1], 4)), "-",
                 color=PART_RGB["torso"], lw=1.3)
    axes[1].invert_yaxis(); axes[1].set_title("torso centroid, y (cells): the bounce", fontsize=9)
    for ax in axes[:2]:
        ax.set_xlabel("model frame"); ax.axvline(8.5, color=MUTED, lw=0.8, ls=":")
    desc = D.descriptors(sig)
    amps = desc[:36].reshape(2, 3, 6)                                      # harmonic, signal, part
    w = 0.38
    cols = [PART_RGB[p] for p in PART_NAMES]
    axes[2].bar(np.arange(6) - w / 2, amps[0, 0] * 12, w, color=cols)
    axes[2].bar(np.arange(6) + w / 2, amps[1, 0] * 12, w, color=cols, alpha=0.45)
    axes[2].set_xticks(range(6), [p.replace("_", "\n") for p in PART_NAMES], fontsize=7)
    axes[2].set_title("x amplitude (cells): once (solid), twice (pale)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "4_signals.png", dpi=110, bbox_inches="tight", facecolor="white"); plt.close(fig)
    info["coupling"] = dict(zip(D.names()[36:], np.round(desc[36:], 2).tolist()))
    info["n_descriptors"] = len(desc)

    # 5. fingerprints: descriptors of every training actor, three manners
    manners = [m] + args.compare
    rows = {mm: np.array([D.descriptors(D.board_signals(part_boards(load_trajectory(cfg, v), pre, es)))
                          for v in train[mm]]) for mm in manners}
    allx = np.concatenate(list(rows.values()))
    zs = lambda x: (x - allx.mean(0)) / (allx.std(0) + 1e-9)
    fig, axes = plt.subplots(1, len(manners), figsize=(11, 6), sharey=True)
    for ax, mm in zip(axes, manners):
        ax.imshow(zs(rows[mm]).T, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
        ax.set_title(f"{mm}: 15 actors", fontsize=9); ax.set_xticks([])
    axes[0].set_yticks(range(len(D.names())), D.names(), fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out / "5_fingerprints.png", dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # 6. a program: its outputs, matched to parts, and its descriptors beside the actors'
    lik = build(cfg, observations_for(cfg, train[m]))[3]
    post = json.loads((run / "posteriors" / f"{m}.json").read_text())
    prog = parse_program(post["map"])
    states = motion_combinations(prog, pre["field_size"])[0][0]
    v = rollout(prog, pre["frames"], pre["field_size"], states)
    v = v if v.ndim == 4 else v[:, None]
    r, c, _ = lik.assignments(prog)[0]
    label = {int(k): PART_NAMES[int(p)] for k, p in zip(r, c) if k < v.shape[1]}
    K = v.shape[1]
    board_grid([(v[:, k], PART_RGB.get(label.get(k), (0.75, 0.3, 0.9))) for k in range(K)],
               [f"out{k} → {label.get(k, 'none').replace('_', ' ')}" for k in range(K)],
               out / "6_program_outputs.png")
    pa, have = np.zeros((6, 6)), np.zeros(6, bool)
    for k, part in label.items():
        a, _ = lik._channel(np.ascontiguousarray(v[:, k]))
        pa[PART_NAMES.index(part)] = a.mean(0)
        have[PART_NAMES.index(part)] = True
    va = lik.amps                                                          # (N, P, A)
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9))
    for ax, h, name in ((axes[0], 0, "once per cycle"), (axes[1], 1, "twice per cycle")):
        for s, (sn, mk) in enumerate(zip(D.SIGNALS, "o^s")):
            col = h * 3 + s
            xs = np.arange(6) + (s - 1) * 0.22
            ax.scatter(np.repeat(xs, va.shape[0]), (va[:, :, col].T * 12).ravel(), s=6, color=MUTED, alpha=0.4)
            ax.scatter(xs[have], pa[have, col] * 12, s=34, marker=mk, edgecolor=INK, linewidth=0.6,
                       color=[PART_RGB[PART_NAMES[i]] for i in np.flatnonzero(have)], label=sn)
        ax.set_xticks(range(6), [p.replace("_", "\n") for p in PART_NAMES], fontsize=7)
        ax.set_title(f"amplitude {name}: program (marks), 15 actors (grey)", fontsize=9)
        ax.set_ylabel("cells (axis: x 12)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7, title="signal", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "7_program_vs_actors.png", dpi=110, bbox_inches="tight", facecolor="white"); plt.close(fig)
    (out / "tree.svg").write_text(tree_svg(prog), encoding="utf-8")
    info.update(program=post["map"], mapping={str(k): p for k, p in label.items()}, n_actors=int(lik.N))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
