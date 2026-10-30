"""What the motion scoring compares, per body part: a fitted program against the actors. Diagnostic only; training
actors only.

    python -m grace_lot.analysis.motion_view --run results/grace_v7_parts/desc_fix_5

For each manner, the best program of the run's gallery (``gallery.py``) and the training actor closest to the
manner's mean motion (the medoid), under the program's most likely motion state for that actor:

  <manner>.gif   the video | rows of boards, one per body part: the actor's part; the matched program output (figure
                 tinted by the part; grey: no claim; black: claimed empty); one fill of it, as the scoring sees it
                 (cells the program makes no claim about filled from a real body part's footage)
  <manner>.png   per part, over the 16 frames: the centre's left-right position and height (up is up) and the tilt,
                 each about its own average, for the actor, for what the program itself draws (its claimed figure
                 only; gaps: frames it draws nothing) and for one fill (dashed); the row's label says what share of
                 the board the output claims. Then the numbers the scoring compares: the training actors' mean and
                 spread, the scoring's prediction (over the fills), and what the program's own drawing alone gives;
                 last, the four limb pairs' couplings. Much of a program's scored motion can come from the fills.
The program's traces are shifted circularly, all by the same amount (one program, one clock), to line up with the
actor's: phase is not scored, so this is for display only. Writes ``<run>/motion_view/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from grace_lot.analysis.gallery import CELL, board_px, cell_px, gif, program_board, sampled_frames, video_frames, \
    walker_centres
from grace_lot.analysis.inspect_parts import COLORS, colour
from grace_lot.config import load_config
from grace_lot.grace import descriptors as D
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.parts import PART_NAMES
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.inference.descriptor_likelihood import FULL_HARMONICS, PAIRS, _amplitudes_and_once, _coupling
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot

RHYTHM = {1: "per clip", 2: "once", 3: "1.5x", 4: "twice"}         # harmonic of the 16 frames (two cycles)


def compared(lik, S: int):
    """The names of the numbers the scoring compares per part, and their display scale (lengths in cells)."""
    ks = FULL_HARMONICS if lik.full else tuple(D.HARMONICS.values())
    names = [f"{s} {RHYTHM[k]}" for k in ks for s in ("x", "y", "tilt")]
    scale = [S if s != "axis" else 1.0 for _ in ks for s in D.SIGNALS]
    if lik.full:
        names += ["mean x", "mean y", "size", "axis c", "axis s"]
        scale += [S, S, S * S, 1.0, 1.0]                                   # size in cells
    return names, np.array(scale)


def tilt_deg(sig: np.ndarray) -> np.ndarray:
    """Axis (e cos 2t, e sin 2t) -> t in degrees, unwrapped over the frames."""
    return np.degrees(np.unwrap(np.arctan2(sig[..., 3], sig[..., 2]), axis=0) / 2)


def traces(sig: np.ndarray, size: int) -> np.ndarray:
    """(T, ..., 4) signals -> (T, ..., 3): x (right) and height (up) in cells and tilt in degrees, each about its own
    mean. Board rows count down, so height is minus the centre's row; tilt is clockwise."""
    t = np.stack([sig[..., 0] * size, -sig[..., 1] * size, tilt_deg(sig)], axis=-1)
    return t - t.mean(axis=0, keepdims=True)


def own_signals(c: np.ndarray) -> np.ndarray:
    """(T, H, W) output -> (T, 4) signals of its own figure (cells it claims as more likely figure); NaN where it draws
    nothing."""
    fig = (c > 0.5).astype(float)
    sig = D.board_signals(fig[:, None], 0.5)[:, 0]
    sig[~fig.any(axis=(1, 2))] = np.nan
    return sig


def own_traces(sig: np.ndarray, size: int) -> np.ndarray:
    """``traces`` for a (T, 4) signal with NaN frames: (T, 3), NaN where the program draws nothing."""
    ok = ~np.isnan(sig[:, 0])
    out = np.full((len(sig), 3), np.nan)
    if ok.any():
        out[ok] = traces(sig[ok], size)
    return out


def fill_board(lik, v: np.ndarray, k: int, t: int, d: int = 0) -> np.ndarray:
    """Output k at frame t in fill draw d, as the scoring sees it: claimed cells as claimed, the rest a real part's."""
    c = v[:, k]
    if np.all((c == 0) | (c == 1)):
        return c[t].astype(float)
    fills, uniforms = lik._draws(*c.shape, k)
    return np.where(uniforms[d] < np.abs(2 * c[t] - 1), c[t] > 0.5, fills[t, d] >= lik.salience).astype(float)


def program_signals(lik, v: np.ndarray, k: int) -> np.ndarray:
    """Output k's signals over the fill draws, (T, draws, 4), drawn exactly as the likelihood draws them."""
    c = np.ascontiguousarray(v[:, k])
    if np.all((c == 0) | (c == 1)):
        boards = c[:, None]
    else:
        fills, uniforms = lik._draws(*c.shape, k)
        boards = np.where(uniforms[None] < np.abs(2 * c - 1)[:, None], (c > 0.5)[:, None], fills)
    return D.board_signals(boards.astype(float), lik.salience)


def label_strip(labels: list, width: int, height: int = 18) -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width * len(labels) + 6 * (len(labels) - 1), height), "white")
    d = ImageDraw.Draw(im)
    for i, s in enumerate(labels):
        d.text((i * (width + 6) + 2, 3), s, fill=(40, 40, 40))
    return np.array(im)


def grid(cols: list) -> np.ndarray:
    gap = lambda h: np.full((h, 6, 3), 255, np.uint8)
    out = []
    for c in cols:
        out += [c, gap(c.shape[0])]
    return np.concatenate(out[:-1], axis=1)


def figure_png(manner, tr_actor, tr_prog, tr_fill, claimed, amps, pred, own, coup, path: Path, AMPS: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = len(PART_NAMES)
    wide = len(AMPS) > 6                         # the full set: four rhythms and the posture
    fig, ax = plt.subplots(P + 1, 4, figsize=(19 if wide else 15, 2.1 * (P + 1)),
                           gridspec_kw=dict(width_ratios=[1, 1, 1, 2.6 if wide else 1.3]))
    t = np.arange(tr_actor.shape[0])
    names = ["left-right (cells, right +)", "height (cells, up +)", "tilt (degrees, clockwise +)"]
    for q, part in enumerate(PART_NAMES):
        col = COLORS[part] if part != "torso" else (0.45, 0.45, 0.45)
        for s in range(3):
            a = ax[q, s]
            a.plot(t, tr_actor[:, q, s], color="black", lw=1.6, label="actor")
            if tr_prog[q] is not None:
                a.plot(t, tr_prog[q][:, s], color=col, lw=1.8, marker=".", label="program's own drawing")
                a.plot(t, tr_fill[q][:, s], color=col, lw=1.0, ls="--", alpha=0.7, label="one fill (as scored)")
            a.axhline(0, color="0.85", lw=0.8, zorder=0)
            if q == 0:
                a.set_title(names[s], fontsize=10)
            if s == 0:
                a.set_ylabel(part.replace("_", " ") + ("\n(background)" if claimed[q] is None
                                                       else f"\nclaims {claimed[q]:.0%}"), fontsize=9)
            a.tick_params(labelsize=7)
        a = ax[q, 3]
        x = np.arange(len(AMPS))
        m, sd = amps[q]
        pm, ps = pred[q]
        a.bar(x - 0.27, m, 0.27, yerr=sd, color="0.3", label="actors (mean, spread)", error_kw=dict(lw=0.8))
        a.bar(x, pm, 0.27, yerr=ps, color=col, label="scored prediction (with fills)", error_kw=dict(lw=0.8))
        if own[q] is not None:
            a.bar(x + 0.27, own[q], 0.27, color="white", edgecolor=col, hatch="///", label="program's own drawing")
        a.set_xticks(x, AMPS if q == P - 1 else [""] * len(AMPS), fontsize=7, rotation=30)
        a.tick_params(labelsize=7)
        if q == 0:
            a.set_title("numbers compared (lengths in cells)", fontsize=10)
            a.legend(fontsize=7, loc="upper right")
    ax[0, 0].legend(fontsize=7, loc="upper right")
    for s in range(3):
        ax[P, s].axis("off")
    a = ax[P, 3]
    labels = [f"{x.replace('_', ' ')} / {y.replace('_', ' ')} {d}" for x, y in D.PAIRS for d in ("x", "y")]
    x = np.arange(len(labels))
    (m, sd), pm = coup
    a.bar(x - 0.2, m, 0.4, yerr=sd, color="0.3", error_kw=dict(lw=0.8))
    a.bar(x + 0.2, np.nan_to_num(pm), 0.4, color="#7a5cff")
    a.axhline(0, color="0.6", lw=0.8)
    a.set_ylim(-1.1, 1.1)
    a.set_xticks(x, labels, fontsize=6, rotation=35, ha="right")
    a.set_title("limb pairs: +1 in step, -1 opposite (grey actors, purple program)", fontsize=9)
    a.tick_params(labelsize=7)
    fig.suptitle(f"{manner}: what the motion scoring compares, part by part", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/grace_v7_parts.toml")
    args = ap.parse_args()
    from grace_lot.inference.run import build, observations_for
    from grace_lot.lot_adapter.adapter import motion_combinations, rollout
    run = Path(args.run)
    cfg = load_config(args.config)
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "inference", "split", "interface"):
        cfg[k] = rc.get(k, cfg[k])
    cfg["lot"] = saved_lot(rc["lot"])
    apply_semantics(cfg["lot"])
    pre = cfg["preprocess"]
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    assert not any(v.actor in set(cfg["split"]["test_actors"]) for vs in train.values() for v in vs)
    out = run / "motion_view"
    out.mkdir(parents=True, exist_ok=True)
    gallery = json.loads((run / "gallery" / "gallery.json").read_text())
    for manner, g in gallery.items():
        videos = train[manner]
        obs = observations_for(cfg, videos)
        lik = build(cfg, obs)[3]
        S = lik.size
        med = int(np.argmin(((lik.amps - lik.amps.mean(0)) ** 2 / lik.s2).sum(axis=(1, 2))))
        prog = parse_program(g["top"][0]["program"])
        L, matched = lik._scores(prog)
        b = int(L[:, med].argmax())
        states = list(motion_combinations(prog, S))[b][0]
        v = rollout(prog, lik.frames, S, states)
        v = v if v.ndim == 4 else v[:, None]
        rows, cols, _ = matched[b][med]
        out_of = {int(c): int(r) for r, c in zip(rows, cols) if r < v.shape[1]}       # part -> its output
        actor = traces(D.board_signals(obs[med].astype(float), lik.salience), S)       # (T, P, 3)
        tr_prog, tr_fill, claimed, pred, own, sigs = [], [], [], [], [], {}
        AMPS, scale = compared(lik, S)
        for q in range(lik.P):
            k = out_of.get(q)
            if k is None:
                tr_prog.append(None); tr_fill.append(None); claimed.append(None); own.append(None)
                pred.append((lik.bg["amu"] * scale, np.sqrt(lik.bg["avar"] + lik.s2[q]) * scale))
                continue
            c = np.ascontiguousarray(v[:, k])
            sig = program_signals(lik, v, k)                                          # (T, draws, 4)
            sigs[q] = sig
            tr_fill.append(traces(sig[:, 0], S))                                      # one fill, as scored
            o = own_signals(c)
            tr_prog.append(own_traces(o, S))
            claimed.append(float((np.abs(2 * c - 1) > 0.5).mean()))
            filled = np.where(np.isnan(o), np.nanmean(o, axis=0) if (~np.isnan(o)).any() else 0.0, o)
            mass = (c > 0.5).mean(axis=(1, 2))[:, None] if lik.full else None
            own.append(_amplitudes_and_once(filled[:, None], mass)[0][0] * scale)
            a, _ = lik._channel(c, k)
            pred.append((a.mean(0) * scale, np.sqrt(a.var(0) + lik.s2[q]) * scale))
        shown = [q for q in range(lik.P) if tr_prog[q] is not None]   # one shift for the whole program: one clock
        shift = int(np.argmin([sum(np.nansum(((np.roll(tr_fill[q], s, 0) - actor[:, q]) ** 2)[:, :2]) for q in shown)
                               for s in range(lik.frames)]))
        tr_prog = [None if x is None else np.roll(x, shift, 0) for x in tr_prog]
        tr_fill = [None if x is None else np.roll(x, shift, 0) for x in tr_fill]
        amps = [(lik.amps[:, q].mean(0) * scale, lik.amps[:, q].std(0) * scale) for q in range(lik.P)]
        pc = []
        for j, (pa, pb) in enumerate(PAIRS):
            if pa in sigs and pb in sigs:
                z = [np.fft.rfft(sigs[x][..., :2], axis=0)[D.HARMONICS["once"]] / lik.frames for x in (pa, pb)]
                pc.append(_coupling(*z).mean(0))
            else:
                pc.append(np.full(2, np.nan))
        coup = ((lik.coup.mean(0).ravel(), lik.coup.std(0).ravel()), np.array(pc).ravel())
        figure_png(manner, actor, tr_prog, tr_fill, claimed, amps, pred, own, coup, out / f"{manner}.png", AMPS)
        # the animation: the video | each part alone | its matched output alone
        traj = load_trajectory(cfg, videos[med])
        idx = sampled_frames(traj, lik.frames)
        cell = cell_px(pre)
        centres = walker_centres(cfg, videos[med], idx)
        aspect = (pre["window_x"][1] - pre["window_x"][0]) / (pre["window_y"][1] - pre["window_y"][0])
        film = video_frames(videos[med].path, idx, 3 * 6 * CELL + 12, centres, 0.8 * aspect)
        small = (cell[0] // 2, cell[1] // 2)
        frames = []
        for t in range(lik.frames):
            top = grid([board_px(colour(obs[med][t][q:q + 1], [p]), small) for q, p in enumerate(PART_NAMES)])
            bot = grid([board_px(program_board(v[t][out_of[q]:out_of[q] + 1], [p]), small) if q in out_of
                        else np.full(board_px(np.zeros((S, S, 3)), small).shape, 255, np.uint8)
                        for q, p in enumerate(PART_NAMES)])
            fills = grid([board_px(colour(fill_board(lik, v, out_of[q], t)[None], [p]), small) if q in out_of
                          else np.full(board_px(np.zeros((S, S, 3)), small).shape, 255, np.uint8)
                          for q, p in enumerate(PART_NAMES)])
            w = board_px(np.zeros((S, S, 3)), small).shape[1]
            head = label_strip([p.replace("_", " ") for p in PART_NAMES], w)
            gap = np.full((6, top.shape[1], 3), 255, np.uint8)
            frames.append(np.concatenate([head, top, gap, bot, gap, fills]))
        gif([film, frames], out / f"{manner}.gif")
        print(f"{manner}: medoid {videos[med].actor}, program {g['top'][0]['nodes']} nodes", flush=True)


if __name__ == "__main__":
    main()
