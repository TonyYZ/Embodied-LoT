"""Diagnostic report: generic relational features vs the current occupancy representation.

    python -m grace_lot.analysis.relational_report [--out results/grace_v2/relational_report]

Uses the training and validation actors only (the test actors are sealed).
It touches no LoT component and no human data.

Quantitative checks:
  1. What occupancy loses: ridge regression from one occupancy frame (12x12)
     to the relational features of that frame, cross-validated across actors.
     R^2 near 0 means occupancy does not contain that relation. Raw normalised
     joints are the reference.
  2. Manner signal vs nuisance: leave-one-actor-out nearest-prototype
     recognition of the 26 manners (higher is better), and actor
     identification from per-video means across manners (lower = less
     actor-specific nuisance).
  3. Left/right label reliability: how often the estimator's L/R labels jump
     (a side-view risk for the bilateral features).

Visual pages for the diagnostic manners show: the original frames, the
canonical pose sequence (left side blue, right side orange), occupancy, the
full relational heatmap, a fixed set of relational traces (the same set for
every manner, overlaid across all actors), and per-video summaries.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.pose import PoseSequence
from grace_lot.grace.preprocess import load_trajectory, processed_paths
from grace_lot.grace.relational import FAMILIES, from_trajectory
from grace_lot.grace.trajectories import resample

DIAGNOSTIC = ["bowing", "crisscrossing", "jumping", "swinging"]
TRACE_SET = ["torso:tilt", "torso:length", "bilateral:knee_dx", "bilateral:ankle_dx", "crossing:thigh",
             "crossing:shank", "com:pelvis_height", "com:vertical_extent", "limb_rel:L_ankle_y",
             "limb_rel:R_ankle_y", "limb_rel:L_wrist_y", "limb_rel:R_wrist_y"]
LEFT = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---- quantitative checks --------------------------------------------------------------

def _ridge_cv(X, Y, groups, alpha=1.0, folds=5):
    """R^2 per output column, with whole actors held out in each fold."""
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)
    pred = np.zeros_like(Y)
    for f in range(folds):
        test = np.isin(groups, ug[f::folds])
        mx, my = X[~test].mean(0), Y[~test].mean(0)
        sx = X[~test].std(0) + 1e-6
        A = (X[~test] - mx) / sx
        W = np.linalg.solve(A.T @ A + alpha * len(A) * np.eye(A.shape[1]), A.T @ (Y[~test] - my))
        pred[test] = ((X[test] - mx) / sx) @ W + my
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - ss_res / ss_tot


def _loao(feats, manners, actors):
    """Leave-one-actor-out nearest-prototype recognition; feats are (N, D)."""
    top1 = top5 = 0
    um = sorted(set(manners))
    for a in sorted(set(actors)):
        tr = actors != a
        proto = np.stack([feats[tr & (manners == m)].mean(0) for m in um])
        for i in np.where(~tr)[0]:
            order = np.argsort(((proto - feats[i]) ** 2).sum(1))
            rank = [um[k] for k in order].index(manners[i])
            top1 += rank == 0
            top5 += rank < 5
    n = len(feats)
    return top1 / n, top5 / n


def _actor_id(feats, manners, actors):
    """Identify the actor of each video from the other videos of the same actor (manner held out)."""
    hits = 0
    for i in range(len(feats)):
        mask = manners != manners[i]
        ua = sorted(set(actors))
        cents = np.stack([feats[mask & (actors == a)].mean(0) for a in ua])
        hits += ua[int(np.argmin(((cents - feats[i]) ** 2).sum(1)))] == actors[i]
    return hits / len(feats)


def quantitative(cfg, videos):
    rows = []
    for v in videos:
        t = load_trajectory(cfg, v)
        rel16 = from_trajectory(t, frames=16)
        rows.append((v, t, rel16))
    manners = np.array([v.manner for v, _, _ in rows])
    actors = np.array([v.actor for v, _, _ in rows])
    names = rows[0][2].names

    # 1. decodability, per frame
    occ = np.concatenate([t.fields.reshape(16, -1) for _, t, _ in rows])
    jnt = np.concatenate([t.joints.reshape(16, -1) for _, t, _ in rows])
    rel = np.concatenate([r.values for _, _, r in rows])
    grp = np.repeat(actors, 16)
    r2_occ, r2_jnt = _ridge_cv(occ, rel, grp), _ridge_cv(jnt, rel, grp)
    decode = {}
    for fam in FAMILIES:
        idx = [i for i, n in enumerate(names) if n.startswith(fam + ":")]
        decode[fam] = {"occupancy": float(np.median(r2_occ[idx])), "joints": float(np.median(r2_jnt[idx]))}
    per_feature = {n: (float(r2_occ[i]), float(r2_jnt[i])) for i, n in enumerate(names)}

    # 2. manner recognition across actors, and actor identification
    mu, sd = rel.mean(0), rel.std(0) + 1e-6
    rel_seq = np.stack([((r.values - mu) / sd).ravel() for _, _, r in rows])
    summ_keys = [k for k in rows[0][2].summary if k.startswith(("periodicity", "alternation"))]
    rel_sum = np.stack([np.r_[((r.values - mu) / sd).mean(0), ((r.values - mu) / sd).std(0),
                              [r.summary[k] for k in summ_keys]] for _, _, r in rows])
    rel_sum = (rel_sum - rel_sum.mean(0)) / (rel_sum.std(0) + 1e-6)
    occ_seq = np.stack([t.fields.ravel() for _, t, _ in rows])
    jnt_seq = np.stack([t.joints.ravel() for _, t, _ in rows])
    reps = {"occupancy (12x12 x T)": occ_seq, "normalised joints (33x2 x T)": jnt_seq,
            "relational time series": rel_seq, "relational summaries (time-invariant)": rel_sum}
    recog = {k: _loao(x, manners, actors) for k, x in reps.items()}
    means = {"occupancy (12x12 x T)": np.stack([t.fields.mean(0).ravel() for _, t, _ in rows]),
             "normalised joints (33x2 x T)": np.stack([t.joints.mean(0).ravel() for _, t, _ in rows]),
             "relational time series": np.stack([((r.values - mu) / sd).mean(0) for _, _, r in rows]),
             "relational summaries (time-invariant)": rel_sum}
    actor_id = {k: _actor_id(x, manners, actors) for k, x in means.items()}

    # 3. L/R label reliability: frame-to-frame jumps of the L-R ankle offset larger than half the leg length
    jumps = []
    for _, t, _ in rows:
        j = t.joints_full
        d = j[:, 27, 0] - j[:, 28, 0]
        leg = np.median(np.linalg.norm(j[:, 23] - j[:, 27], axis=1))
        jumps.append(np.mean(np.abs(np.diff(d)) > 0.5 * leg))
    lr = {"median_jump_rate": float(np.median(jumps)), "videos_over_5pct": int(np.sum(np.array(jumps) > 0.05)),
          "n_videos": len(jumps)}
    return {"decode": decode, "per_feature": per_feature, "recognition": recog, "actor_id": actor_id,
            "lr_labels": lr, "n_videos": len(rows), "n_features": len(names)}, rows


# ---- figures ----------------------------------------------------------------------------

def _frames_from_video(cfg, video, traj, k=8, pad=0.15):
    """k original frames at evenly spaced times in the valid segment, cropped around the person."""
    import cv2
    pose = PoseSequence.load(processed_paths(cfg["data"]["processed_dir"], video)["pose"])
    s0, s1 = traj.segment
    idx = np.linspace(s0, s1 - 1, k).round().astype(int)
    cap = cv2.VideoCapture(str(video.path))
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        pts = pose.image[i]
        x0, y0 = np.nanmin(pts, 0); x1, y1 = np.nanmax(pts, 0)
        h = (y1 - y0) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a, b = int(max(0, cy - h / 2)), int(min(fr.shape[0], cy + h / 2))
        c, d = int(max(0, cx - h / 2)), int(min(fr.shape[1], cx + h / 2))
        out.append(cv2.cvtColor(fr[a:b, c:d], cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def pose_figure(cfg, video, traj, rel_occ_fields, k=8):
    frames = _frames_from_video(cfg, video, traj, k)
    joints = resample(traj.joints_full, k)
    occ = rel_occ_fields[np.linspace(0, len(rel_occ_fields) - 1, k).round().astype(int)]
    fig, ax = plt.subplots(3, k, figsize=(k * 1.1, 3 * 1.35))
    for i in range(k):
        if i < len(frames):
            ax[0, i].imshow(frames[i])
        for a, b in traj.connections:
            col = "tab:blue" if a in LEFT and b in LEFT else "tab:orange" if a in RIGHT and b in RIGHT else "k"
            ax[1, i].plot(joints[i, [a, b], 0], joints[i, [a, b], 1], color=col, lw=1.2)
        ax[1, i].set_xlim(-0.65, 0.65); ax[1, i].set_ylim(0.7, -0.6); ax[1, i].set_aspect("equal")
        ax[2, i].imshow(occ[i], cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            ax[r, i].axis("off")
    ax[0, 0].set_title("original", fontsize=7, loc="left")
    ax[1, 0].set_title("pose (L blue, R orange)", fontsize=7, loc="left")
    ax[2, 0].set_title("occupancy (current)", fontsize=7, loc="left")
    return _img(fig)


def heatmap_figure(rel, title):
    z = (rel.values - rel.values.mean(0)) / (rel.values.std(0) + 1e-6)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    fam_of = [n.split(":")[0] for n in rel.names]
    starts = [i for i in range(len(fam_of)) if i == 0 or fam_of[i] != fam_of[i - 1]]
    for s in starts:
        ax.axhline(s - 0.5, color="k", lw=0.6)
        ax.text(-0.8, s + 0.5, fam_of[s], ha="right", va="top", fontsize=7)
    ax.set_yticks([]); ax.set_xlabel("time (normalised, 32 frames)"); ax.set_title(title, fontsize=9)
    return _img(fig)


def traces_figure(rels_all, highlight, names):
    n = len(TRACE_SET)
    fig, axes = plt.subplots(3, 4, figsize=(11, 6.3))
    for ax, feat in zip(axes.flat, TRACE_SET):
        i = names.index(feat)
        for r in rels_all:
            ax.plot(r.values[:, i], color="0.75", lw=0.6)
        ax.plot(highlight.values[:, i], color="crimson", lw=1.6)
        ax.set_title(feat, fontsize=8); ax.tick_params(labelsize=6)
    fig.suptitle("fixed trace set (the same for every manner): all actors grey, the shown actor red", fontsize=8)
    fig.tight_layout()
    return _img(fig)


def summary_figure(rels_all, manner):
    keys = [k for k in rels_all[0].summary if k.startswith("alternation")]
    vals = np.array([[r.summary[k] for k in keys] for r in rels_all])
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.boxplot(vals, tick_labels=[k.split(":")[1] for k in keys])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("L-R correlation"); ax.set_title(f"{manner}: left/right alternation across actors (-1 = anti-phase)", fontsize=8)
    ax.tick_params(labelsize=7)
    return _img(fig)


def decode_figure(q):
    fams = list(q["decode"])
    occ = [q["decode"][f]["occupancy"] for f in fams]
    jnt = [q["decode"][f]["joints"] for f in fams]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(x - 0.2, jnt, 0.4, label="from normalised joints")
    ax.bar(x + 0.2, occ, 0.4, label="from occupancy (current)")
    ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=8); ax.set_ylabel("median R^2 (actor-held-out)")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=7); ax.set_ylim(min(-0.1, min(occ) - 0.05), 1.0)
    return _img(fig)


# ---- page --------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/grace_v2/relational_report")
    args = ap.parse_args()
    cfg = load_config()
    d = cfg["data"]
    tv = split_videos(d["raw_dir"], cfg["split"], "trainval")
    sealed = set(cfg["split"]["test_actors"])
    assert not any(v.actor in sealed for vs in tv.values() for v in vs)
    videos = [v for m in sorted(tv) for v in tv[m]]
    q, rows = quantitative(cfg, videos)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(q, indent=1))

    css = ("body{font:14px system-ui,sans-serif;margin:24px;max-width:1250px;color:#222}"
           "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}"
           "th{background:#f3f3f3}td.l{text-align:left}section{border-top:2px solid #ddd;margin-top:26px}img{max-width:100%}")
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Relational front-end diagnostics</title><style>{css}</style></head><body>",
         "<h1>Relational motion features vs occupancy: diagnostics</h1>",
         f"<p>{q['n_videos']} videos from the 20 training and validation actors (test actors sealed). "
         f"{q['n_features']} generic relational time series per video, in the families "
         f"{', '.join(FAMILIES)}. The LoT, prior, sampler and human data are untouched.</p>",
         "<p><b>Assessment and recommendation:</b> <a href='interface_assessment.md'>interface_assessment.md</a></p>",
         "<h2>1. What occupancy retains (actor-held-out decoding)</h2>",
         f"<img src='{decode_figure(q)}'>",
         "<table><tr><th>family</th><th>R&sup2; from joints</th><th>R&sup2; from occupancy</th></tr>"]
    for f, v in q["decode"].items():
        H.append(f"<tr><td class='l'>{f}</td><td>{v['joints']:.2f}</td><td>{v['occupancy']:.2f}</td></tr>")
    H.append("</table>")
    worst = sorted(q["per_feature"].items(), key=lambda kv: kv[1][0])[:12]
    H.append("<p>Least recoverable from occupancy (feature: occupancy R&sup2; / joints R&sup2;): "
             + "; ".join(f"{html.escape(n)} {o:.2f}/{j:.2f}" for n, (o, j) in worst) + "</p>")
    H += ["<h2>2. Manner signal vs actor nuisance</h2>",
          "<table><tr><th>representation</th><th>manner top-1 (LOAO)</th><th>manner top-5</th>"
          "<th>actor identification (lower = less nuisance)</th></tr>"]
    for k, (t1, t5) in q["recognition"].items():
        H.append(f"<tr><td class='l'>{k}</td><td>{t1:.3f}</td><td>{t5:.3f}</td><td>{q['actor_id'][k]:.3f}</td></tr>")
    H.append("</table><p>Chance: manner top-1 0.038, top-5 0.19; actor identification 0.05.</p>")
    lr = q["lr_labels"]
    H += ["<h2>3. Left/right label reliability</h2>",
          f"<p>Median per-video rate of implausible L/R ankle jumps: {lr['median_jump_rate']:.3f}; "
          f"{lr['videos_over_5pct']} of {lr['n_videos']} videos exceed 5% of frames. In side view, L/R labels are a "
          "risk for the bilateral and crossing families.</p>"]

    by_manner = {}
    for v, t, r in rows:
        by_manner.setdefault(v.manner, []).append((v, t))
    for m in DIAGNOSTIC:
        vt = by_manner[m]
        rels32 = [from_trajectory(t, frames=32) for _, t in vt]
        mean = np.mean([r.values for r in rels32], axis=0)
        med = int(np.argmin([((r.values - mean) ** 2).sum() for r in rels32]))
        v, t = vt[med]
        r = rels32[med]
        alt = {k.split(":")[1]: round(np.median([x.summary[k] for x in rels32]), 2)
               for k in r.summary if k.startswith("alternation")}
        H += [f"<section><h2>{m}</h2><p>Shown actor: <b>{v.stem}</b> (medoid in relational space among "
              f"{len(vt)} actors). Median L/R alternation across actors: {alt}.</p>",
              f"<img src='{pose_figure(cfg, v, t, t.fields)}'>",
              f"<img src='{heatmap_figure(r, f'{v.stem}: all {len(r.names)} relational features (z-scored)')}'>",
              f"<img src='{traces_figure(rels32, r, r.names)}'>",
              f"<img src='{summary_figure(rels32, m)}'></section>"]
    H.append("</body></html>")
    (out / "index.html").write_text("\n".join(H), encoding="utf-8")
    print(json.dumps({k: q[k] for k in ("decode", "recognition", "actor_id", "lr_labels")}, indent=1))


if __name__ == "__main__":
    main()
