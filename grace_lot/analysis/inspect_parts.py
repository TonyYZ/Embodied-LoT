"""Inspect a parts-interface run: anonymous LoT outputs matched to skeleton parts. Diagnostic only.

    python -m grace_lot.analysis.inspect_parts --run results/grace_parts/diag

For each manner (training actors only), score every sampled program on the
same training videos, keep the top 3 distinct programs by log posterior, and
show for the medoid actor:
  * the observed part channels, coloured by part;
  * each program's outputs, coloured by the part each was matched to (one
    mapping for the whole sequence);
  * centre-of-mass traces (x and y) of every matched output/part pair;
  * an animated GIF (observed parts | top-1 outputs) and each top program's tree.
Under the unknown background, outputs are shown by what they draw as figure (2a - 1, clipped),
so undrawn (unknown) cells stay dark. The page is written to <run>/inspection/index.html
(self-contained).
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

import embodied_lot
from grace_lot.analysis.inspect_programs import tree_svg
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.parts import PART_NAMES, part_fields
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.inference.grammar import Grammar
from grace_lot.inference.likelihood import MatchedPartsLikelihood
from grace_lot.inference.prior import SizePrior
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.program import depth, parse_program, saved_lot, size

COLORS = {"head": (0.6, 0.6, 0.6), "torso": (0.95, 0.95, 0.95), "left_arm": (0.2, 0.45, 1.0),
          "right_arm": (1.0, 0.55, 0.1), "left_leg": (0.1, 0.85, 0.9), "right_leg": (0.95, 0.15, 0.2),
          None: (0.75, 0.3, 0.9)}           # unmatched output


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colour(chans, labels):
    """(C, H, W) channels -> RGB image, each channel tinted by its label's colour (max-composited)."""
    out = np.zeros(chans.shape[1:] + (3,))
    for c, lab in zip(chans, labels):
        out = np.maximum(out, np.clip(c, 0, 1)[..., None] * np.array(COLORS[lab]))
    return out


def figure(y):
    """What an output draws as figure: unknown (0.5) and known-empty cells both show as dark."""
    return np.clip(2 * y - 1, 0, 1) if embodied_lot.UNKNOWN_BACKGROUND else y


def parts_gif(obs, y, labels) -> str:
    """Animated: observed parts (coloured by part) | top-1 outputs (coloured by matched part), as a data URI."""
    from PIL import Image
    frames = []
    for t in range(len(obs)):
        a = colour(obs[t], PART_NAMES); b = colour(figure(y[t]), labels)
        gap = np.ones((a.shape[0], 1, 3))
        row = np.concatenate([a, gap, b], axis=1)
        img = (np.kron(row, np.ones((16, 16, 1))) * 255).astype(np.uint8)
        frames.append(Image.fromarray(img))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=250, loop=0)
    return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()


def centroid(f):
    m = f.sum() + 1e-9
    H, W = f.shape
    return (f.sum(0) * np.arange(W)).sum() / m, (f.sum(1) * np.arange(H)).sum() / m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/grace_parts/diag")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = load_config()
    rc = json.loads((run / "config.json").read_text())
    for k in ("preprocess", "lot", "inference", "split", "interface"):
        if k in rc:
            cfg[k] = rc[k]
    cfg["lot"] = saved_lot(cfg["lot"])
    scan_mode = cfg.get("interface", {}).get("representation") == "parts_scan"
    itf = cfg.get("interface", {})
    pre, inf = cfg["preprocess"], cfg["inference"]
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    assert not any(v.actor in set(cfg["split"]["test_actors"]) for vs in train.values() for v in vs)
    g = Grammar.from_config(cfg["lot"])
    prior = SizePrior.from_config(g, inf)
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    css = ("body{font:14px system-ui,sans-serif;margin:24px;max-width:1250px}img{max-width:100%}"
           "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px}"
           "section{border-top:2px solid #ddd;margin-top:24px}pre{font-size:11px;background:#f7f7f7;padding:6px}")
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Parts interface inspection</title><style>{css}</style></head><body>",
         "<h1>Anonymous LoT outputs matched to skeleton parts</h1>",
         f"<p>Run <code>{html.escape(str(run))}</code>, training actors only. One output-to-part mapping per video "
         "for the whole sequence. Part colours: " + ", ".join(f"<span style='color:rgb{tuple(int(255*x) for x in COLORS[p])}'>"
                                                          f"&#9632;</span> {p}" for p in PART_NAMES)
         + "; purple = output matched to no part.</p>"]
    summary_rows = []
    for post_file in sorted((run / "posteriors").glob("*.json")):
        manner = post_file.stem
        post = json.loads(post_file.read_text())
        trajs = [load_trajectory(cfg, v) for v in train[manner]]
        obs = [part_fields(t, pre) for t in trajs]
        if scan_mode:
            from grace_lot.grace.scanner import scan_parts
            from grace_lot.inference.likelihood import MatchedScanLikelihood
            acts = [scan_parts(t, pre, itf["scan_softness"], itf["edge_samples"]) for t in trajs]
            mk = lambda os_: MatchedScanLikelihood(os_, renderer, inf["beta"], itf["scan_softness"], itf["lot_salience"],
                                                   inf.get("temporal_weight", 0.0))
            lik = mk(acts)
            mean_a = np.mean(acts, axis=0)
            med = int(np.argmin([((a - mean_a) ** 2).mean() for a in acts]))
            lik_med = mk([acts[med]])
        else:
            lik = MatchedPartsLikelihood(obs, renderer, inf["beta"], inf.get("temporal_weight", 0.0))
            mean = np.mean(obs, axis=0)
            med = int(np.argmin([((o - mean) ** 2).mean() for o in obs]))
            lik_med = MatchedPartsLikelihood([obs[med]], renderer, inf["beta"], inf.get("temporal_weight", 0.0))
        scored = sorted(((prior(p) + lik(p), prior(p), lik(p), p) for p in
                         (parse_program(s) for s, _ in post["posterior"][:600])), key=lambda x: -x[0])
        top, seen = [], set()
        for row in scored:
            if str(row[3]) not in seen:
                seen.add(str(row[3])); top.append(row)
            if len(top) == 3:
                break
        o = obs[med]
        T = len(o)
        fig, ax = plt.subplots(1 + len(top), T, figsize=(T * 0.6, (1 + len(top)) * 0.7))
        traces = []
        for t in range(T):
            ax[0, t].imshow(colour(o[t], PART_NAMES)); ax[0, t].axis("off")
        ax[0, 0].set_title("observed parts", fontsize=6, loc="left")
        rows_html = []
        for r, (lp_post, lp, ll, p) in enumerate(top, start=1):
            y = renderer(p)
            y = y if y.ndim == 4 else y[:, None]
            ri, ci, d = lik_med.assignments(p)[0]
            label = {k: (PART_NAMES[c] if c < len(PART_NAMES) else None) for k, c in zip(ri, ci) if k < y.shape[1]}
            for t in range(T):
                ax[r, t].imshow(colour(figure(y[t]), [label[k] for k in range(y.shape[1])])); ax[r, t].axis("off")
            ax[r, 0].set_title(f"top-{r}", fontsize=6, loc="left")
            pairs = [(k, label[k]) for k in range(y.shape[1]) if label[k] is not None]
            traces.append((r, y, pairs))
            rows_html.append(f"<tr><td>{r}</td><td>{lp_post:.1f}</td><td>{lp:.1f}</td><td>{ll:.1f}</td><td>{d:.4f}</td>"
                             f"<td>{y.shape[1]}</td><td>{size(p.tree)}</td><td>{depth(p.tree)}</td><td>{len(p.dashed)}</td>"
                             f"<td style='text-align:left'>{', '.join(f'out{k}&rarr;{lab}' for k, lab in pairs) or '-'}</td></tr>"
                             f"<tr><td colspan=10><pre>{html.escape(str(p))}</pre>"
                             f"<div style='overflow-x:auto'>{tree_svg(p)}</div></td></tr>")
        strip = _img(fig)
        r, y, pairs = traces[0]
        anim = parts_gif(o, y, [dict(pairs).get(k) for k in range(y.shape[1])])
        if scan_mode:
            # detector activations of the top-1 matched pairs: observed part (solid) vs LoT output (dashed)
            from grace_lot.grace.scanner import DETECTORS
            a_lot = lik_med.scan(top[0][3])
            fig, axs = plt.subplots(1, max(1, len(pairs)), figsize=(3.0 * max(1, len(pairs)), 2.6), squeeze=False)
            cols = plt.cm.tab10(np.arange(6))
            for j, (k, lab) in enumerate(pairs):
                q = PART_NAMES.index(lab)
                for d in range(6):
                    axs[0, j].plot(acts[med][:, q, d], color=cols[d], lw=1.6, label=DETECTORS[d])
                    axs[0, j].plot(a_lot[:, k, d], color=cols[d], lw=1.0, ls="--")
                axs[0, j].set_title(f"out{k} -> {lab}", fontsize=7); axs[0, j].set_ylim(-0.05, 1.05); axs[0, j].tick_params(labelsize=6)
            axs[0, 0].legend(fontsize=5, ncol=2)
            fig.tight_layout()
        else:
            # centroid traces for the top-1 matched pairs
            fig, axs = plt.subplots(2, max(1, len(pairs)), figsize=(2.3 * max(1, len(pairs)), 3.4), squeeze=False)
            for j, (k, lab) in enumerate(pairs):
                q = PART_NAMES.index(lab)
                co = np.array([centroid(o[t, q]) for t in range(T)]); cl = np.array([centroid(y[t, k]) for t in range(T)])
                for a, name in ((0, "x"), (1, "y")):
                    axs[a, j].plot(co[:, a], color=COLORS[lab], lw=2, label="observed")
                    axs[a, j].plot(cl[:, a], "k--", lw=1.2, label="LoT")
                    axs[a, j].set_title(f"{lab} centroid {name}", fontsize=7); axs[a, j].tick_params(labelsize=6)
            axs[0, 0].legend(fontsize=6)
            fig.tight_layout()
        tr = _img(fig)
        H += [f"<section><h2>{manner}</h2><p>Medoid actor {trajs[med].segment and train[manner][med].stem}; "
              f"R-hat {post['r_hat_log_post']:.2f}; {post['unique_programs']} distinct programs.</p>",
              f"<p>Observed parts (left) and the top-1 program's outputs, each tinted by the part it is matched to "
              f"(right):</p><img src='{anim}' style='image-rendering:pixelated'>",
              f"<img src='{strip}'>",
              "<table><tr><th>rank</th><th>log post</th><th>log prior</th><th>log lik</th><th>D (medoid)</th><th>outputs</th>"
              "<th>nodes</th><th>depth</th><th>dashed</th><th>mapping (medoid)</th></tr>" + "".join(rows_html) + "</table>",
              f"<p>{'Detector activations (solid = observed part, dashed = LoT output)' if scan_mode else 'Centroid traces'}, "
              f"top-1 program, matched pairs:</p><img src='{tr}'></section>"]
        summary_rows.append((manner, top[0][0], len(traces[0][2]), post["r_hat_log_post"]))
    H.insert(3, "<table><tr><th>manner</th><th>top-1 log post</th><th>matched outputs</th><th>R-hat</th></tr>"
             + "".join(f"<tr><td>{m}</td><td>{a:.1f}</td><td>{b}</td><td>{c:.2f}</td></tr>" for m, a, b, c in summary_rows)
             + "</table>")
    H.append("</body></html>")
    out = run / "inspection"; out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text("\n".join(H), encoding="utf-8")
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
