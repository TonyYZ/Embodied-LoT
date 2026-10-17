"""Synthetic recovery checks as a page: a known program, its synthetic video, and what inference found.

    python -m grace_lot.analysis.recovery_page --recovery results/grace_v3/recovery/recovery.json

For each example in a ``grace_lot.inference.recovery`` run (occupancy interface), writes a GIF
(the first synthetic actor's video | the true program's rendering | the recovered MAP's
rendering), both program trees, and the recovery scores, into ``recovery.html`` next to the
json. The GIFs are embedded, so the page stands alone. The run's own recorded semantics are applied, and the synthetic video is rebuilt from
the recorded seed, so it is exactly what inference saw. No human data is read.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from grace_lot.analysis.inspect_programs import _panel, tree_svg
from grace_lot.inference.recovery import synthetic_copies
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot


def recovery_gif(video: np.ndarray, truth: np.ndarray, found: np.ndarray, path: Path, fps: int = 4,
                 grace: np.ndarray = None) -> None:
    """(the real GRACE video, when given |) synthetic video | true program | recovered, on one grey scale."""
    import matplotlib
    from PIL import Image, ImageDraw
    gray = matplotlib.colormaps["gray"]            # one scale for all: yin black, unknown grey, yang white
    seqs = ([grace] if grace is not None else []) + [video, truth, found]
    titles = (["GRACE video"] if grace is not None else []) + ["synthetic video", "true program", "recovered"]
    frames = []
    for t in range(len(video)):
        panels = [_panel(y[t], gray) for y in seqs]
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        row = np.concatenate(sum([[p, gap] for p in panels], [])[:-1], axis=1)
        canvas = Image.new("RGB", (row.shape[1], row.shape[0] + 18), "white")
        canvas.paste(Image.fromarray(row), (0, 18))
        d = ImageDraw.Draw(canvas)
        for i, title in enumerate(titles):
            d.text((i * (panels[0].shape[1] + 6) + 3, 3), title, fill="black")
        d.text((canvas.width - 40, 3), f"t={t}", fill="black")
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def grace_medoid(rec: dict, manner: str) -> np.ndarray:
    """The manner's medoid training actor's field sequence (the one closest to the training mean)."""
    from grace_lot.grace.dataset import split_videos
    from grace_lot.grace.preprocess import load_trajectory
    from grace_lot.lot_adapter.distance import euclid
    cfg = {"data": {k: Path(v) if k.endswith("_dir") else v for k, v in rec["data"].items()},
           "preprocess": rec["preprocess"], "split": rec["split"]}
    videos = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")[manner]
    obs = [load_trajectory(cfg, v).fields for v in videos]
    mean = np.mean(obs, axis=0)
    return obs[int(np.argmin([euclid(o, mean) for o in obs]))]


def write_page(rec: dict, out: Path) -> Path:
    apply_semantics(saved_lot(rec.get("lot", {})))
    pre = rec["preprocess"]
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    (out / "recovery_gifs").mkdir(parents=True, exist_ok=True)
    css = ("body{font:14px system-ui,sans-serif;margin:24px;max-width:1300px;color:#222}"
           "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}"
           "th{background:#f3f3f3}.trees{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}"
           ".ok{color:#1f7a45;font-weight:600}.bad{color:#b3261e;font-weight:600}code{font-size:12px}")
    yes = lambda b: "<span class='ok'>yes</span>" if b else "<span class='bad'>no</span>"
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>Synthetic recovery</title><style>{css}</style></head><body>",
         "<h1>Synthetic recovery checks</h1>",
         f"<p>Each example is a known program rendered as {rec['actors']} synthetic actors (Gaussian noise "
         f"{rec['noise']}, time shift up to {rec['shift']} frames), fitted by exactly the configured inference "
         f"(lambda {rec['inference']['lambda_size']}, beta {rec['inference']['beta']}, "
         f"{rec['inference']['iterations']} iterations). The GIF shows the first synthetic actor, the true program's "
         "rendering and the recovered MAP's rendering; for a truth fitted to a GRACE manner, that manner's real video "
         "comes first. <b>Same rendering</b> is what matters: programs that render "
         "identically are indistinguishable by any likelihood.</p>",
         f"<p>Search found a state at least as good as the truth in {rec['search_ok_rate']:.0%} of examples; "
         f"the MAP renders like the truth in {rec['map_equiv_rate']:.0%}; mean posterior mass on truth-equivalent "
         f"renderings {rec['mean_equiv_mass']:.2f}.</p>"]
    for r in rec["results"]:
        truth, found = parse_program(r["truth"]), parse_program(r["map"])
        y_true, y_found = renderer(truth), renderer(found)
        video = synthetic_copies(y_true, 1, rec["noise"], rec["shift"], np.random.default_rng(r["seed"]))[0]
        gif = out / "recovery_gifs" / f"example_{r['k']}.gif"
        label = r.get("label", "")
        grace = grace_medoid(rec, label.split(":", 1)[1]) if label.startswith("manner:") else None
        recovery_gif(video, y_true, y_found, gif, grace=grace)
        source = (f"the MAP program fitted to <b>{html.escape(label.split(':', 1)[1])}</b> (shown first: its medoid "
                  "training actor)" if grace is not None else html.escape(label or "prior draw"))
        h += [f"<section><h2>Example {r['k']}</h2><p>True program: {source}.</p>",
              f"<img src='data:image/gif;base64,{base64.b64encode(gif.read_bytes()).decode()}' "
              "alt='synthetic video, true and recovered renderings'>",       # embedded: the page stands alone
              "<table><tr><th>search found truth-level state</th><th>MAP renders like truth</th>"
              "<th>mass on truth-equivalent</th><th>mass on exact truth</th><th>similarity to truth (prior)</th>"
              "<th>R-hat</th></tr>",
              f"<tr><td>{yes(r['search_ok'])}</td><td>{yes(r['map_equiv'])}</td><td>{r['equiv_mass']:.2f}</td>"
              f"<td>{r['exact_mass']:.2f}</td><td>{r['sim_to_truth']:.2f} ({r['sim_prior']:.2f})</td>"
              f"<td>{r['r_hat']:.2f}</td></tr></table>",
              "<div class='trees'>",
              f"<div><h3>True program</h3><code>{html.escape(r['truth'])}</code><br>{tree_svg(truth)}</div>",
              f"<div><h3>Recovered (MAP)</h3><code>{html.escape(r['map'])}</code><br>{tree_svg(found)}</div>",
              "</div></section>"]
    h.append("</body></html>")
    page = out / "recovery.html"
    page.write_text("\n".join(h), encoding="utf-8")
    return page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery", required=True, help="a recovery.json from grace_lot.inference.recovery")
    args = ap.parse_args()
    path = Path(args.recovery)
    rec = json.loads(path.read_text())
    if rec.get("interface", {}).get("representation", "occupancy") != "occupancy":
        raise SystemExit("recovery_page shows occupancy-interface runs only")
    print(f"wrote {write_page(rec, path.parent)}")


if __name__ == "__main__":
    main()
