"""A one-page demonstration that motion is represented in the grounded LoT.

    python -m grace_lot.analysis.motion_demo --recovery results/grace_v3/motion_demo/recovery.json

Writes ``motion_demo.html`` next to the recovery json: programs whose structure makes things
move (rendered by the engine, with their trees), and synthetic recovery of moving and static
programs from noisy videos (synthetic video | true | recovered, both trees, and each frame's
position of the figure). The page is self-contained (GIFs embedded). No human data is read.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

from grace_lot.analysis.inspect_programs import _panel, tree_svg
from grace_lot.analysis.recovery_page import recovery_gif
from grace_lot.inference.recovery import synthetic_copies
from grace_lot.lot_adapter.adapter import rollout
from grace_lot.lot_adapter.program import apply_semantics, parse_program, saved_lot

SHOWCASE = [
    ("Self-observing oscillator",
     "(serial_h (parallel bottom/H* center/H) (parallel top/H* center/H (body[1,0.5,0.06] center/H))) | .0.1>.0.0 .1.1>.1.0",
     "Two observers watch the board. Seeing the body on the right switches on the leftward motion leaf; "
     "seeing it on the left switches on the rightward one. Rhythm comes from a loop of dashed edges, not from "
     "a clock."),
    ("Motion inside a turned frame",
     "(parallel (plan top/H* (serial_h neutral/H (parallel top/H* (body[1,0.5,0.06] center/V)))))",
     "The template turns the whole content to face left. Inside it, a leaf pushes the body 'left' in its own "
     "frame, which on the board is downward. Motion is relative to the frame the program sets up."),
    ("Two motions add",
     "(parallel (body[0.34,0.34,0.06] (parallel universal/V bottom/V* bottom/H*)))",
     "An upward and a rightward motion leaf in the same space: their orders add, and the body moves diagonally."),
]

CSS = """
:root{--bg:#f7f6f2;--panel:#fff;--ink:#1d1d1b;--muted:#6b6a64;--line:#d9d6cc;--ok:#1f7a45;--bad:#b3261e;
  --truth:#2f6fdb;--found:#d9761f;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--panel:#212126;--ink:#ecebe6;
  --muted:#a3a29b;--line:#3a3a40;--ok:#4cc98a;--bad:#ff8a80;--truth:#6fa0ff;--found:#f0a050}}
:root[data-theme="dark"]{--bg:#17171a;--panel:#212126;--ink:#ecebe6;--muted:#a3a29b;--line:#3a3a40;--ok:#4cc98a;
  --bad:#ff8a80;--truth:#6fa0ff;--found:#f0a050}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);line-height:1.5}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.6rem;margin:0 0 6px}
h2{font-size:1.2rem;margin:34px 0 8px}h3{font-size:1rem;margin:14px 0 6px}p{margin:6px 0}.muted{color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.row{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-start}.row>div{min-width:0}
img{max-width:100%;height:auto;image-rendering:pixelated;border-radius:4px}
.tree{background:#fff;border-radius:8px;padding:4px;max-width:100%;overflow-x:auto}
code{font-family:ui-monospace,Consolas,monospace;font-size:.8rem;word-break:break-all;color:var(--muted)}
.verdict{font-weight:600}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{border-collapse:collapse;font-size:.85rem}td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}
.note{border-left:4px solid var(--line);padding:6px 12px;margin:10px 0}
"""


def _gif_uri(path: Path) -> str:
    return "data:image/gif;base64," + base64.b64encode(path.read_bytes()).decode()


def board_gif(frames: np.ndarray, path: Path, fps: int = 5) -> None:
    import matplotlib
    from PIL import Image
    gray = matplotlib.colormaps["gray"]
    imgs = [Image.fromarray(_panel(f, gray)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def track(y: np.ndarray) -> np.ndarray:
    """Each frame's centre of the clearly drawn (yang) cells, as (row, column); NaN where nothing is drawn."""
    out = []
    for f in y:
        r, c = np.nonzero(f > 0.75)
        out.append((r.mean() + 0.5, c.mean() + 0.5) if r.size else (np.nan, np.nan))
    return np.array(out)


def track_svg(truth: np.ndarray, found: np.ndarray, size: int = 12) -> str:
    """Two small plots (column, row) of where the figure is in each frame: true program vs recovered."""
    W, H, P = 230, 120, 26
    T = len(truth)
    x = lambda t: P + t * (W - P - 8) / max(T - 1, 1)
    yv = lambda v: 8 + v * (H - 30) / size
    out = []
    for axis, name in ((1, "column (left → right)"), (0, "row (top → bottom)")):
        parts = [f"<svg viewBox='0 0 {W} {H}' width='{W}' style='max-width:100%;height:auto'>",
                 f"<rect x='{P}' y='8' width='{W - P - 8}' height='{H - 30}' fill='none' stroke='currentColor' "
                 "stroke-opacity='.25'/>"]
        for v in (0, 6, 12):
            parts.append(f"<text x='{P - 4}' y='{yv(v) + 3:.1f}' text-anchor='end' font-size='8' fill='currentColor' "
                         f"fill-opacity='.6'>{v}</text>")
        for seq, colour, dash in ((truth, "var(--truth)", ""), (found, "var(--found)", " stroke-dasharray='5 3'")):
            pts = " ".join(f"{x(t):.1f},{yv(v):.1f}" for t, v in enumerate(seq[:, axis]) if not np.isnan(v))
            if pts:
                parts.append(f"<polyline points='{pts}' fill='none' stroke='{colour}' stroke-width='2.5'{dash}/>")
        parts.append(f"<text x='{P}' y='{H - 8}' font-size='9' fill='currentColor' fill-opacity='.7'>{name} · frame →</text>")
        parts.append("</svg>")
        out.append("".join(parts))
    return "<div class='row'>" + "".join(f"<div>{s}</div>" for s in out) + "</div>"


def describe_motion(tr: np.ndarray) -> str:
    ok = tr[~np.isnan(tr).any(axis=1)]
    if len(ok) < 2:
        return "nothing drawn"
    dr, dc = ok[-1] - ok[0]
    parts = [f"{abs(dc):.0f} cells {'right' if dc > 0 else 'left'}" if abs(dc) >= 0.5 else "",
             f"{abs(dr):.0f} cells {'down' if dr > 0 else 'up'}" if abs(dr) >= 0.5 else ""]
    parts = [p for p in parts if p]
    return " and ".join(parts) if parts else "stays still"


def write(rec: dict, out: Path) -> Path:
    apply_semantics(saved_lot(rec.get("lot", {})))
    frames, size = rec["preprocess"]["frames"], rec["preprocess"]["field_size"]
    tmp = out / "demo_gifs"
    tmp.mkdir(parents=True, exist_ok=True)
    h = ["<title>Motion in a grounded LoT</title>", f"<style>{CSS}</style>", "<main>",
         "<h1>Motion in a grounded language of thought</h1>",
         "<p class='muted'>Programs are trees of perceptual primitives (trigrams), compositions (parallel, serial, "
         "plan, body) and dashed gate edges, run on a 12×12 board. Everything below is produced by the engine itself.</p>",
         "<div class='card'><b>The claim.</b> Motion is not a separate feature bolted on: it is program structure. "
         "A motion leaf orders bodies in its space to move; observers watching the board gate those leaves through "
         "dashed edges, which gives rhythm; templates turn content, so motion is relative to a frame; and inference "
         "can recover these moving programs from noisy videos.</div>",
         "<h2>1. Programs whose structure makes things move</h2>",
         "<p class='muted'>Each board is the program's own rendering, frame by frame (black: known empty, grey: unknown, "
         "white: figure). In the trees, orange leaves are motion leaves; dashed outlines mark templates; pink arrows are "
         "dashed gate edges.</p>"]
    for k, (title, src, note) in enumerate(SHOWCASE):
        prog = parse_program(src)
        y = rollout(prog, 24, size)
        gif = tmp / f"show_{k}.gif"
        board_gif(y, gif)
        h += [f"<div class='card'><h3>{html.escape(title)}</h3><p>{html.escape(note)}</p>",
              f"<div class='row'><div><img src='{_gif_uri(gif)}' width='192' alt='{html.escape(title)} board'></div>"
              f"<div class='tree'>{tree_svg(prog)}</div></div>",
              f"<p><code>{html.escape(src)}</code></p></div>"]
    h += ["<p>A step-through explainer of the board rules (who observes, when orders are issued, how bodies move) "
          "is in the separate <i>Gameboard dynamics</i> page.</p>",
          "<h2>2. Inference recovers motion from noisy videos</h2>",
          f"<p class='muted'>Each true program below is rendered as {rec['actors']} synthetic actors (noise "
          f"{rec['noise']}, each shifted in time by up to {rec['shift']} frame) and fitted by the same Bayesian "
          f"program inference used for GRACE ({rec['inference']['iterations']} iterations × "
          f"{rec['inference']['chains']} chains, tempered MCMC). The fit never sees the true program. The last example is "
          "a static control: the method should not invent motion.</p>"]
    names = ["Slides left", "Drops", "Up and right at once", "Static control (no motion)"]
    for r in rec["results"]:
        truth, found = parse_program(r["truth"]), parse_program(r["map"])
        y_t, y_f = rollout(truth, frames, size), rollout(found, frames, size)
        video = synthetic_copies(y_t, 1, rec["noise"], rec["shift"], np.random.default_rng(r["seed"]))[0]
        gif = tmp / f"rec_{r['k']}.gif"
        recovery_gif(video, y_t, y_f, gif)
        tt, tf = track(y_t), track(y_f)
        same = r["map_equiv"]
        short = rec["inference"]["iterations"] < 40000
        this_run = "this short run" if short else "the search"
        if same:
            verdict = "<span class='verdict ok'>Recovered</span>: the recovered program renders exactly like the true one"
        elif not r["search_ok"]:
            verdict = (f"<span class='verdict bad'>Not found yet</span>: a search failure, not a model failure. The "
                       f"model scores the true program far higher (log posterior {r['truth_log_post']:.0f} vs "
                       f"{r['best_log_post']:.0f} for the best program the chains reached), but {this_run} "
                       "did not reach it")
        else:
            verdict = "<span class='verdict bad'>Not exact</span>: the recovered program renders differently"
        name = names[r["k"]] if r["k"] < len(names) else f"Example {r['k']}"
        h += [f"<div class='card'><h3>{html.escape(name)}</h3>",
              f"<p>{verdict}. True motion: <b>{describe_motion(tt)}</b>; recovered: <b>{describe_motion(tf)}</b>. "
              f"Posterior mass on programs that render like the truth: {r['equiv_mass']:.2f}.</p>",
              f"<img src='{_gif_uri(gif)}' alt='synthetic video, true and recovered renderings'>",
              "<p class='muted'>Where the figure is in each frame — <span style='color:var(--truth)'>true program</span> "
              "vs <span style='color:var(--found)'>recovered</span>:</p>", track_svg(tt, tf, size),
              "<div class='row'>",
              f"<div><h3>True program</h3><div class='tree'>{tree_svg(truth)}</div><code>{html.escape(r['truth'])}</code></div>",
              f"<div><h3>Recovered</h3><div class='tree'>{tree_svg(found)}</div><code>{html.escape(r['map'])}</code></div>",
              "</div></div>"]
    n_ok = sum(r["map_equiv"] for r in rec["results"])
    n_search = sum(not r["search_ok"] for r in rec["results"])
    length = rec["inference"]["iterations"]
    tail = (f" In the other {n_search}, the model prefers the true program, but the search "
            + ("(shorter than the full run length) " if length < 40000 else f"(full length, {length} iterations) ")
            + "did not reach it. Proposals that add a motion leaf together with its body are the fix on the "
            "search side." if n_search else "")
    h += [f"<p><b>{n_ok} of {len(rec['results'])}</b> recovered exactly (same rendering in every frame).{tail}</p>",
          "<h2>3. Where this stands</h2>",
          "<div class='note'>The model, renderer and inference run end to end on the GRACE locomotion videos "
          "(25 actors, 26 manners; training / validation / sealed test split), and hyperparameters are being selected "
          "on validation actors only. In fits to the real videos so far, the programs capture the body's static "
          "configuration, but they rarely use motion: the figure is centred, and limbs move less than one board cell, "
          "while program motion moves whole bodies cell by cell. Representing limb motion at that scale (a finer board, "
          "or motion of parts relative to the body) is the next step.</div>",
          "</main>"]
    page = out / "motion_demo.html"
    page.write_text("\n".join(h), encoding="utf-8")
    return page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery", default="results/grace_v3/motion_demo/recovery.json")
    args = ap.parse_args()
    path = Path(args.recovery)
    print(f"wrote {write(json.loads(path.read_text()), path.parent)}")


if __name__ == "__main__":
    main()
