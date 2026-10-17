"""Qualitative inspection of inferred programs: purely diagnostic, changes nothing.

    python -m grace_lot.analysis.inspect_programs --run results/grace_v2/select/lambda1_beta250

For each manner, read the posterior already produced on the training actors
and:
  * score every distinct sampled program on those same training videos:
    log posterior = log prior + log likelihood, both from the run's own config;
  * group programs into *modes*: programs whose renderings are identical
    are indistinguishable by any likelihood. Rank modes by their best member's
    log posterior and keep the top 5;
  * render each mode (``lot_trajectory``), compare it with the observed motion,
    and write side-by-side GIFs (observed | top-1 | top-2 | top-3), static
    strips, and an HTML index.

"Observed" is the medoid training actor (the one whose field sequence is
closest to the training mean), shown as a stick figure over its field. The
training mean is also shown. No test actor, no human data, and no setting is
read or changed.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.inference.grammar import Grammar
from grace_lot.inference.likelihood import TrajectoryLikelihood
from grace_lot.inference.prior import SizePrior
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.distance import DISTANCES, euclid
from grace_lot.lot_adapter.program import LEAF, depth, parse_program, saved_lot, size

TOP_MODES = 5
SCALE = 16          # pixels per field cell in the GIFs


def tree_text(prog) -> str:
    """Indented tree, with dashed edges listed by path."""
    lines = []

    def walk(p, path, indent):
        label = str(p) if p.kind == LEAF else (p.kind + (str(list(p.params)) if p.params else ""))
        lines.append("  " * indent + f"{label}   [{'.' + '.'.join(map(str, path)) if path else '.'}]")
        for i, c in enumerate(p.children):
            walk(c, path + (i,), indent + 1)

    walk(prog.tree, (), 0)
    for a, b in prog.dashed:
        fmt = lambda x: "." + ".".join(map(str, x))
        lines.append(f"dashed  {fmt(a)} --> {fmt(b)}")
    return "\n".join(lines)


def tree_svg(prog) -> str:
    """The program drawn as a tree: solid composition edges, dashed gate edges as arrows.

    Motion (dynamic) leaves are orange, static leaves blue. Nodes in a template (any child but
    the last of a plan or body) have a dashed outline: they place or turn the content rather
    than draw.
    """
    pos, nxt = {}, [0]

    def place(p, path, d):
        if not p.children:
            pos[path] = (nxt[0], d)
            nxt[0] += 1
            return
        for i, c in enumerate(p.children):
            place(c, path + (i,), d + 1)
        pos[path] = (sum(pos[path + (i,)][0] for i in range(len(p.children))) / len(p.children), d)
    place(prog.tree, (), 0)

    nodes, templated = {}, set()

    def collect(p, path, in_template):
        nodes[path] = p
        if in_template:
            templated.add(path)
        for i, c in enumerate(p.children):
            is_template = p.kind in ("plan", "body") and i < len(p.children) - 1
            collect(c, path + (i,), in_template or is_template)
    collect(prog.tree, (), False)

    DX, DY, PX, PY = 92, 64, 50, 46
    xy = {k: (PX + x * DX, PY + 14 + d * DY) for k, (x, d) in pos.items()}
    W = max(x for x, _ in pos.values()) * DX + 2 * PX
    H = max(d for _, d in pos.values()) * DY + 2 * PY + 24
    out = [f"<svg viewBox='0 0 {W:.0f} {H:.0f}' width='{W:.0f}' style='max-width:100%;height:auto' "
           f"xmlns='http://www.w3.org/2000/svg' font-family='system-ui,sans-serif'>",
           "<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
           "orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#b8336a'/></marker></defs>"]
    for path in nodes:
        if path:
            (x1, y1), (x2, y2) = xy[path[:-1]], xy[path]
            out.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='#c9c6bd' stroke-width='2'/>")
    for a, b in prog.dashed:
        (x1, y1), (x2, y2) = xy[a], xy[b]
        lift = 30 + abs(x2 - x1) * 0.12
        my = min(y1, y2) - lift
        out.append(f"<path d='M{x1:.0f},{y1 - 18:.0f} C{x1:.0f},{my:.0f} {x2:.0f},{my:.0f} {x2:.0f},{y2 - 20:.0f}' "
                   f"fill='none' stroke='#b8336a' stroke-width='2' stroke-dasharray='6 4' marker-end='url(#a)'/>")
    for path, p in nodes.items():
        x, y = xy[path]
        dash = " stroke-dasharray='4 3'" if path in templated else ""
        if p.kind == LEAF:
            name, ori, dyn = p.params
            fill, stroke = ("#fbe1c6", "#d9761f") if dyn else ("#dbe7fb", "#2f6fdb")
            out.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='15' fill='{fill}' stroke='{stroke}' stroke-width='2'{dash}/>")
            label = f"{name}/{ori}{'*' if dyn else ''}"
        else:
            out.append(f"<rect x='{x - 22:.0f}' y='{y - 14:.0f}' width='44' height='28' rx='7' fill='#f1efe9' "
                       f"stroke='#6b6a64' stroke-width='2'{dash}/>")
            label = "[" + ",".join(f"{v:g}" for v in p.params) + "]" if p.params else ""   # body parameters
            out.append(f"<text x='{x:.0f}' y='{y + 4:.0f}' text-anchor='middle' font-size='10' fill='#1d1d1b'>"
                       f"{html.escape(p.kind.replace('serial_', 's_')[:8])}</text>")
        if label:
            out.append(f"<text x='{x:.0f}' y='{y + 31:.0f}' text-anchor='middle' font-size='10.5' fill='#444'>"
                       f"{html.escape(label)}</text>")
    out.append("</svg>")
    return "".join(out)


def _features(y: np.ndarray) -> dict:
    """Simple descriptors of a (T, H, W) field sequence, used for the failure-mode summary."""
    H = y.shape[1]
    mass = y.sum(axis=(1, 2)) + 1e-9
    row_c = (y.sum(axis=2) * np.arange(H)).sum(axis=1) / mass        # vertical centre of mass per frame
    return {"mean": y.mean(0), "dyn": y - y.mean(0), "row_c": row_c,
            "motion": float(np.abs(np.diff(y, axis=0)).mean())}


def compare(obs: np.ndarray, pred: np.ndarray) -> dict:
    fo, fp = _features(obs), _features(pred)
    return {"D": euclid(obs, pred),
            "D_spatial": float(((fo["mean"] - fp["mean"]) ** 2).mean()),       # time-averaged layout
            "D_temporal": float(((fo["dyn"] - fp["dyn"]) ** 2).mean()),        # frame-to-frame changes
            "centre_err": float(np.abs(fo["row_c"] - fp["row_c"]).mean()),    # global vertical trajectory (cells)
            "motion_obs": fo["motion"], "motion_pred": fp["motion"]}


def analyse_manner(job) -> dict:
    cfg, run_dir, manner, videos = job
    inf, pre, lot = cfg["inference"], cfg["preprocess"], cfg["lot"]
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(lot)
    trajs = [load_trajectory(cfg, v) for v in videos]
    obs = [t.fields for t in trajs]
    mean_obs = np.mean(obs, axis=0)
    medoid = int(np.argmin([euclid(o, mean_obs) for o in obs]))
    grammar = Grammar.from_config(lot)
    renderer = LotRenderer(pre["frames"], pre["field_size"])
    prior = SizePrior.from_config(grammar, inf)
    lik = TrajectoryLikelihood(obs, renderer, DISTANCES[inf["distance"]], inf["beta"], inf.get("temporal_weight", 0.0))
    post = json.loads((run_dir / "posteriors" / f"{manner}.json").read_text())

    modes = defaultdict(lambda: {"mass": 0.0, "members": []})
    for s, w in post["posterior"]:
        p = parse_program(s)
        y = renderer(p)
        key = np.round(y, 5).tobytes()
        lp, ll = prior(p), lik(p)
        modes[key]["mass"] += w
        modes[key]["members"].append((lp + ll, lp, ll, p))
    ranked = []
    for m in modes.values():
        m["members"].sort(key=lambda x: -x[0])
        best = m["members"][0]
        ranked.append((best, m["mass"], len(m["members"])))
    ranked.sort(key=lambda x: -x[0][0])

    out = []
    for (post_lp, lp, ll, p), mass, n_members in ranked[:TOP_MODES]:
        y = renderer(p)
        nodes, edges, dep = size(p.tree), len(p.dashed), depth(p.tree)
        out.append({
            "program": str(p), "tree": tree_text(p), "tree_svg": tree_svg(p), "log_post": post_lp, "log_prior": lp, "log_lik": ll,
            "mode_mass": mass, "mode_programs": n_members,
            "D_train_mean": float(np.mean([euclid(o, y) for o in obs])),
            "D_medoid": euclid(obs[medoid], y), **{f"cmp_{k}": v for k, v in compare(mean_obs, y).items()},
            "nodes": nodes, "depth": dep, "dashed": edges,
            "at_node_bound": nodes == lot["max_nodes"], "at_depth_bound": dep == lot["max_depth"],
            "at_dashed_bound": edges == lot["max_dashed"], "render": y,
        })
    best_map = parse_program(post["map"])
    map_render = renderer(best_map)
    map_mode = next((i + 1 for i, m in enumerate(out) if np.allclose(m["render"], map_render)), None)
    map_lp = prior(best_map) + lik(best_map)
    return {"map_mode": map_mode, "map_log_post": map_lp,"manner": manner, "videos": [v.stem for v in videos], "medoid": videos[medoid].stem,
            "medoid_traj": trajs[medoid], "mean_obs": mean_obs, "obs_medoid": obs[medoid],
            "modes": out, "n_modes": len(ranked), "n_programs": len(post["posterior"]),
            "map_program": post["map"],
            "mass_top5": float(sum(m["mode_mass"] for m in out)),
            "r_hat": post["r_hat_log_post"]}


# ---- figures -------------------------------------------------------------------

def _panel(field: np.ndarray, cmap) -> np.ndarray:
    rgb = (cmap(np.clip(field, 0, 1))[:, :, :3] * 255).astype(np.uint8)
    return np.kron(rgb, np.ones((SCALE, SCALE, 1), dtype=np.uint8))


def _skeleton_overlay(img: np.ndarray, joints: np.ndarray, connections, pre: dict) -> np.ndarray:
    import cv2
    img = img.copy()
    (x0, x1), (y0, y1) = pre["window_x"], pre["window_y"]
    R = img.shape[0]
    pts = np.stack([(joints[:, 0] - x0) / (x1 - x0) * R, (joints[:, 1] - y0) / (y1 - y0) * R], 1)
    for a, b in connections:
        cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), (80, 220, 255), 2, cv2.LINE_AA)
    return img


def side_by_side_gif(res: dict, cfg: dict, path: Path, fps: int = 4) -> None:
    """Observed (medoid actor: field + skeleton) | top-1 | top-2 | top-3, with the same frames, framing and speed."""
    import matplotlib
    from PIL import Image, ImageDraw
    gray = matplotlib.colormaps["gray"]            # one scale: yin black, unknown grey, yang white
    traj, pre = res["medoid_traj"], cfg["preprocess"]
    frames = []
    titles = ["observed (medoid)"] + [f"LoT top-{i + 1}" for i in range(min(3, len(res["modes"])))]
    for t in range(len(res["obs_medoid"])):
        panels = [_skeleton_overlay(_panel(res["obs_medoid"][t], gray), traj.joints[t], traj.connections, pre)]
        panels += [_panel(m["render"][t], gray) for m in res["modes"][:3]]
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


def strip_png(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("observed: medoid actor", res["obs_medoid"], "gray"), ("observed: training mean", res["mean_obs"], "gray")]
    rows += [(f"top-{i + 1}  logpost {m['log_post']:.1f}", m["render"], "gray") for i, m in enumerate(res["modes"])]
    T = len(res["obs_medoid"])
    fig, axes = plt.subplots(len(rows), T, figsize=(T * 0.55, len(rows) * 0.62 + 0.3), squeeze=False)
    for r, (label, y, cmap) in enumerate(rows):
        for t in range(T):
            axes[r, t].imshow(y[t], cmap=cmap, vmin=0, vmax=1); axes[r, t].axis("off")
        axes[r, 0].text(-0.5, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=6)
    fig.subplots_adjust(left=0.18, wspace=0.05, hspace=0.15)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ---- index -------------------------------------------------------------------------

def write_index(results, out: Path, run_dir: Path, cfg: dict) -> None:
    lot = cfg["lot"]
    css = ("body{font:14px system-ui,sans-serif;margin:24px;max-width:1300px;color:#222}"
           "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}"
           "th{background:#f3f3f3}td.l{text-align:left}pre{font-size:11px;background:#f8f8f8;padding:6px;overflow-x:auto}"
           ".bound{color:#b00;font-weight:600}section{border-top:2px solid #ddd;margin-top:28px;padding-top:8px}"
           "nav a{margin-right:10px}")
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>LoT program inspection</title><style>{css}</style></head><body>",
         "<h1>Inferred LoT programs: qualitative inspection</h1>",
         f"<p>Run: <code>{html.escape(str(run_dir))}</code> (posteriors from the 15 training actors). Search bounds: "
         f"{lot['max_nodes']} nodes, depth {lot['max_depth']}, {lot['max_dashed']} dashed edges. Modes are groups of "
         "posterior programs with identical renderings, ranked by their best member's log posterior. "
         "Observed = medoid training actor. No test actors and no human data are used.</p>",
         "<p><b>Qualitative failure-mode summary:</b> <a href='failure_modes.md'>failure_modes.md</a></p>"
         if (out / "failure_modes.md").exists() else "",
         "<nav>" + "".join(f"<a href='#{r['manner']}'>{r['manner']}</a>" for r in results) + "</nav>",
         "<h2>Overview</h2><table><tr><th>manner</th><th>R-hat</th><th>distinct programs</th><th>modes</th>"
         "<th>top-5 mode mass</th><th>top-1 log post</th><th>top-1 D</th><th>top-1 nodes</th><th>top-1 dashed</th>"
         "<th>top-1 at a bound</th><th>gap to top-2</th></tr>"]
    for r in results:
        m = r["modes"][0]
        gap = m["log_post"] - r["modes"][1]["log_post"] if len(r["modes"]) > 1 else float("nan")
        bound = ", ".join(k for k, f in (("nodes", m["at_node_bound"]), ("depth", m["at_depth_bound"]),
                                          ("dashed", m["at_dashed_bound"])) if f) or "-"
        h.append(f"<tr><td class='l'><a href='#{r['manner']}'>{r['manner']}</a></td><td>{r['r_hat']:.2f}</td>"
                 f"<td>{r['n_programs']}</td><td>{r['n_modes']}</td><td>{r['mass_top5']:.2f}</td><td>{m['log_post']:.1f}</td>"
                 f"<td>{m['D_train_mean']:.4f}</td><td>{m['nodes']}</td><td>{m['dashed']}</td>"
                 f"<td class='{'bound' if bound != '-' else ''}'>{bound}</td><td>{gap:.2f}</td></tr>")
    h.append("</table>")
    for r in results:
        h += [f"<section id='{r['manner']}'><h2>{r['manner']}</h2>",
              f"<p>Training actors: {', '.join(r['videos'])}. Medoid shown: <b>{r['medoid']}</b>. "
              f"R-hat {r['r_hat']:.2f}; {r['n_programs']} distinct sampled programs in {r['n_modes']} rendering-modes; "
              f"the top 5 modes hold {r['mass_top5']:.2f} of posterior mass.</p>",
              f"<img src='gifs/{r['manner']}.gif' alt='side by side'><br>",
              f"<img src='strips/{r['manner']}.png' style='max-width:100%' alt='strip'>",
              "<table><tr><th>rank</th><th>log post</th><th>log prior</th><th>log lik</th><th>mode mass</th>"
              "<th>programs in mode</th><th>D (train mean)</th><th>D spatial</th><th>D temporal</th>"
              "<th>centre err (cells)</th><th>motion obs / LoT</th><th>nodes</th><th>depth</th><th>dashed</th></tr>"]
        for i, m in enumerate(r["modes"]):
            cls = lambda f: " class='bound'" if f else ""
            h.append(f"<tr><td>{i + 1}</td><td>{m['log_post']:.2f}</td><td>{m['log_prior']:.2f}</td><td>{m['log_lik']:.2f}</td>"
                     f"<td>{m['mode_mass']:.3f}</td><td>{m['mode_programs']}</td><td>{m['D_train_mean']:.4f}</td>"
                     f"<td>{m['cmp_D_spatial']:.4f}</td><td>{m['cmp_D_temporal']:.4f}</td><td>{m['cmp_centre_err']:.2f}</td>"
                     f"<td>{m['cmp_motion_obs']:.3f} / {m['cmp_motion_pred']:.3f}</td>"
                     f"<td{cls(m['at_node_bound'])}>{m['nodes']}</td><td{cls(m['at_depth_bound'])}>{m['depth']}</td>"
                     f"<td{cls(m['at_dashed_bound'])}>{m['dashed']}</td></tr>")
        h.append("</table>")
        for i, m in enumerate(r["modes"]):
            h.append(f"<details{' open' if i == 0 else ''}><summary>top-{i + 1}: <code>{html.escape(m['program'])}</code></summary>"
                     f"{m.get('tree_svg', '')}<details><summary>as text (with paths)</summary>"
                     f"<pre>{html.escape(m['tree'])}</pre></details></details>")
        where = (f"renders identically to top-{r['map_mode']}" if r["map_mode"]
                 else "<b>renders differently from all top-5 modes</b>")
        h.append(f"<p>Sampler MAP (best state visited by any chain, log post {r['map_log_post']:.2f}): "
                 f"<code>{html.escape(r['map_program'])}</code>; {where}.</p></section>")
    h.append("</body></html>")
    page = "\n".join(h)
    # Embed the images, so the page is a single self-contained file
    import base64
    for sub, mime in (("gifs", "image/gif"), ("strips", "image/png")):
        for f in (out / sub).iterdir():
            data = base64.b64encode(f.read_bytes()).decode()
            page = page.replace(f"src='{sub}/{f.name}'", f"src='data:{mime};base64,{data}'")
    (out / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/grace_v2/select/lambda1_beta250")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--recovery", help="a recovery.json: link its synthetic recovery page from the index")
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "inspection"
    (out / "gifs").mkdir(parents=True, exist_ok=True)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    run_cfg = json.loads((run_dir / "config.json").read_text())
    for k in ("preprocess", "lot", "inference", "split"):
        cfg[k] = run_cfg[k]
    cfg["lot"] = saved_lot(cfg["lot"])
    train = split_videos(cfg["data"]["raw_dir"], cfg["split"], "train")
    sealed = set(cfg["split"]["test_actors"])
    assert not any(v.actor in sealed for vs in train.values() for v in vs)
    jobs = [(cfg, run_dir, m, train[m]) for m in sorted(train)]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = pool.map(analyse_manner, jobs)
    for r in results:
        side_by_side_gif(r, cfg, out / "gifs" / f"{r['manner']}.gif")
        strip_png(r, out / "strips" / f"{r['manner']}.png")
    write_index(results, out, run_dir, cfg)
    if args.recovery:                          # synthetic checks: known programs, their videos, what was found
        from grace_lot.analysis.recovery_page import write_page
        import os
        page = write_page(json.loads(Path(args.recovery).read_text()), Path(args.recovery).parent)
        link = os.path.relpath(page, out).replace(os.sep, "/")
        index = out / "index.html"
        text = index.read_text(encoding="utf-8")
        text = text.replace("<h2>Overview</h2>", f"<p><b>Synthetic recovery checks</b> (known programs as synthetic "
                            f"videos, true vs recovered trees): <a href='{link}'>{link}</a></p><h2>Overview</h2>", 1)
        index.write_text(text, encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k not in ("medoid_traj", "mean_obs", "obs_medoid")}
            | {"modes": [{k: v for k, v in m.items() if k != "render"} for m in r["modes"]]} for r in results]
    (out / "inspection.json").write_text(json.dumps(slim, indent=1, default=float))
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
