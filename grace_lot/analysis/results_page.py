"""One page of results for a talk: fitted programs, held-out scores, gestures and human similarity.

    python -m grace_lot.analysis.results_page --run results/grace_v7_parts/all26_current --out results/presentation
        [--compare old=results/.../w32_l1.0_cap32 ...]

Reads what the other analyses wrote into ``--run`` (``gallery/``, ``predictive.json``, ``motion_check.json``,
``gestures/``, ``analysis/``) and the held-out scores of the ``--compare`` runs, and writes ``<out>/index.html`` with its GIFs
beside it (``gallery/``, ``gestures/``), ready to publish as one artifact. Sections whose inputs are missing are
left out. Reads no test actors and fits nothing.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np

from grace_lot.analysis.gestures import CHANNEL_COLOURS, double_centred

PART_COLOURS = {"head": "#999999", "torso": "#e8e8e8", "left_arm": "#3373ff", "right_arm": "#ff8c1a",
                "left_leg": "#1ad9e6", "right_leg": "#f22633"}
E = html.escape


def css() -> str:
    return """
/* Layout: one reading column; galleries break out to a wider grid of cards. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #1b1f27; --muted: #5d6573; --rule: #dde1e7;
  --accent: #0f8a96; --good: #2b8a3e; --board: #0b0d10; --chip: #eef1f4;
  --display: "Schibsted Grotesk", "Segoe UI", system-ui, sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111317; --surface: #1a1d23; --ink: #e7e9ee; --muted: #9aa3b1; --rule: #2c313a;
  --accent: #39b8c4; --good: #5fbf72; --board: #000000; --chip: #23272f; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111317; --surface: #1a1d23; --ink: #e7e9ee; --muted: #9aa3b1; --rule: #2c313a;
  --accent: #39b8c4; --good: #5fbf72; --board: #000000; --chip: #23272f; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 16px/1.55 var(--body); }
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; line-height: 1.2; }
h1 { font-size: clamp(28px, 4vw, 40px); margin: 0 0 8px; letter-spacing: -0.01em; }
h2 { font-size: 26px; margin: 56px 0 8px; padding-top: 24px; border-top: 1px solid var(--rule); }
h3 { font-size: 18px; margin: 0; text-transform: capitalize; }
p, li { max-width: 70ch; }
.lede { color: var(--muted); font-size: 18px; max-width: 72ch; }
.eyebrow { font: 600 12px/1 var(--body); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
.findings { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin: 24px 0; }
.finding { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; min-width: 0; }
.finding b { display: block; font: 600 22px/1.2 var(--display); margin-bottom: 4px; font-variant-numeric: tabular-nums; }
.finding span { color: var(--muted); font-size: 14px; }
.wide { display: grid; gap: 20px; margin-top: 20px; } .wide .card img { width: auto; max-width: 100%; }
.grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); margin-top: 20px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; min-width: 0;
  display: flex; flex-direction: column; gap: 8px; }
.card img { width: 100%; image-rendering: pixelated; border-radius: 4px; background: var(--board); }
.meta { color: var(--muted); font-size: 13px; }
code, pre { font-family: var(--mono); font-size: 12px; }
pre { white-space: pre-wrap; word-break: break-word; background: var(--chip); padding: 8px 10px; border-radius: 6px; margin: 0; }
details summary { cursor: pointer; color: var(--accent); font-size: 14px; }
details .tree { overflow-x: auto; background: #fff; border-radius: 6px; margin-top: 6px; }
.table { overflow-x: auto; margin: 16px 0; }
table { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 12px; border-bottom: 1px solid var(--rule); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { font-weight: 600; color: var(--muted); font-size: 13px; }
tr.hi td { font-weight: 600; }
.stack { display: grid; gap: 28px; }
.partview { min-width: 0; } .partview img { display: block; width: 100%; max-width: 100%; height: auto; margin-top: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { background: var(--chip); border-radius: 999px; padding: 2px 10px; font-size: 12px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; color: var(--muted); }
.note { color: var(--muted); font-size: 14px; }
figure { margin: 16px 0; } figcaption { color: var(--muted); font-size: 13px; margin-top: 6px; }
svg text { fill: var(--ink); font-family: var(--body); }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""


def legend(colours: dict) -> str:
    return "<div class=legend>" + "".join(f"<span><i class=sw style='background:{c}'></i>{E(n.replace('_', ' '))}</span>"
                                         for n, c in colours.items()) + "</div>"


def rgb(c) -> str:
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in c)


def gallery_cards(run: Path, out: Path, sub: str = "gallery") -> str:
    """One card per manner of ``run``'s gallery; its GIFs are copied to ``<out>/<sub>/``."""
    g = json.loads((run / "gallery" / "gallery.json").read_text())
    (out / sub).mkdir(parents=True, exist_ok=True)
    cards = []
    for m, r in g.items():
        shutil.copy(run / "gallery" / r["gif"], out / sub / f"{m}.gif")
        best = r["top"][0]
        mapping = "".join(f"<span class=chip>out{k} &rarr; {E(v.replace('_', ' '))}</span>" for k, v in sorted(r["mapping"].items()))
        others = "".join(f"<p class=meta>#{i} log posterior {t['log_post']:.1f}</p><pre>{E(t['program'])}</pre>"
                         for i, t in enumerate(r["top"][1:], 2))
        cards.append(f"""<article class=card id="{sub}-{m}"><h3>{E(m)}</h3>
<img src="{sub}/{m}.gif" alt="{E(m)}: the video, its body parts on the board, and the best program's outputs" loading=lazy>
<p class=meta>Video ({E(r['medoid'])}) &middot; body parts on the 12&times;12 board &middot; best program's outputs</p>
<pre>{E(best['program'])}</pre>
<p class=meta>log posterior {best['log_post']:.1f} &middot; {best['nodes']} nodes &middot; {r['unique_programs']} distinct programs sampled &middot; R&#770; {r['r_hat']:.2f}</p>
<details><summary>Tree, output matching and runners-up</summary><div class=tree>{best['svg']}</div>
<div class=chips>{mapping}</div>{others}</details></article>""")
    return f"<div class=grid>{''.join(cards)}</div>"


def gallery_section(run: Path, out: Path) -> str:
    return f"""<h2 id=programs>Fitted programs</h2>
<p>For each manner, the program with the highest posterior score among the most frequently sampled, fitted to the
training actors only. In each animation: the medoid training video at the 16 frames the model sees, its body parts on the
board, and the program's outputs, each tinted by the body part it is matched to. Dark grey cells are cells the
program makes no claim about; black cells are claimed empty.</p>{legend(PART_COLOURS)}
{gallery_cards(run, out)}"""


def heldout_section(run: Path, compare: dict) -> str:
    rows = []
    for name, path in compare.items():
        p = json.loads((Path(path) / "predictive.json").read_text())
        rows.append((name, p))
    manners = rows[0][1]["manners"] if rows else []
    head = "".join(f"<th>{E(m)}</th>" for m in manners)
    body = "".join(f"<tr class={'hi' if n.startswith('current') else ''}><td>{E(n)}</td><td>{p['own_lpd_total']:.1f}</td>"
                   f"<td>{p['recognition_top1']:.2f}</td>" + "".join(f"<td>{p['own_lpd_by_manner'][m]:.1f}</td>" for m in manners)
                   + "</tr>" for n, p in rows)
    s = (f"<h2 id=heldout>Held-out validation actors</h2><p>Five manners, fitted to the 15 training actors in each "
         f"language version and scored on the 5 validation actors (log predictive density, higher is better; "
         f"recognition among the five manners). The sealed test actors were not used.</p>"
         f"<div class=table><table><tr><th>language</th><th>density</th><th>recognition</th>{head}</tr>{body}</table></div>")
    pred = run / "predictive.json"
    if pred.exists():
        p = json.loads(pred.read_text())
        lpd, n = np.array(p["lpd"]), len(p["manners"])
        truth = np.array([p["manners"].index(v.split("_", 1)[1]) for v in p["videos"]])
        rank = (lpd > lpd[truth, np.arange(len(truth))]).sum(axis=0) + 1
        s += (f"<p>All {n} manners (the lighter fit used below): recognition of the {len(truth)} validation videos "
              f"among {n} manners is <b>{p['recognition_top1']:.2f}</b> top-1 and {np.mean(rank <= 5):.2f} top-5 "
              f"(chance {1 / n:.3f} and {5 / n:.2f}); median rank of the true manner {np.median(rank):.0f}.</p>")
    kb = run / "kinematic_baselines.json"
    if kb.exists():
        k = json.loads(kb.read_text())
        names = {"prototype_raw": "nearest manner average", "1nn_raw": "nearest training video",
                 "prototype_standardised": "nearest manner average, standardised features",
                 "1nn_standardised": "nearest training video, standardised features"}
        body = f"<tr class=hi><td>LoT programs</td><td>{k['lot']['top1']:.2f}</td><td>{k['lot']['top5']:.2f}</td>" \
               f"<td>{k['lot']['mean_rank']:.1f}</td><td></td></tr>"
        body += "".join(f"<tr><td>{names[b]}</td><td>{k[b]['top1']:.2f}</td><td>{k[b]['top5']:.2f}</td>"
                        f"<td>{k[b]['mean_rank']:.1f}</td><td>{k[b]['vs_lot']['only_first']} / "
                        f"{k[b]['vs_lot']['only_second']} (p {k[b]['vs_lot']['p']:.3f})</td></tr>" for b in names)
        best = max(names, key=lambda b: k[b]["top1"])
        s += (f"<h3>Against kinematic baselines</h3><p>The same {k['n']} validation videos, recognised without "
              f"programs: each video is compared directly with the training actors' part features (the input the "
              f"LoT sees), with the fit's change term and phase shifts. Last column: videos only the LoT gets right "
              f"/ only the baseline gets right (exact McNemar test).</p><div class=table><table><tr><th>method</th>"
              f"<th>top-1</th><th>top-5</th><th>mean rank</th><th>LoT only / baseline only</th></tr>{body}"
              f"</table></div><p>The programs recognise well above chance but below simple templates "
              f"({k['lot']['top1']:.2f} vs up to {k[best]['top1']:.2f} top-1); top-5 is on a par. Static programs "
              f"keep a manner's typical posture and lose the motion detail that templates keep.</p>")
    return s


def heatmap_svg(M: np.ndarray, labels: list, title: str) -> str:
    """Double-centred scores as a diverging heatmap; the matched (diagonal) cells outlined."""
    D = double_centred(M)
    lim = np.abs(D).max() or 1.0
    n, c, pad = len(labels), 14, 92
    cells = []
    for i in range(n):
        for j in range(n):
            v = D[i, j] / lim
            col = (1.0, 1 - v * 0.75, 1 - v * 0.85) if v >= 0 else (1 + v * 0.8, 1 + v * 0.55, 1.0)
            stroke = " stroke='currentColor' stroke-width='1.5'" if i == j else ""
            cells.append(f"<rect x='{pad + j * c}' y='{pad + i * c}' width='{c}' height='{c}' fill='{rgb(col)}'{stroke}>"
                         f"<title>{E(labels[i])} programs, {E(labels[j])} gesture: {D[i, j]:.1f}</title></rect>")
    ticks = "".join(f"<text x='{pad - 4}' y='{pad + i * c + 10}' font-size='10' text-anchor='end'>{E(l)}</text>"
                    f"<text transform='translate({pad + i * c + 10},{pad - 4}) rotate(-60)' font-size='10'>{E(l)}</text>"
                    for i, l in enumerate(labels))
    size = pad + n * c + 4
    return (f"<figure><div class=table style='color:var(--ink)'><svg viewBox='0 0 {size} {size}' width='{size}' "
            f"height='{size}' role=img aria-label='{E(title)}'>{''.join(cells)}{ticks}</svg></div>"
            f"<figcaption>{E(title)}. Rows: the manner whose locomotion programs score the gesture; columns: the "
            f"gesture. Red: better than the row and column averages; blue: worse. Outlined: each gesture under its own "
            f"manner.</figcaption></figure>")


def gesture_section(run: Path, out: Path) -> str:
    g = json.loads((run / "gestures" / "gestures.json").read_text())
    manners, lot, feat = g["manners"], g["lot"], g["features"]
    (out / "gestures").mkdir(parents=True, exist_ok=True)

    def row(name, s):
        return (f"<tr><td>{name}</td><td>{s['mean_rank_own_manner']:.1f}</td><td>{s['top1']:.2f}</td>"
                f"<td>{s['top5']:.2f}</td><td>{s['p_permutation']:.3f}</td></tr>")
    n = len(manners)
    table = (f"<div class=table><table><tr><th>score</th><th>mean rank of own manner (chance {(n + 1) / 2:.1f})</th>"
             f"<th>top-1 (chance {1 / n:.2f})</th><th>top-5 (chance {5 / n:.2f})</th><th>permutation p</th></tr>"
             + row("locomotion programs (LoT)", lot) + row("raw features (nearest manner mean)", feat) + "</table></div>")
    if "humans" in g:
        h, hl, hf = g["humans"]["ratings"], g["humans"]["lot"], g["humans"]["features"]
        people = np.mean([v["own"] > v["foil_rating"] for v in h.values()])
        table += (f"<h3 style='text-transform:none;margin-top:24px'>Against people (GRACE Norming Study 1)</h3>"
                  f"<p>{len(h)} gestures, each rated by participants for how well it matches its own manner and one "
                  f"foil manner (1&ndash;7; ratings of test actors' videos left out). Participants rated the own manner "
                  f"higher for {people:.0%} of gestures (mean {np.mean([v['own'] for v in h.values()]):.1f} vs "
                  f"{np.mean([v['foil_rating'] for v in h.values()]):.1f}).</p><div class=table><table><tr><th>score</th>"
                  f"<th>own manner beats foil (chance 0.50)</th><th>&rho; with participants' margin</th><th>p</th></tr>"
                  + "".join(f"<tr><td>{n}</td><td>{x['own_beats_foil']:.2f}</td><td>{x['spearman_with_humans']:.2f}</td>"
                            f"<td>{x['p_spearman']:.3f}</td></tr>" for n, x in (("locomotion programs (LoT)", hl),
                                                                                 ("raw features", hf)))
                  + "</table></div>")
    cards = []
    for i, m in enumerate(manners):
        b = g["best_for_own_gesture"][m]
        shutil.copy(run / "gestures" / b["gif"], out / "gestures" / f"{m}.gif")
        outs = "".join(f"<span class=chip>{E((o['body_part'] or '-').replace('_', ' '))} &harr; {E(o['hand_channel'] or '-')}</span>"
                       for o in b["outputs"].values() if o["body_part"] and o["hand_channel"])
        cards.append(f"""<article class=card><h3>{E(m)}</h3>
<img src="gestures/{m}.gif" alt="{E(m)} gesture, its hand channels, and a {E(m)} locomotion program on it" loading=lazy>
<p class=meta>Gesture &middot; hand channels &middot; the {E(m)} program that best explains it. Own manner ranked
<b>{lot['rank_of_own_manner'][i]}</b> of {n} (features: {feat['rank_of_own_manner'][i]}).</p>
<details><summary>Program and body part &harr; hand part</summary><pre>{E(b['program'])}</pre><div class=chips>{outs}</div></details></article>""")
    found = lot["p_permutation"] < 0.05
    verdict = ("The locomotion programs do pick out the gestures' own manners more often than chance."
               if found else
               "The locomotion programs do not find the resemblance: a gesture's own manner ranks no better than chance "
               "among the 26, and no better than raw features do. The likely reason is the previous section: the "
               "programs mostly encode posture, where the body parts sit, and a seated hand mime shares little posture "
               "with a walking body. What a gesture and its manner share is the pattern of motion (a rhythm, a "
               "direction, a bounce), which these programs largely leave out. This is a hypothesis, not yet tested.")
    pc = g["part_channel_pairs"]
    C = np.array(pc["counts"])
    pairs = ("<div class=table><table><tr><th>body part \\ hand part</th>" + "".join(f"<th>{E(c)}</th>" for c in pc["channels"])
             + "</tr>" + "".join(f"<tr><td>{E(p.replace('_', ' '))}</td>" + "".join(f"<td>{v}</td>" for v in C[k]) + "</tr>"
                                 for k, p in enumerate(pc["parts"])) + "</table></div>")
    chans = {name: rgb(c) for name, c in zip(pc["channels"], CHANNEL_COLOURS)}
    return f"""<h2 id=gestures>Do locomotion programs explain the gestures?</h2>
<p>Each manner has a gesture video in which the experimenter, seated, mimes it with her hands. The hands were
detected (21 points each), framed so that the hands' extent is one body height, and split into six channels, three
per hand. Each gesture was then scored under every manner's locomotion posterior, fitted to bodies only, with
outputs matched to hand channels anonymously, as they are matched to body parts. No program was fitted to a gesture,
and nothing was tuned on these scores. Scores are double-centred, so a gesture that is easy for every manner, or a
manner whose programs are broad, does not count as a match.</p>{table}
<p>{verdict}</p>
{heatmap_svg(np.array(g['lpd']), manners, 'Gesture scores under each manner (log predictive density, double-centred)')}
<p>Body part &harr; hand part: for each manner's program, the body part each output explains on the training actor
and the hand channel it explains in the gesture (outputs drawn identically to another are left out).</p>{pairs}
{legend(chans)}<div class=grid>{''.join(cards)}</div>"""


def motion_section(path: Path, fix: Path = None, control: Path = None) -> str:
    """Why the programs are static (``motion_check.py``), and the timing fix tested against its control."""
    r = json.loads(path.read_text())
    tested = ""
    if fix and control:
        f, c = (json.loads((d / "predictive.json").read_text())["own_lpd_by_manner"] for d in (fix, control))
        tf, tc = sum(f.values()), sum(c[m] for m in f)
        tested = (f"<p>A first fix was tested: a likelihood that allows for timing error (each output's variance grows "
                  f"with its speed, for a timing error of about one frame), which makes the moving actor win in 4 of "
                  f"5 manners. A fit with it on the same {len(f)} manners predicted the validation actors no better "
                  f"(held-out density {tf:.1f} vs {tc:.1f} without it), and moving programs did not become consistently "
                  f"more common. It was not adopted.</p>")
    rows = "".join(
        f"<tr><td>{E(m)}</td><td>{max(v['part_motion_cells'].values()):.1f}</td>"
        f"<td>{v['with_changes']['actor_moving']:.0f}</td><td>{v['with_changes']['actor_frozen']:.0f}</td>"
        f"<td>{v['with_changes']['actor_moving'] - v['with_changes']['actor_frozen']:+.0f}</td>"
        f"<td>{v['positions_only']['actor_moving'] - v['positions_only']['actor_frozen']:+.0f}</td>"
        f"<td><b>{v['with_changes']['best_program']:.0f}</b></td><td>{v['with_changes']['empty_program']:.0f}</td></tr>"
        for m, v in r.items())
    lost = sum(v["with_changes"]["actor_moving"] < v["with_changes"]["actor_frozen"] for v in r.values())
    top = sum(v["with_changes"]["best_program"] >= max(v["with_changes"]["actor_moving"], v["with_changes"]["actor_frozen"])
              for v in r.values())
    return f"""<h2 id=static>Why the programs are mostly static</h2>
<p>Most fitted programs draw a posture that does not change over the 16 frames, although nearly all of them contain
motion leaves. This is not a search failure: the likelihood does not reward motion. To show it, a real training actor
was used as if it were a program (its six body parts as six outputs, drawn exactly) and scored on the other
training actors, once moving and once frozen in its best single frame. In {lost} of {len(r)} manners the
actor's own motion scored <em>worse</em> than the frozen pose, and in {top} the fitted program, mostly static,
beat both.</p>
<div class=table><table><tr><th>manner</th><th>largest part swing (cells)</th><th>actor moving</th>
<th>actor frozen</th><th>gain from motion</th><th>gain, positions only</th><th>best program</th><th>empty program</th></tr>
{rows}</table></div>
<p class=note>Log likelihood summed over the other training actors (higher is better). Part swing: how far a body
part's centre travels on the 12&times;12 board, mean over actors; limbs swing about 1&ndash;3 cells.</p>
<p>Three things combine. Actors differ in timing and amplitude, so any one trajectory is out of step with most
others, and under a Gaussian error a cautious average pose costs less than a confident motion that is off. The
programs can also express that caution directly: cells they make no claim about are filled from real bodies, which
spreads the prediction where limbs might be. Second, the frame-to-frame change term doubles the cost of being out of
step: a still prediction pays for the observed change once, a mistimed one up to twice (compare the last two gain
columns). Third, limb swings are only one to three cells on the board, close to its resolution, and the language
has no cheap rhythm: back-and-forth motion needs a self-observing oscillator, two observers linked by dashed edges.</p>
<p>Bowing, where the trunk itself travels several cells, is the exception: there the moving actor wins, and moving
programs appear among the best ones.</p>{tested}
<p>Next step: keep the walk across the room. The body is now pinned horizontally (x relative to the torso), which
removes the travel that defines locomotion: over the two cycles shown, actors cover 0.4&ndash;1.7 body heights, all
left to right, consistently across actors. That is large, one-directional motion of the kind the language's
self-propelled bodies express, unlike limb swing. It needs a wider board, slower-than-a-cell-per-tick travel and a
new way to handle phase. Other remedies: a motion summary (speed, rhythm) instead of frame-by-frame positions, or a
cheaper rhythm in the language. Any of them would be chosen on held-out actors, never on human data.</p>"""


def inputs_section(src: Path, out: Path) -> str:
    """What each board gives the fits (``src``: <manner>.gif, all parts, and <manner>_parts.gif, one board per part)."""
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    cards = []
    for g in sorted(src.glob("*_parts.gif")):
        m = g.stem[:-len("_parts")]
        for f in (src / f"{m}.gif", g):
            shutil.copy(f, out / "inputs" / f.name)
        cards.append(f"<article class=card><h3>{E(m)}</h3><figure><img src='inputs/{m}.gif' alt='{E(m)}: the parts "
                     f"together on each board' loading=lazy><figcaption>All six parts together, each in its colour"
                     f"</figcaption></figure><figure><img src='inputs/{m}_parts.gif' alt='{E(m)}: one board per part' "
                     f"loading=lazy><figcaption>One board per part: what the programs' outputs are matched to"
                     f"</figcaption></figure></article>")
    return f"""<h2 id=inputs>What the new fits see</h2>
<p>The same training actor on the three boards: the 12&times;12 body frame of the fits above (top row in each
animation), a 24&times;24 body frame with cells half as big (middle), and a 24&times;24 world frame that keeps the
walk across the screen (bottom: the camera stays put, and the board spans twice the width in body heights, with columns as
wide as before and rows half as tall). Fits on the two new boards are running; their programs will be added here.</p>
<div class=wide>{''.join(cards)}</div>"""


def boards_section(runs: dict, out: Path) -> str:
    """The same manners fitted on a finer board and in the world frame, beside the 12 x 12 control (``runs``:
    label -> run, the control first; ``board_compare.json`` next to the last run, ``motion_check.json`` in each)."""
    labels = {"control": "12&times;12, body frame (control)", "fine": "24&times;24, body frame",
              "world": "world frame: the walk across the screen kept"}
    cmp_path = list(runs.values())[-1] / "board_compare.json"
    cmp = json.loads(cmp_path.read_text()) if cmp_path.exists() else None
    manners = cmp["manners"] if cmp else sorted(json.loads((list(runs.values())[-1] / "gallery" / "gallery.json")
                                                          .read_text()))
    rows = ""
    for name, r in runs.items():
        mc = r / "motion_check.json"
        won = ""
        if mc.exists():
            x = {m: v for m, v in json.loads(mc.read_text()).items() if m in manners}
            won = f"{sum(v['with_changes']['actor_moving'] > v['with_changes']['actor_frozen'] for v in x.values())} of {len(x)}"
        c = cmp[name] if cmp and name in cmp else None
        rows += (f"<tr{' class=hi' if name != 'control' else ''}><td>{labels.get(name, E(name))}</td>"
                 + (f"<td>{c['recognition']:.2f}</td><td>{c['templates']:.2f}</td>"
                    f"<td>{np.mean(list(c['moving'].values())):.2f}</td>" if c else "<td></td>" * 3)
                 + f"<td>{won}</td></tr>")
    cards = []
    for m in manners:
        figs = []
        for name, r in runs.items():
            g = json.loads((r / "gallery" / "gallery.json").read_text()).get(m)
            if not g:
                continue
            (out / "boards" / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(r / "gallery" / g["gif"], out / "boards" / name / f"{m}.gif")
            figs.append(f"<figure><img src='boards/{name}/{m}.gif' alt='{E(m)}, {labels.get(name, E(name))}' "
                        f"loading=lazy><figcaption>{labels.get(name, E(name))}</figcaption></figure>")
        cards.append(f"<article class=card><h3>{E(m)}</h3>{''.join(figs)}</article>")
    return f"""<h2 id=boards>Finer board and world frame</h2>
<p>Two attempts to let motion count, fitted on the same {len(manners)} manners with the same settings as the
control. <b>Finer board</b>: 24&times;24 cells instead of 12&times;12, so a limb swing spans two to six cells
instead of one to three. <b>World frame</b>: the walker is no longer pinned horizontally; the board is twice as wide
in body heights (columns as wide as before, rows half as tall), so the walk across the screen is on it. The latent
phase shifts the body's cycle but not its travel. Training actors fitted, validation actors scored; the test actors
stay sealed.</p>
<div class=table><table><tr><th>board</th><th>recognition (top-1)</th><th>kinematic templates (top-1)</th>
<th>share of moving programs</th><th>moving actor beats frozen</th></tr>{rows}</table></div>
<p class=note>Recognition among these {len(manners)} manners, from each fit's held-out predictive densities;
templates: the best nearest-template classifier on the same board's features. Densities themselves are not compared:
on a different board the observations differ. Moving: share of posterior draws whose outputs change over the 16
frames, mean over manners. Last column: the actor test above, rerun on each board.</p>
<p>In each animation: the video, the body parts on that board, and the best program's outputs.</p>
<div class=grid>{''.join(cards)}</div>"""


def similarity_section(run: Path) -> str:
    s = json.loads((run / "analysis" / "summary.json").read_text())
    names = {"kin_manner_prototype": "kinematic: manner mean fields", "kin_manner_pairwise": "kinematic: actor pairs",
             "kin_manner_joints": "kinematic: joint positions", "lot_predictive": "LoT: cross-prediction",
             "lot_full": "LoT: program trees (full)", "lot_tree_ordered": "LoT: trees, no composition type",
             "lot_tree_unordered": "LoT: trees, order ignored", "lot_flat": "LoT: leaf labels only",
             "map_full": "LoT: best program only (full)"}
    rows = "".join(
        f"<tr class={'hi' if k == 'lot_predictive' else ''}><td>{E(v)}</td><td>{r['rho']:.3f}</td><td>[{r['ci_low']:.2f}, {r['ci_high']:.2f}]</td>"
        f"<td>{r['p']:.4f}</td><td>{r.get('delta_r2', float('nan')):.3f}</td><td>{r.get('p_delta', float('nan')):.3f}</td></tr>"
        for k, v in names.items() if k in s["models"] for r in [s["models"][k]])
    nc = s["noise_ceiling"]
    verdict = ""
    if "lot_predictive" in s["models"]:
        pr, tree = s["models"]["lot_predictive"], s["models"]["lot_full"]
        verdict = (f"<p>Two kinds of LoT similarity behave differently. Comparing what the programs <em>predict</em> "
                   f"works: manners whose programs explain each other's new actors well are the ones people rate as "
                   f"similar (&rho; = {pr['rho']:.2f}), and this adds to the kinematic baselines (&Delta;R&sup2; = "
                   f"{pr['delta_r2']:.3f}, p = {pr['p_delta']:.4f}; cross-validated &rho; with the kinematic baselines "
                   f"{pr['cv_rho_kin_plus_this']:.2f} vs {s['cv_rho_kinematic_only']:.2f} without). Comparing the "
                   f"programs' <em>text</em> (tree edit distance) does not (&rho; = {tree['rho']:.2f}), likely because the posteriors "
                   f"are diffuse, so many differently written programs draw nearly the same thing. In an earlier "
                   f"version of the language the tree measure reached &rho; = 0.30.</p>")
    return f"""<h2 id=similarity>Human similarity judgments</h2>
<p>GRACE Norming Study 3: participants rated the similarity of every pair of the 26 manners (1&ndash;7). Model
similarities come from the manner posteriors fitted to the training actors (cross-prediction uses the validation
actors), and kinematic baselines from the same actors. The human ratings enter only here, as the thing compared
against; no model setting was chosen with them, and the test actors stay sealed.</p>
<div class=table><table><tr><th>similarity</th><th>Spearman &rho;</th><th>95% CI (participants)</th><th>Mantel p</th>
<th>&Delta;R&sup2; over kinematic</th><th>p(&Delta;R&sup2;)</th></tr>{rows}</table></div>
<p class=note>Noise ceiling (split-half, Spearman&ndash;Brown) &rho; = {nc['split_half_sb']:.2f}; kinematic baselines
together, cross-validated &rho; = {s['cv_rho_kinematic_only']:.2f}. Cross-prediction: how well one manner's
programs explain the other manner's validation videos, double-centred and symmetrised.</p>{verdict}"""


def descriptor_section(run: Path, out: Path) -> str:
    """The fit scored on limb motion (``inference.likelihood = "descriptors"``): its programs, how close they come to
    an ideal program against the earlier fits, and how big the legs' swings are against what the programs draw."""
    c = json.loads((run / "compare_scoring.json").read_text())
    pred = json.loads((run / "predictive.json").read_text())
    labels = {"fixed": "motion scoring", "trial": "trial (short, motion scoring)", "old": "position scoring"}
    fits = [f for f in labels if any(f in v for v in c.values())]
    pct = lambda r: f"{100 * r['share']:.0f}% <span class=note>({r['nodes']} nodes)</span>"
    rows = "".join(f"<tr><td>{E(m)}</td>" + "".join(f"<td>{pct(v[f]) if f in v else '&ndash;'}</td>" for f in fits)
                   + "</tr>" for m, v in c.items())
    head = "".join(f"<th>{labels[f]}</th>" for f in fits)
    gallery = gallery_cards(run, out, "motion") if (run / "gallery" / "gallery.json").exists() else ""
    view = ""
    mv = run / "motion_view"
    if mv.exists():
        (out / "motion_view").mkdir(parents=True, exist_ok=True)
        cards = []
        for f in sorted(mv.glob("*.png")):
            m = f.stem
            for ext in ("png", "gif"):
                shutil.copy(mv / f"{m}.{ext}", out / "motion_view" / f"{m}.{ext}")
            cards.append(f"""<article class=partview id="part-{m}"><h3>{E(m)}</h3>
<img src="motion_view/{m}.gif" alt="{E(m)}: the video, each body part on its own board, and the program output matched to it" loading=lazy>
<img src="motion_view/{m}.png" alt="{E(m)}: per body part, x, y and tilt over the frames, actor against program; the swing amplitudes and limb-pair couplings compared" loading=lazy></article>""")
        view = f"""<h3>Part by part</h3><p>Each body part against the program output matched to it, for the training
actor closest to the manner's average motion. In the animation, rows: the actor's parts, one board each; the matched
program output (grey: no claim, black: claimed empty); one fill of it, as the scoring sees it (cells the program
makes no claim about filled from a real body part's footage). In the chart, per part: the centre's left&ndash;right
position and height (in cells, up is up) and the tilt (degrees, clockwise), each about its own average, for the
actor (black), for what the program itself draws (solid, its claimed figure only; gaps: frames it draws nothing)
and for one fill (dashed); the row label gives the share of the board the output claims. Right: the numbers the
scoring compares: actors' mean and spread, the scoring's prediction (over the fills) and what the program's own
drawing alone gives (hatched). Programs' traces are shifted in time, all parts alike, since timing is not scored.</p>
<p><b>Most of the scored motion is not the program's.</b> Outputs often claim a few percent of the board; their swing
comes from the fills, where real footage and the program's partial claims mix cell by cell. A program can match the
actors' swing sizes without drawing any motion of its own.</p><div class=stack>{''.join(cards)}</div>"""
    legs = ""
    sz = run / "swing_size.json"
    if sz.exists():
        z = json.loads(sz.read_text())
        cell = lambda a, d, p: f"{a:.2f} <span class=note>&plusmn;{d:.2f}</span></td><td>{p:.2f}"
        legs_rows = "".join(
            f"<tr><td>{E(m)}</td><td>{E(part.replace('_', ' '))}</td><td>"
            + "</td><td>".join(cell(r["actor"][i], r["spread"][i], r["program"][i]) for i in (0, 1, 3, 4)) + "</td></tr>"
            for m, v in z.items() for part, r in v.items() if part.endswith("leg"))
        legs = f"""<h3>How far the legs swing</h3>
<p>How far each leg's centre moves, in board cells (12 to the window), once and twice per cycle: actors (mean
&plusmn; spread across actors) beside the matched program output's prediction. Actors move a leg's centre by a fifth
to a third of a cell, less than the smallest whole-cell step of the language; the programs still match them closely,
because cells they make no claim about are filled from real bodies and their motion is averaged over those fills.
The clear miss is jumping's up&ndash;down swing once per cycle (about 0.35 cells predicted against the actors' 0.06).</p>
<div class=table><table><tr><th>manner</th><th>part</th><th>x once, actors</th><th>program</th><th>y once, actors</th>
<th>program</th><th>x twice, actors</th><th>program</th><th>y twice, actors</th><th>program</th></tr>{legs_rows}
</table></div>"""
    return f"""<h2 id=motion>Scoring motion instead of positions</h2>
<p>A second fit of the same five manners, with programs scored on how their parts move rather than on where each
cell is in each frame: for each body part, how far its centre and its tilt swing once and twice per cycle, and for
four limb pairs, whether they move in step or opposite. Timing, tempo and where the body stands are not scored. The
language is unchanged. Held-out recognition among the five manners: <b>{pred['recognition_top1']:.2f}</b>, as with
position scoring.</p>
<p>Below, each fit's best program on the training actors, scored by motion: 0% is the empty program, 100% an ideal
program that predicts every actor's motion exactly at the manner's average.</p>
<div class=table><table><tr><th>manner</th>{head}</tr>{rows}</table></div>
<p class=note>Position-scored programs fall below the empty program here: they were fitted to cell positions, and
their motion was never asked to match.</p>
<p>Every output is matched to a body part, and in every manner each part's swing is predicted better than by an
average body part; the limb pairs add little on top.</p>{legs}
<h3>The programs</h3><p>Each manner's best program and the medoid training actor. With motion scoring, where the
outputs sit no longer matters, so they need not look like a body; only how they move is compared.</p>
{legend(PART_COLOURS)}{gallery}{view}"""


def full_section(run: Path, out: Path) -> str:
    """A short fit on the full descriptors (``inference.descriptors = "full"``): its programs and where their score
    comes from, against the motion fit's programs scored the same way (``compare_scoring.json`` in the run)."""
    c = json.loads((run / "compare_scoring.json").read_text())
    pred = json.loads((run / "predictive.json").read_text())
    inf = json.loads((run / "config.json").read_text())["inference"]
    it, warmup = inf["iterations"], inf.get("warmup", 0)
    rows = "".join(
        f"<tr><td>{E(m)}</td>" + "".join(
            f"<td>{100 * v[f]['share']:.0f}%</td><td>{v[f]['nodes']}</td><td>{v[f]['outputs']}</td>"
            f"<td>{v[f]['motion']:+.0f} / {v[f]['posture']:+.0f} / {v[f]['pairs']:+.0f}</td>"
            for f in ("full fit", "motion fit")) + "</tr>" for m, v in c.items())
    gallery = gallery_cards(run, out, "full") if (run / "gallery" / "gallery.json").exists() else ""
    views = ""
    mv = run / "motion_view"
    if mv.exists():
        (out / "full_view").mkdir(parents=True, exist_ok=True)
        for f in sorted(mv.glob("*.png")):
            for ext in ("png", "gif"):
                shutil.copy(mv / f"{f.stem}.{ext}", out / "full_view" / f"{f.stem}.{ext}")
        views = "<div class=stack>" + "".join(
            f"""<article class=partview><h3>{E(f.stem)}</h3><img src="full_view/{f.stem}.gif" alt="{E(f.stem)}: each body
part and its matched output" loading=lazy><img src="full_view/{f.stem}.png" alt="{E(f.stem)}: the numbers compared, part by
part" loading=lazy></article>""" for f in sorted(mv.glob("*.png"))) + "</div>"
    length, pilot = ("A full-length", "") if it >= 24000 else ("A short pilot", (
        "The motion fit's programs collapse once posture counts: they never had to put a part where it belongs. The "
        "pilot gains on both motion and posture but stays far from the ideal with small programs; a full-length fit "
        "is the test. "))
    note = (run / "note.html").read_text(encoding="utf-8") if (run / "note.html").exists() else ""
    warm = (f" Each program ran {warmup} ticks before scoring began, so what counts is how it keeps behaving: a one-off "
            "change at the start of a clip used to read as a swing at every rhythm." if warmup else "")
    return f"""<h2 id=full>Adding rhythm and posture</h2>
<p>Two gaps in the motion scoring, fixed by one rule set before fitting: it compared only two rhythms, so a single
step across the clip could stand in for a repeating movement; and it ignored where each part is, so programs needed
not look like a body. The full set adds every rhythm from once per clip to twice per cycle, and each part's posture:
its average position on the board (which follows the body), size and tilt. {length} fit ({it} iterations) of the same
five manners; held-out recognition <b>{pred['recognition_top1']:.2f}</b>.{warm}</p>
<div class=table><table><tr><th rowspan=2>manner</th><th colspan=4>full fit</th><th colspan=4>motion fit, rescored</th></tr>
<tr><th>to ideal</th><th>nodes</th><th>outputs</th><th>motion / posture / pairs</th><th>to ideal</th><th>nodes</th>
<th>outputs</th><th>motion / posture / pairs</th></tr>{rows}</table></div>
<p class=note>Scored on the full set, training actors. "To ideal": the share of the way from the empty program to an
ideal one. The last columns: gain over the empty program, in log likelihood, from the rhythm amplitudes, the posture
and the limb pairs.</p>
{note}<p>{pilot}In the charts below the posture bars (several cells) dwarf the swing bars; the scoring weighs each in noise
units, not cells.</p>
{legend(PART_COLOURS)}{gallery}{views}"""


def findings(run: Path, compare: dict) -> str:
    """The headline numbers, from the same files the sections read."""
    tiles = []
    if (run / "predictive.json").exists():
        p = json.loads((run / "predictive.json").read_text())
        n = len(p["manners"])
        tiles.append((f"{p['recognition_top1']:.0%}", f"of new actors' videos recognised among {n} manners "
                                                       f"(chance {1 / n:.0%})"))
        if (run / "kinematic_baselines.json").exists():
            k = json.loads((run / "kinematic_baselines.json").read_text())
            best = max(v["top1"] for b, v in k.items() if b.startswith(("prototype", "1nn")))
            tiles.append((f"{best:.0%}", "recognised by the best kinematic template (no programs): "
                                         "the programs are below it"))
    if compare:
        d = {k: json.loads((Path(v) / "predictive.json").read_text())["own_lpd_total"] for k, v in compare.items()}
        cur = [k for k in d if k.startswith("current")]
        if cur:
            best_other = max(v for k, v in d.items() if k not in cur)
            tiles.append((f"+{d[cur[0]] - best_other:.0f}", "held-out log density of the current language over the "
                                                            "best earlier version (5 manners)"))
    g = run / "gestures" / "gestures.json"
    if g.exists():
        x = json.loads(g.read_text())
        tiles.append((f"{x['lot']['mean_rank_own_manner']:.1f} / {len(x['manners'])}",
                      f"mean rank of a gesture's own manner (chance {x['lot']['chance_mean_rank']:.1f}): "
                      "no resemblance found"))
    a = run / "analysis" / "summary.json"
    if a.exists():
        m = json.loads(a.read_text())["models"]
        lot = max((k for k in m if k.startswith("lot_")), key=lambda k: m[k]["rho"])
        kin = max((k for k in m if k.startswith("kin_")), key=lambda k: m[k]["rho"])
        tiles.append((f"&rho; {m[lot]['rho']:.2f}", f"best LoT similarity vs human ratings (kinematic best "
                                                    f"{m[kin]['rho']:.2f}; gain over kinematic p = {m[lot]['p_delta']:.4f})"))
    return "<div class=findings>" + "".join(f"<div class=finding><b>{a}</b><span>{b}</span></div>" for a, b in tiles) + "</div>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/presentation")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--motion", help="motion_check.json to show (default: the run's own)")
    ap.add_argument("--fix", nargs=2, metavar=("FIX_RUN", "CONTROL_RUN"), help="a tested motion fix and its control")
    ap.add_argument("--boards", nargs="*", default=[], help="label=run ..., the 12 x 12 control first")
    ap.add_argument("--descriptors", help="a fit scored on limb motion, with compare_scoring.json (and swing_size.json)")
    ap.add_argument("--full", help="a fit on the full descriptors, with compare_scoring.json")
    ap.add_argument("--inputs", help="a folder of input animations per board (<manner>.gif, <manner>_parts.gif)")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    compare = dict(c.split("=", 1) for c in args.compare)
    parts = []
    if (run / "gallery" / "gallery.json").exists():
        parts.append(gallery_section(run, out))
    if compare:
        parts.append(heldout_section(run, compare))
    if args.descriptors:
        parts.append(descriptor_section(Path(args.descriptors), out))
    if args.full:
        parts.append(full_section(Path(args.full), out))
    motion = Path(args.motion) if args.motion else run / "motion_check.json"
    if motion.exists():
        fx = [Path(x) for x in args.fix] if args.fix else [None, None]
        parts.append(motion_section(motion, *fx))
    if args.inputs:
        parts.append(inputs_section(Path(args.inputs), out))
    if args.boards:
        parts.append(boards_section({k: Path(v) for k, v in (b.split("=", 1) for b in args.boards)}, out))
    if (run / "gestures" / "gestures.json").exists():
        parts.append(gesture_section(run, out))
    if (run / "analysis" / "summary.json").exists():
        parts.append(similarity_section(run))
    page = f"""<title>Locomotion Programs</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&family=Schibsted+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600&display=swap">
<style>{css()}</style>
<div class=wrap>
<p class=eyebrow>Embodied LoT &middot; GRACE locomotion</p>
<h1>Locomotion programs</h1>
<p class=lede>Programs in the grounded language of thought, fitted to videos of 26 manners of locomotion. What they
look like, how well they predict new actors, whether they explain the gestures that mime each manner, and how they
relate to human similarity judgments.</p>
<!--FINDINGS-->
{''.join(parts)}
</div>"""
    page = page.replace("<!--FINDINGS-->", findings(run, compare))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
