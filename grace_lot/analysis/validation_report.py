"""Build the validation-only report. The test actors are never read.

    python -m grace_lot.analysis.validation_report --root results/grace_v2

The inputs are produced earlier in the pipeline:
    <root>/recovery*/recovery.json         synthetic recovery runs
    <root>/coverage.json                   grammar and search coverage (written here if missing)
    <root>/select/selection.json           validation grid (grace_lot.inference.select)
    <root>/select/<chosen>/posteriors      posteriors of the chosen setting, fitted on the training actors
    <root>/select/<chosen>/analysis        human comparison (grace_lot.analysis.run on those posteriors)

The report goes to <root>/validation_report.md, with figures in <root>/report_figures/.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grace_lot.analysis.heldout import evaluate
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.inference.grammar import Grammar
from grace_lot.inference.prior import SizePrior
from grace_lot.lot_adapter.adapter import LotRenderer
from grace_lot.lot_adapter.program import parse_program, saved_lot, size


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/grace_v2")
    ap.add_argument("--config")
    args = ap.parse_args()
    root = Path(args.root)
    figs = root / "report_figures"
    figs.mkdir(parents=True, exist_ok=True)
    sel = json.loads((root / "select" / "selection.json").read_text())
    chosen_dir = Path(sel["chosen_dir"])
    run_cfg = json.loads((chosen_dir / "config.json").read_text())
    cfg = load_config(args.config)
    for k in ("preprocess", "lot", "inference", "split"):
        cfg[k] = run_cfg[k]
    cfg["lot"] = saved_lot(cfg["lot"])
    d = cfg["data"]
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    L = ["# GRACE x LoT: validation-only report", "",
         "**The test actors (" + ", ".join(cfg["split"]["test_actors"]) + ") are sealed.** Nothing in this report "
         "reads, scores or plots them. Posteriors are fitted on the 15 training actors, and hyperparameters are "
         "chosen on the 5 validation actors (" + ", ".join(cfg["split"]["val_actors"]) + ").", ""]

    # 1. Recovery
    L += ["## 1. Synthetic recovery", "",
          "Programs drawn from the prior are rendered, copied into noisy 'actors' and inferred with the same "
          "sampler. *search ok*: the best posterior found is at least as good as the truth's. *equiv mass*: "
          "posterior mass on programs that render identically to the truth.", ""]
    rows = []
    for f in sorted(root.glob("recovery*/recovery.json")):
        r = json.loads(f.read_text())
        rows.append([f.parent.name, r["n"], r["actors"], r["noise"], r.get("shift", 0),
                     f"{r['search_ok_rate']:.2f}", f"{r['mean_equiv_mass']:.2f}", f"{r['map_equiv_rate']:.2f}",
                     f"{r['mean_sim_to_truth']:.2f} vs {r['mean_sim_prior']:.2f}"])
    L += _table(["run", "n", "actors", "noise", "time shift", "search ok", "equiv mass", "MAP equiv",
                 "sim to truth vs prior"], rows) + [""]
    for f in sorted(root.glob("recovery*/recovery.json")):
        fails = [x for x in json.loads(f.read_text())["results"] if not x["search_ok"]]
        if fails:
            L += [f"Failures in `{f.parent.name}`:", ""]
            L += [f"- truth `{x['truth']}`: best {x['best_log_post']:.1f} vs truth {x['truth_log_post']:.1f}, "
                  f"R-hat {x['r_hat']:.2f}" for x in fails] + [""]

    # 2. Coverage
    cov = json.loads((root / "coverage.json").read_text())
    L += ["## 2. Grammar and search coverage", "",
          f"- Exact reachability (small grammar, every legal program enumerated): {cov['reachability']['reached']} / "
          f"{cov['reachability']['support']} programs reached from a single leaf.",
          f"- Configured grammar: {cov['family']['families_visited']} of {cov['family']['families_with_prior_mass']} "
          f"program families visited by flat-likelihood MCMC; families with clear prior mass left unvisited: "
          f"{len(cov['family']['unvisited_with_mass'])}; total variation of the family distribution against an independent "
          f"importance-sampling estimate: {cov['family']['tv_family_marginal']:.3f} (selected search bounds).",
          "- Exactness unit tests (tests/test_grace_lot.py) check that MH and parallel tempering reproduce exact priors "
          "and posteriors for every move type, dashed edges included.", ""]

    # 3. Selection
    L += ["## 3. Hyperparameter selection (validation actors)", "",
          "Criterion: top-1 recognition of validation videos, where a video's score for manner m is the expected "
          "distance to programs from that manner's training posterior (ties broken by MRR). No human data are used. "
          "Chance top-1 = 0.038.", ""]
    L += _table(["setting", "LoT top-1", "top-5", "MRR", "kin prototype top-1", "kin 1-NN top-1"],
                [[r["tag"], f"{r['lot_top1']:.3f}", f"{r['lot_top5']:.3f}", f"{r['lot_mrr']:.3f}",
                  f"{r['kin_prototype_top1']:.3f}", f"{r['kin_1nn_top1']:.3f}"] for r in sel["rows"]]) + [""]
    L += [f"**Selected:** `{sel['best']['tag']}`.", ""]

    # 3b. Mixing of every candidate (validation-side information only)
    mix_rows = []
    for r in sel["rows"]:
        pdir = Path(r["dir"]) / "posteriors"
        ps = [json.loads(p.read_text()) for p in pdir.glob("*.json")]
        rh = [p["r_hat_log_post"] for p in ps]
        cold = np.mean([c["swaps"][0] for p in ps for c in p["chains"]])
        mix_rows.append([r["tag"], f"{r['lot_top1']:.3f}", f"{np.median(rh):.2f}", f"{np.max(rh):.2f}",
                         sum(x < 1.1 for x in rh), f"{cold:.2f}"])
    L += ["### Mixing of every candidate", "",
          "Only the beta = 250 settings converge (median R-hat about 1.06). Every beta >= 1000 setting, including the "
          "selected one, has chains that disagree. Recognition rewards those settings even though their 'posteriors' "
          "behave more like stochastic optimisation than Bayesian inference. **Choosing between a converged posterior "
          "(beta 250) and better recognition from unconverged chains is a decision for you, not for the pipeline.**", ""]
    L += _table(["setting", "val top-1", "R-hat median", "R-hat max", "manners < 1.1", "cold-end swap"], mix_rows) + [""]

    # 4. MCMC diagnostics for the chosen run
    posts = {p.stem: json.loads(p.read_text()) for p in sorted((chosen_dir / "posteriors").glob("*.json"))}
    acc = Counter(); prop = Counter(); rhat = []; swaps = []
    for r in posts.values():
        rhat.append(r["r_hat_log_post"])
        for c in r["chains"]:
            acc.update(c["accepted"]); prop.update(c["proposed"]); swaps.append(c["swaps"])
    L += ["## 4. MCMC acceptance and mixing (chosen setting, training actors)", "",
          f"R-hat of the log-posterior over {len(rhat)} manners: median {np.median(rhat):.2f}, "
          f"max {np.max(rhat):.2f}; {sum(x < 1.1 for x in rhat)} below 1.1 and {sum(x < 1.5 for x in rhat)} below 1.5.", ""]
    L += _table(["move", "proposed", "accepted", "rate"],
                [[k, prop[k], acc[k], f"{acc[k] / max(prop[k], 1):.3f}"] for k in sorted(prop)]) + [""]
    sw = np.mean(np.array(swaps), axis=0)
    L += ["Mean swap acceptance between neighbouring tempering rungs (cold to hot): "
          + ", ".join(f"{x:.2f}" for x in sw), ""]
    L += _table(["manner", "R-hat", "unique programs", "MAP D (train)"],
                [[m, f"{r['r_hat_log_post']:.2f}", r["unique_programs"], f"{r['map_distance']:.4f}"]
                 for m, r in posts.items()]) + [""]

    # 5. Program length distribution
    g = Grammar.from_config(cfg["lot"])
    prior = SizePrior.from_config(g, cfg["inference"])
    max_nodes, max_dashed = cfg["lot"]["max_nodes"], cfg["lot"]["max_dashed"]
    sizes, edges, dl = Counter(), Counter(), []
    for r in posts.values():
        for s, w in r["posterior"]:
            p = parse_program(s)
            sizes[size(p.tree)] += w / len(posts)
            edges[len(p.dashed)] += w / len(posts)
            dl.append((prior.length(p), w / len(posts)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].bar(list(sizes), list(sizes.values())); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("posterior mass (avg over manners)")
    ax[1].bar(list(edges), list(edges.values())); ax[1].set_xlabel("dashed edges")
    fig.tight_layout(); fig.savefig(figs / "program_length.png", dpi=120); plt.close(fig)
    mean_dl = sum(l * w for l, w in dl)
    L += ["## 5. Posterior program length", "",
          f"Average over manners: mean description length {mean_dl:.1f} nats; node-count distribution "
          + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(sizes.items()))
          + "; dashed edges " + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(edges.items())) + ".",
          f"Mass at the search bounds (nodes = {max_nodes} or edges = {max_dashed}): "
          f"{sizes.get(max_nodes, 0):.2f} / {edges.get(max_dashed, 0):.2f}.",
          "", "![program length](report_figures/program_length.png)", ""]

    # 6. Trajectory fit (training) and validation fit
    train = split_videos(d["raw_dir"], cfg["split"], "train")
    val = split_videos(d["raw_dir"], cfg["split"], "val")
    ev = evaluate(cfg, chosen_dir / "posteriors", train, val)
    renderer = LotRenderer(cfg["preprocess"]["frames"], cfg["preprocess"]["field_size"])
    fit_rows = []
    for m in sorted(posts):
        mp = renderer(parse_program(posts[m]["map"]))
        tr = np.mean([((load_trajectory(cfg, v).fields - mp) ** 2).mean() for v in train[m]])
        va = np.mean([((load_trajectory(cfg, v).fields - mp) ** 2).mean() for v in val[m]])
        empty = np.mean([(load_trajectory(cfg, v).fields ** 2).mean() for v in train[m]])
        proto = np.mean([load_trajectory(cfg, v).fields for v in train[m]], axis=0)
        pr = np.mean([((load_trajectory(cfg, v).fields - proto) ** 2).mean() for v in val[m]])
        fit_rows.append([m, f"{tr:.4f}", f"{va:.4f}", f"{pr:.4f}", f"{empty:.4f}", ev["per_video_lot_rank_mean"][m]])
    L += ["## 6. Trajectory fit", "",
          "Mean squared cell difference between the MAP program's rendering and the actors' field sequences. "
          "*train-mean vs val* is the validation error of the training actors' mean field sequence, a strong "
          "non-compositional reference; *empty* is the error of predicting nothing; *mean val rank* is the rank of the "
          "true manner for this manner's validation videos (1 = recognised).", ""]
    L += _table(["manner", "MAP vs train", "MAP vs val", "train-mean vs val", "empty", "mean val rank"], fit_rows) + [""]
    L += ["Validation recognition with the chosen setting:", ""]
    L += _table(["model", "top-1", "top-5", "MRR", "n"],
                [[k, f"{v['top1']:.3f}", f"{v['top5']:.3f}", f"{v['mrr']:.3f}", v["n"]]
                 for k, v in ev.items() if isinstance(v, dict) and "top1" in v]) + [""]

    # 7/8. Human comparison
    summary_path = chosen_dir / "analysis" / "summary.json"
    if not summary_path.exists():
        L += ["## 7-9. Human comparison", "", "*Not run yet for the chosen setting.*", ""]
        (root / "validation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
        return
    summ = json.loads(summary_path.read_text())
    nc = summ["noise_ceiling"]
    L += ["## 7. LoT similarity vs human similarity (posteriors from the training actors)", "",
          f"Human split-half noise ceiling: rho = {nc['split_half_sb']:.3f}. Human data were not used for any choice above.", ""]
    keys = [k for k in summ["models"] if k.startswith("lot_") or k.startswith("map_")]
    L += _table(["model", "rho", "95% CI", "Mantel p", "dR2 over kinematic", "p(dR2)", "CV rho kin+model"],
                [[k, f"{summ['models'][k]['rho']:.3f}", f"[{summ['models'][k]['ci_low']:.3f}, {summ['models'][k]['ci_high']:.3f}]",
                  f"{summ['models'][k]['p']:.4f}", f"{summ['models'][k]['delta_r2']:.3f}", f"{summ['models'][k]['p_delta']:.4f}",
                  f"{summ['models'][k]['cv_rho_kin_plus_this']:.3f}"] for k in keys]) + [""]
    L += ["## 8. Kinematic baselines vs human similarity (same training actors)", ""]
    kin = [k for k in summ["models"] if k.startswith("kin_")]
    L += _table(["baseline", "rho", "95% CI", "Mantel p"],
                [[k, f"{summ['models'][k]['rho']:.3f}", f"[{summ['models'][k]['ci_low']:.3f}, {summ['models'][k]['ci_high']:.3f}]",
                  f"{summ['models'][k]['p']:.4f}"] for k in kin]) + [""]
    L += [f"Leave-one-manner-out CV rho, kinematic baselines only: {summ['cv_rho_kinematic_only']:.3f}.",
          "", "![heatmaps](select/" + chosen_dir.name + "/analysis/heatmaps.png)",
          "", "![scatter](select/" + chosen_dir.name + "/analysis/scatter.png)", ""]

    # 9. Failure cases
    worst_val = sorted(ev["per_video_lot_rank_mean"].items(), key=lambda kv: -kv[1])[:6]
    res = summ["residuals_lot_full"]
    L += ["## 9. Failure cases", "",
          "Manners whose validation videos the LoT ranks worst (mean rank of the true manner, 1 = best): "
          + ", ".join(f"{m} ({r:.1f})" for m, r in worst_val) + ".", "",
          "Human pairs where the full LoT measure is most *over*-similar (rank residual): "
          + "; ".join(f"{a}-{b} ({r:+.0f})" for (a, b), r in res["model_too_similar"][:6]) + ".", "",
          "Human pairs where it is most *under*-similar: "
          + "; ".join(f"{a}-{b} ({r:+.0f})" for (a, b), r in res["model_too_dissimilar"][:6]) + ".", "",
          "Recovery failures are listed in section 1. Manners with R-hat > 1.5 are in section 4.", ""]
    n_val = ev["lot"]["n"]
    se = (sel["best"]["lot_top1"] * (1 - sel["best"]["lot_top1"]) / n_val) ** 0.5
    L += ["## 10. Caveats", "",
          f"- **The selection is noisy.** With {n_val} validation videos, top-1 has a standard error of about "
          f"{se:.3f}, so settings within about 0.05 of each other are not reliably different.",
          "- **The chosen setting is at the edge of the grid** (lowest lambda, highest beta, larger bounds), so the "
          "optimum may lie outside it. Stage-2 variants were each tried alone at the best grid point, not in combination.",
          "- **Identifiability.** In recovery with actor-like variability (noise 0.08, time shifts of up to 3 "
          "frames), search always reached the truth's posterior level, but exact equivalence to the true program "
          "dropped sharply. Under realistic variability, program identity is only partly recoverable; posterior "
          "*similarity* to the truth stays well above the prior.",
          "- **Mixing in the larger search space is worse** than at the default bounds (see section 4). Posterior "
          "expectations for manners with high R-hat rest on chains that disagree.",
          "- **Data hygiene.** Pose preprocessing ran on all 650 videos, including the test actors' (none of those "
          "outputs was inspected). The first prototype (results/grace_poc) predates the split and used the Study-3 "
          "exemplars, five of which belong to test actors. It is not part of this report.",
          "- The scurrying exemplar is a substitute (10F withdrew consent); it does not affect the manner-level "
          "posteriors used here.", ""]
    frozen_path = Path(__file__).resolve().parents[2] / "configs" / "grace_frozen.toml"
    frozen_tag = None
    if frozen_path.exists():
        for line in frozen_path.read_text(encoding="utf-8").splitlines()[:6]:
            if "setting '" in line:
                frozen_tag = line.split("setting '")[1].split("'")[0]
    L += ["## 11. Frozen configuration", ""]
    if frozen_tag and frozen_tag != sel["best"]["tag"]:
        L += [f"**Frozen: `{frozen_tag}`** (chosen by the user over the top-ranked `{sel['best']['tag']}`, because "
              "it is the best-recognising setting whose MCMC converges). Sections 4-9 describe the top-ranked "
              f"setting; the human comparison has **not** been run for `{frozen_tag}`.", ""]
    else:
        L += [f"Frozen: `{sel['best']['tag']}`.", ""]
    L += ["File: `configs/grace_frozen.toml`. Test evaluation (a train+val fit, then "
          "`python -m grace_lot.analysis.run --results <final> --unseal-test`) has **not** been run.", ""]

    # Readiness summary, inserted after the title block
    flags = []
    if np.median(rhat) > 1.1:
        flags.append(f"**MCMC has not converged** at the selected setting: median R-hat {np.median(rhat):.2f}, "
                     f"max {np.max(rhat):.2f}; cold-end swap rate {sw[0]:.2f}. The posteriors (and the human "
                     "comparison built from them) rest on chains that disagree.")
    if sizes.get(max_nodes, 0) > 0.2 or edges.get(max_dashed, 0) > 0.2:
        flags.append(f"**Search bounds bind**: {sizes.get(max_nodes, 0):.0%} of posterior mass at {max_nodes} nodes "
                     f"and {edges.get(max_dashed, 0):.0%} at {max_dashed} dashed edges.")
    flags.append(f"Validation recognition: LoT top-1 {ev['lot']['top1']:.2f} vs kinematic prototype "
                 f"{ev['kin_prototype']['top1']:.2f} (chance 0.04).")
    flags.append(f"Human similarity: best LoT measure rho = {max(summ['models'][k]['rho'] for k in keys):.2f}, "
                 f"vs kinematic prototype rho = {summ['models']['kin_manner_prototype']['rho']:.2f}; no LoT measure "
                 f"adds variance beyond kinematics (max dR2 = {max(summ['models'][k]['delta_r2'] for k in keys):.3f}).")
    dense = chosen_dir.parent / (chosen_dir.name + "_dense_ladder") / "dense_ladder_summary.json"
    if dense.exists():
        dl = json.loads(dense.read_text())
        flags.append(f"Dense-ladder rerun (24 replicas, diagnostic only): validation top-1 {dl['lot']['top1']:.2f}, "
                     f"median R-hat {dl['r_hat_median']:.2f} (max {dl['r_hat_max']:.2f}), "
                     f"cold-end swap {dl['swap_rates'][0]:.2f}. Human comparison not run on it.")
    L[4:4] = ["## Readiness summary", ""] + [f"- {f}" for f in flags] + [""]
    (root / "validation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
