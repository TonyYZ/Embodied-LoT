"""Compare LoT-derived, kinematic and human similarity for the 26 GRACE manners.

    python -m grace_lot.analysis.run [--config ...] [--results results/grace_poc]

This reads ``<results>/posteriors/*.json`` (from ``grace_lot.inference.run``)
and writes the following to ``<results>/analysis/``:
    summary.json          every statistic
    summary.md            readable tables
    matrices.npz          every similarity matrix (human, kinematic, LoT measures, ablations)
    heatmaps.png, scatter.png, traces.png, fits/<manner>.png
"""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from grace_lot.analysis import plots
from grace_lot.analysis.baselines import kinematic_matrices
from grace_lot.analysis.human_similarity import load_ratings, similarity_matrix, subjects
from grace_lot.analysis.rsa import (bootstrap_ci, bootstrap_humans, incremental, loo_manner_cv,
                                    mantel, noise_ceiling, rho, upper)
from grace_lot.config import load_config
from grace_lot.grace.dataset import split_videos, study3_exemplars, substituted
from grace_lot.analysis.heldout import evaluate as heldout_evaluate
from grace_lot.lot_adapter.distance import euclid
from grace_lot.grace.preprocess import load_trajectory
from grace_lot.grace.trajectories import preview
from grace_lot.lot_adapter.adapter import rollout
from grace_lot.lot_adapter.program import parse_program, saved_lot
from grace_lot.lot_adapter.similarity import ABLATIONS, MEASURES, expected_similarity, sample_pairs

ALL_MEASURES = {**MEASURES, **ABLATIONS}


def _pair_job(job):
    i, j, pairs, map_pair = job
    exp = {k: expected_similarity(m, pairs) for k, m in ALL_MEASURES.items()}
    mp = {k: m(*map_pair) for k, m in MEASURES.items()}
    return i, j, exp, mp


def lot_matrices(posteriors, maps, manners, n_pairs, seed):
    n = len(manners)
    rng = random.Random(seed)
    jobs = [(i, j, sample_pairs(posteriors[manners[i]], posteriors[manners[j]], n_pairs, rng),
             (maps[manners[i]], maps[manners[j]]))
            for i in range(n) for j in range(i + 1, n)]
    out = {f"lot_{k}": np.full((n, n), np.nan) for k in ALL_MEASURES}
    out.update({f"map_{k}": np.full((n, n), np.nan) for k in MEASURES})
    with Pool(8) as pool:
        for i, j, exp, mp in pool.imap_unordered(_pair_job, jobs, chunksize=4):
            for k, v in exp.items():
                out[f"lot_{k}"][i, j] = out[f"lot_{k}"][j, i] = v
            for k, v in mp.items():
                out[f"map_{k}"][i, j] = out[f"map_{k}"][j, i] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--results")
    ap.add_argument("--unseal-test", action="store_true",
                    help="evaluate on the held-out test actors (only after the configuration is frozen)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.results) if args.results else cfg["data"]["results_dir"]
    run_cfg = json.loads((res / "config.json").read_text()) if (res / "config.json").exists() else {}
    for section in ("preprocess", "lot", "inference", "split"):
        cfg[section] = run_cfg.get(section, cfg.get(section))      # analyse with the settings inference used
    if "lot" in run_cfg:
        cfg["lot"] = saved_lot(cfg["lot"])
    split = run_cfg.get("fit_split")
    from grace_lot.lot_adapter.program import apply_semantics
    apply_semantics(cfg["lot"])
    d, an = cfg["data"], cfg["analysis"]
    out = res / "analysis"
    (out / "fits").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["inference"]["seed"])

    exemplars = study3_exemplars(d["similarity_csv"], d["raw_dir"], d["substitutes"])
    manners = sorted(exemplars)
    posts = {m: json.loads((res / "posteriors" / f"{m}.json").read_text()) for m in manners}
    if split is None:      # older runs: infer which actors the posteriors were fitted to
        actors = {v.split("_")[0] for p in posts.values() for v in p["videos"]}
        val = set(cfg["split"]["val_actors"]) if isinstance(cfg.get("split"), dict) else set()
        split = "exemplar" if all(len(p["videos"]) == 1 for p in posts.values()) else (
            "trainval" if actors & val else "train")
    posteriors = {m: {parse_program(s): w for s, w in posts[m]["posterior"]} for m in manners}
    maps = {m: parse_program(posts[m]["map"]) for m in manners}

    # Human data enter only here.
    ratings = load_ratings(d["similarity_csv"])
    subs = subjects(ratings)
    human = similarity_matrix(ratings, manners)
    ceiling = noise_ceiling(ratings, manners, subs, 200, rng)
    boot = bootstrap_humans(ratings, manners, subs, an["n_bootstrap"], rng)

    heldout = None
    sealed = set(cfg["split"]["test_actors"]) if "split" in cfg and cfg["split"] else set()
    if split == "exemplar":
        trajs = [load_trajectory(cfg, exemplars[m]) for m in manners]
        mats = {"human": human, **kinematic_matrices(trajs)}       # on the exemplar videos the raters saw
    else:
        # The test split is sealed: exemplar videos of test actors are not used. Kinematic
        # baselines use the same actors the LoT posteriors were fitted to.
        mats = {"human": human}
    if split in ("train", "trainval"):
        fit_videos = split_videos(d["raw_dir"], cfg["split"], split)
        assert not any(v.actor in sealed for vs in fit_videos.values() for v in vs), "test actor in fit set"
        trajs = [load_trajectory(cfg, fit_videos[m][0]) for m in manners]     # one training actor, for the fit figures
        fields = {m: np.stack([load_trajectory(cfg, v).fields for v in fit_videos[m]]) for m in manners}
        n = len(manners)
        proto = np.full((n, n), np.nan); pairwise = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = fields[manners[i]], fields[manners[j]]
                proto[i, j] = proto[j, i] = -euclid(a.mean(0), b.mean(0))
                pairwise[i, j] = pairwise[j, i] = -float(((a[:, None] - b[None]) ** 2).mean())
        mats["kin_manner_prototype"] = proto          # same actors the LoT posteriors were fitted to
        mats["kin_manner_pairwise"] = pairwise
        joints = {m: np.stack([load_trajectory(cfg, v).joints for v in fit_videos[m]]) for m in manners}
        jpair = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = joints[manners[i]], joints[manners[j]]
                jpair[i, j] = jpair[j, i] = -float(np.linalg.norm(a[:, None] - b[None], axis=-1).mean())
        mats["kin_manner_joints"] = jpair              # mean joint-position distance over actor pairs
        if split == "trainval" and args.unseal_test:  # the single, explicit use of the test actors
            heldout = heldout_evaluate(cfg, res / "posteriors", fit_videos,
                                       split_videos(d["raw_dir"], cfg["split"], "test"), joints=True,
                                       unsealed=True)
    mats.update(lot_matrices(posteriors, maps, manners, an["max_pairs"], cfg["inference"]["seed"]))
    np.savez_compressed(out / "matrices.npz", manners=np.array(manners), **mats)

    kin_names = [k for k in mats if k.startswith("kin_")]
    models = {}
    for name, m in mats.items():
        if name == "human":
            continue
        models[name] = {**mantel(m, human, an["n_permutations"], rng), **bootstrap_ci(m, boot)}
        if name.startswith(("lot_", "map_")):
            models[name].update(incremental(human, [mats[k] for k in kin_names], m, 2000, rng))
            models[name]["cv_rho_kin_plus_this"] = loo_manner_cv(human, [mats[k] for k in kin_names] + [m])
    cv_kin = loo_manner_cv(human, [mats[k] for k in kin_names])

    # Largest successes and failures of the full LoT measure (by rank residual).
    from scipy.stats import rankdata
    iu = np.triu_indices(len(manners), 1)
    rh, rm = rankdata(upper(human)), rankdata(upper(mats["lot_full"]))
    resid = rm - rh
    pairs = [(manners[a], manners[b]) for a, b in zip(*iu)]
    order = np.argsort(np.abs(resid))
    report = {
        "best_agreement": [(pairs[k], float(resid[k])) for k in order[:10]],
        "model_too_similar": [(pairs[k], float(resid[k])) for k in np.argsort(-resid)[:10]],
        "model_too_dissimilar": [(pairs[k], float(resid[k])) for k in np.argsort(resid)[:10]],
    }
    diagnostics = {m: {k: posts[m][k] for k in ("r_hat_log_post", "unique_programs", "map", "map_distance")}
                   | {"acceptance": [round(c["acceptance"], 3) for c in posts[m]["chains"]]}
                   for m in manners}
    summary = {"split": split, "heldout_test": heldout,
               "manners": manners, "substituted": substituted(exemplars, d["similarity_csv"]),
               "noise_ceiling": ceiling, "cv_rho_kinematic_only": cv_kin, "models": models,
               "residuals_lot_full": report, "inference": diagnostics, "config": {
                   k: cfg[k] for k in ("preprocess", "lot", "inference", "analysis")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    _write_markdown(out / "summary.md", summary)

    # Figures
    order = np.argsort(-np.nanmean(human, axis=1))
    kin_show = [k for k in ("kin_fields", "kin_joints_dtw", "kin_manner_pairwise", "kin_manner_joints") if k in mats]
    plots.heatmaps({"human": human, kin_show[0]: mats[kin_show[0]], "lot_full": mats["lot_full"],
                    "lot_flat": mats["lot_flat"]}, manners, out / "heatmaps.png", order)
    plots.scatter(human, {k: mats[k] for k in kin_show + ["lot_flat", "lot_tree_ordered", "lot_full"]},
                  out / "scatter.png")
    traces = {m: np.load(res / "traces" / f"{m}.npz")["log_post"] for m in manners}
    plots.traces(traces, cfg["inference"]["burn_in"], out / "traces.png")
    for m, t in zip(manners, trajs):
        lot_fields = rollout(maps[m], cfg["preprocess"]["frames"], cfg["preprocess"]["field_size"])
        preview(t, out / "fits" / f"{m}.png", f"{m}  MAP: {maps[m]}", lot_fields=lot_fields)
    print((out / "summary.md").read_text())


def _write_markdown(path: Path, s: dict) -> None:
    lines = ["# GRACE x LoT: similarity analysis", "", f"Posteriors fitted on: {s['split']}", ""]
    if s.get("heldout_test"):
        lines += ["## Held-out test actors: recognition (chance top-1 = 0.038, top-5 = 0.192)", "",
                  "| model | top-1 | top-5 | MRR | n |", "| --- | --- | --- | --- | --- |"]
        for k, v in s["heldout_test"].items():
            if isinstance(v, dict) and "top1" in v:
                lines.append(f"| {k} | {v['top1']:.3f} | {v['top5']:.3f} | {v['mrr']:.3f} | {v['n']} |")
        lines.append("")
    if s["substituted"]:
        lines += [f"**Substituted exemplars:** {s['substituted']} (the original video is not in the public release).", ""]
    nc = s["noise_ceiling"]
    lines += [f"Noise ceiling (split-half, Spearman-Brown): rho = {nc['split_half_sb']:.3f} +/- {nc['sd']:.3f}",
              f"Cross-validated rho, kinematic baselines only: {s['cv_rho_kinematic_only']:.3f}", "",
              "| model | rho | 95% CI | Mantel p | dR2 over kinematic | p(dR2) | CV rho kin+model |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in s["models"].items():
        extra = (f"{r['delta_r2']:.3f} | {r['p_delta']:.4f} | {r['cv_rho_kin_plus_this']:.3f}"
                 if "delta_r2" in r else "- | - | -")
        lines.append(f"| {name} | {r['rho']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p']:.4f} | {extra} |")
    lines += ["", "## Inference diagnostics", "", "| manner | R-hat | unique programs | MAP D | MAP program |",
              "| --- | --- | --- | --- | --- |"]
    for m, dg in s["inference"].items():
        lines.append(f"| {m} | {dg['r_hat_log_post']:.2f} | {dg['unique_programs']} | {dg['map_distance']:.4f} | `{dg['map']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
