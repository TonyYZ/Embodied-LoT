"""Figures: heatmaps, human-vs-model scatter, residuals, MAP fits and MCMC traces."""

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from grace_lot.analysis.rsa import upper


def heatmaps(mats: Dict[str, np.ndarray], manners: Sequence[str], path, order=None) -> None:
    order = np.arange(len(manners)) if order is None else order
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 5))
    for ax, (name, m) in zip(np.atleast_1d(axes), mats.items()):
        im = ax.imshow(m[np.ix_(order, order)], cmap="viridis")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
        ax.set_xticklabels([manners[i] for i in order], rotation=90, fontsize=6)
        ax.set_yticklabels([manners[i] for i in order], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def scatter(human: np.ndarray, models: Dict[str, np.ndarray], path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    y = upper(human)
    for ax, (name, m) in zip(np.atleast_1d(axes), models.items()):
        ax.scatter(rankdata(upper(m)), rankdata(y), s=6, alpha=0.6)
        ax.set_xlabel(f"{name} (rank)"); ax.set_ylabel("human (rank)"); ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def traces(traces_by_manner: Dict[str, np.ndarray], burn_in: int, path) -> None:
    names = sorted(traces_by_manner)
    cols = 6
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 1.8), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for chain in traces_by_manner[name]:
            ax.plot(chain[::50], lw=0.6)
        ax.axvline(burn_in / 50, color="k", lw=0.5, ls="--")
        ax.set_title(name, fontsize=7); ax.tick_params(labelsize=5)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
