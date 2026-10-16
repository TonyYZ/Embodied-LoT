"""Compare model and human similarity matrices.

* Spearman rho over the upper triangle
* a Mantel permutation test (model rows and columns permuted together)
* a bootstrap CI over participants (the human matrix is resampled, the model is held fixed)
* a split-half noise ceiling (Spearman-Brown corrected)
* incremental fit: human ~ kinematic + LoT on ranks, with a permutation test
  of the LoT increment and leave-one-manner-out cross-validation
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from grace_lot.analysis.human_similarity import similarity_matrix


def upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(len(m), 1)]


def rho(a: np.ndarray, b: np.ndarray) -> float:
    x, y = upper(a), upper(b)
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(spearmanr(x[ok], y[ok]).statistic)


def mantel(model: np.ndarray, human: np.ndarray, n: int, rng: np.random.Generator) -> Dict[str, float]:
    observed = rho(model, human)
    null = np.empty(n)
    for k in range(n):
        p = rng.permutation(len(model))
        null[k] = rho(model[np.ix_(p, p)], human)
    return {"rho": observed, "p": float((np.sum(null >= observed) + 1) / (n + 1)),
            "null_95": float(np.quantile(null, 0.95))}


def bootstrap_humans(ratings, manners, subjects: Sequence[str], n: int,
                     rng: np.random.Generator) -> List[np.ndarray]:
    """Human matrices from participant-resampled data (computed once, reused for every model)."""
    return [similarity_matrix(ratings, manners, rng.choice(subjects, len(subjects))) for _ in range(n)]


def bootstrap_ci(model: np.ndarray, humans: Sequence[np.ndarray]) -> Dict[str, float]:
    vals = [rho(model, h) for h in humans]
    return {"ci_low": float(np.quantile(vals, 0.025)), "ci_high": float(np.quantile(vals, 0.975))}


def noise_ceiling(ratings, manners, subjects: Sequence[str], n: int, rng: np.random.Generator) -> Dict[str, float]:
    """Split-half reliability of the human matrix over random halves of participants."""
    vals = []
    for _ in range(n):
        p = rng.permutation(subjects)
        h = len(p) // 2
        r = rho(similarity_matrix(ratings, manners, p[:h]), similarity_matrix(ratings, manners, p[h:]))
        vals.append(2 * r / (1 + r))
    return {"split_half_sb": float(np.mean(vals)), "sd": float(np.std(vals))}


# ---- incremental variance --------------------------------------------------

def _ranked(m: np.ndarray) -> np.ndarray:
    v = upper(m)
    return (rankdata(v) - 1) / (len(v) - 1)


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(1 - resid.var() / y.var())


def incremental(human: np.ndarray, kinematic: Sequence[np.ndarray], lot: np.ndarray, n: int,
                rng: np.random.Generator) -> Dict[str, float]:
    """R^2 of human ~ kinematic (+ LoT), on ranks. The LoT matrix is label-permuted to give a null."""
    y = _ranked(human)
    K = np.column_stack([_ranked(k) for k in kinematic])
    base = _r2(y, K)
    full = _r2(y, np.column_stack([K, _ranked(lot)]))
    null = np.empty(n)
    for k in range(n):
        p = rng.permutation(len(lot))
        null[k] = _r2(y, np.column_stack([K, _ranked(lot[np.ix_(p, p)])])) - base
    return {"r2_kinematic": base, "r2_kinematic_plus_lot": full, "delta_r2": full - base,
            "p_delta": float((np.sum(null >= full - base) + 1) / (n + 1))}


def loo_manner_cv(human: np.ndarray, predictors: Sequence[np.ndarray]) -> float:
    """Leave one manner out: fit on pairs without it, predict pairs with it. Return Spearman of all predictions."""
    n = len(human)
    iu = np.triu_indices(n, 1)
    y = _ranked(human)
    X = np.column_stack([np.ones(len(y))] + [_ranked(p) for p in predictors])
    pred, count = np.zeros(len(y)), np.zeros(len(y))
    for m in range(n):
        test = (iu[0] == m) | (iu[1] == m)
        beta, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
        pred[test] += X[test] @ beta
        count[test] += 1
    # Each pair is held out twice (once for each of its manners); average the two predictions.
    return float(spearmanr(pred / count, y).statistic)
