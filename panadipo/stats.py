"""Statistical helpers: patient-level (cluster) bootstrap, discrimination, Spearman and partial Spearman correlation,
Holm adjustment, and Youden thresholds."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ----------------------------------------------------------------------------- bootstrap
def cluster_bootstrap(df: pd.DataFrame, cluster: str, fn: Callable[[pd.DataFrame], float | list[float]],
                      n_boot: int, seed: int) -> np.ndarray:
    """Resample patients (clusters) with replacement, keeping all of their rows. Returns (n_boot, k); failed
    replicates are NaN. With one row per patient this is the ordinary bootstrap."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(cluster, sort=False).indices.values())
    reps = []
    for _ in range(n_boot):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            reps.append(np.atleast_1d(np.asarray(fn(df.iloc[rows]), float)))
        except Exception:  # noqa: BLE001 - e.g., a replicate without events
            reps.append(None)
    k = max(len(r) for r in reps if r is not None)
    return np.vstack([r if r is not None and len(r) == k else np.full(k, np.nan) for r in reps])


def ci(reps: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval per column, ignoring failed replicates."""
    a = (1 - level) / 2
    reps = np.asarray(reps, float).reshape(len(reps), -1)
    return np.nanquantile(reps, a, axis=0), np.nanquantile(reps, 1 - a, axis=0)


# ----------------------------------------------------------------------------- discrimination
def auroc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def auprc(y, s) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if y.sum() > 0 else np.nan


def youden_threshold(y, s) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def operating_point(y, s, thr: float) -> dict:
    y, pos = np.asarray(y).astype(int), np.asarray(s) >= thr
    tp, fp = int((pos & (y == 1)).sum()), int((pos & (y == 0)).sum())
    fn, tn = int((~pos & (y == 1)).sum()), int((~pos & (y == 0)).sum())
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp),
            "npv": div(tn, tn + fn)}


# ----------------------------------------------------------------------------- correlation
def spearman(x, y) -> tuple[float, float, int]:
    """Spearman rho, two-sided P, and n on pairwise-complete data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    r, p = st.spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def partial_spearman(x, y, covariates) -> tuple[float, float, int]:
    """Partial Spearman correlation: the ranks of x and y are each regressed on the ranks of the covariates (with an
    intercept) and the residuals are correlated; t-test P on n - 2 - k degrees of freedom."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(covariates, float).reshape(len(x), -1)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c).all(axis=1)
    n, k = int(ok.sum()), c.shape[1]
    if n < k + 4:
        return np.nan, np.nan, n
    X = np.column_stack([np.ones(n)] + [st.rankdata(c[ok, j]) for j in range(k)])
    res = [r - X @ np.linalg.lstsq(X, r, rcond=None)[0] for r in (st.rankdata(x[ok]), st.rankdata(y[ok]))]
    r = float(np.corrcoef(*res)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / max(1e-300, 1 - r * r))
    return r, float(2 * st.t.sf(abs(t), df)), n


def spearman_ci(df: pd.DataFrame, x: str, y: str, n_boot: int, seed: int, cluster: str = "patient_id",
                covariates: list[str] | None = None) -> dict:
    """(Partial) Spearman rho with a patient-level bootstrap 95% CI."""
    cols = [x, y] + (covariates or [])
    d = df[[cluster] + cols].apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name != cluster else s)
    d = d.dropna().reset_index(drop=True)
    if covariates:
        fn = lambda b: partial_spearman(b[x], b[y], b[covariates].values)[0]  # noqa: E731
    else:
        fn = lambda b: spearman(b[x], b[y])[0]  # noqa: E731
    r, p, n = partial_spearman(d[x], d[y], d[covariates].values) if covariates else spearman(d[x], d[y])
    lo, hi = ci(cluster_bootstrap(d, cluster, fn, n_boot, seed))
    return {"rho": r, "lo": float(lo[0]), "hi": float(hi[0]), "p": p, "n": n, "patients": int(d[cluster].nunique())}


# ----------------------------------------------------------------------------- multiplicity
def holm(p) -> np.ndarray:
    """Holm step-down adjusted P values (NaN preserved)."""
    p = np.asarray(p, float)
    adj, ok = np.full_like(p, np.nan), np.isfinite(p)
    m = int(ok.sum())
    if m:
        order = np.argsort(p[ok])
        stepped = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[ok][order]), 1)
        out = np.empty(m)
        out[order] = stepped
        adj[ok] = out
    return adj


def zscore(x) -> pd.Series:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else pd.Series(np.nan, index=x.index)
