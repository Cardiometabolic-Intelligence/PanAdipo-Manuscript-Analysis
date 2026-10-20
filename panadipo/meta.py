"""Random-effects meta-analysis of cohort-specific log hazard ratios: restricted maximum likelihood (REML) tau²,
Cochran Q, I² with a Q-profile confidence interval, and the modified Hartung-Knapp-Sidik-Jonkman (HKSJ) interval."""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy import stats as st

Z = st.norm.ppf(0.975)


def cochran_q(y, v) -> tuple[float, int, float]:
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    q, df = float((w * (y - mu) ** 2).sum()), len(y) - 1
    return q, df, float(st.chi2.sf(q, df)) if df > 0 else np.nan


def tau2_reml(y, v) -> float:
    def nll(t2):
        w = 1 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(v + t2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())

    upper = max(10 * np.var(y, ddof=1) if len(y) > 1 else 1.0, 1e-4)
    res = optimize.minimize_scalar(nll, bounds=(0, upper), method="bounded", options={"xatol": 1e-10})
    return 0.0 if nll(0.0) <= nll(res.x) else float(res.x)


def i2_ci(y, v, level: float = 0.95) -> tuple[float, float]:
    """Q-profile CI for tau² (Viechtbauer 2007), expressed as I² with the typical within-cohort variance."""
    k = len(y)
    if k < 2:
        return np.nan, np.nan

    def q(t2):
        w = 1 / (v + t2)
        mu = (w * y).sum() / w.sum()
        return (w * (y - mu) ** 2).sum()

    upper = 100 * (np.var(y, ddof=1) + v.max()) + 1

    def solve(crit):
        if q(0) < crit:
            return 0.0
        hi = upper
        for _ in range(60):                     # widen the bracket when the critical value is tiny (two cohorts)
            if q(hi) < crit:
                return optimize.brentq(lambda t: q(t) - crit, 0, hi)
            hi *= 10
        return np.inf

    t_lo, t_hi = solve(st.chi2.ppf(1 - (1 - level) / 2, k - 1)), solve(st.chi2.ppf((1 - level) / 2, k - 1))
    w = 1 / v
    s2 = (k - 1) * w.sum() / (w.sum() ** 2 - (w ** 2).sum())
    i2 = lambda t: 1.0 if np.isinf(t) else t / (t + s2)  # noqa: E731
    return float(i2(t_lo)), float(i2(t_hi))


def pool(log_hr, se) -> dict:
    """REML random-effects pooled HR with 95% CI, cohort weights, Q, and I² (95% CI)."""
    y, se = np.asarray(log_hr, float), np.asarray(se, float)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, v = y[ok], se[ok] ** 2
    if len(y) == 0:
        return {"k": 0, "HR": np.nan, "lo": np.nan, "hi": np.nan}
    t2 = tau2_reml(y, v) if len(y) > 1 else 0.0
    w = 1 / (v + t2)
    mu, s = (w * y).sum() / w.sum(), np.sqrt(1 / w.sum())
    q, df, qp = cochran_q(y, v)
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    i2_lo, i2_hi = i2_ci(y, v)
    return {"k": int(len(y)), "logHR": float(mu), "se": float(s), "HR": float(np.exp(mu)),
            "lo": float(np.exp(mu - Z * s)), "hi": float(np.exp(mu + Z * s)), "p": float(2 * st.norm.sf(abs(mu / s))),
            "tau2": t2, "Q": q, "Q_p": qp, "I2": i2, "I2_lo": i2_lo, "I2_hi": i2_hi,
            **hksj(y, v, t2), "weights": (w / w.sum()).tolist()}


def hksj(y, v, tau2: float) -> dict:
    """Modified HKSJ interval: variance scaled by q* = max(1, Q_w / (k - 1)) and a t distribution with k - 1 degrees
    of freedom, so the interval is never narrower than the REML interval (wide with few cohorts)."""
    k = len(y)
    if k < 2:
        return {"HKSJ_lo": np.nan, "HKSJ_hi": np.nan, "HKSJ_p": np.nan}
    w = 1 / (v + tau2)
    mu = (w * y).sum() / w.sum()
    se = np.sqrt(max(1.0, (w * (y - mu) ** 2).sum() / (k - 1)) / w.sum())
    t = st.t.ppf(0.975, k - 1)
    return {"HKSJ_lo": float(np.exp(mu - t * se)), "HKSJ_hi": float(np.exp(mu + t * se)),
            "HKSJ_p": float(2 * st.t.sf(abs(mu / se), k - 1))}
