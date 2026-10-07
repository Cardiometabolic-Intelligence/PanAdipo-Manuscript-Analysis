"""Landmark analyses of newly documented disease and mortality (Methods, "Longitudinal cardiometabolic outcomes").

Cohort tables have one row per patient (index = earliest echocardiogram) with: patient_id, index_date, score (study
score), age, female, bmi, race_ethnicity, death_date, last_known_date (optional; last recorded contact), the first
diagnosis date of each endpoint (<endpoint>_date; see panadipo.outcomes), and optionally alcohol_excluded (MIMIC-IV).

Design: patients with the endpoint on or before the index date are excluded (all available history); follow-up ends
at the outcome, death (censored: cause-specific hazards), last recorded contact, 5 years, or the end of data;
follow-up starts after a blanking window (landmark: 30, 90, or 365 days), and patients with the outcome, death, or
censoring during the window are excluded. Exposure: log(score), standardized within each cohort.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from scipy import stats as st

EXPOSURE = "log_score_z"
AGE_SEX = ["age", "female"]
AGE_SEX_BMI = ["age", "female", "bmi"]
ADJUSTMENTS = {"Age, sex": AGE_SEX, "Age, sex, BMI": AGE_SEX_BMI}
FINE_GRAY_R = Path(__file__).with_name("fine_gray.R")


def add_exposure(df: pd.DataFrame, floor: float = 1e-3) -> pd.DataFrame:
    """log(score) floored at `floor`, standardized within the cohort (per SD of log[PanAdipo])."""
    lp = np.log(pd.to_numeric(df["score"], errors="coerce").clip(lower=floor, upper=1 - 1e-6))
    return df.assign(log_score=lp, log_score_z=(lp - lp.mean()) / lp.std(ddof=1))


def _dates(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce") if col in df else pd.Series(pd.NaT, index=df.index)


def at_risk(df: pd.DataFrame, endpoint: str, max_years: float = 5, admin_date=None) -> pd.DataFrame:
    """Patients free of `endpoint` at the index date, with event (0/1), time_years, end_date, and competing death.

    endpoint: an <endpoint>_date column stem (e.g., "masld") or "death" for all-cause mortality.
    admin_date: end of data; defaults to the latest date recorded in the cohort table.
    """
    d = df.copy()
    d["index_date"] = _dates(d, "index_date")
    if endpoint == "masld" and "alcohol_excluded" in d:
        d = d[~d["alcohol_excluded"].fillna(False).astype(bool)]
    outcome = _dates(d, "death_date" if endpoint == "death" else f"{endpoint}_date")
    d = d[~(outcome <= d["index_date"])].copy()                        # prevalent disease excluded
    outcome = outcome.loc[d.index]
    death, last = _dates(d, "death_date"), _dates(d, "last_known_date")
    if admin_date is None:
        cand = [pd.to_datetime(d[c], errors="coerce").max() for c in d.columns if c.endswith("_date")
                and c != "index_date"]
        admin_date = max([c for c in cand if pd.notna(c)], default=d["index_date"].max())
    ends = [d["index_date"] + pd.Timedelta(days=int(max_years * 365.25)),
            pd.Series(pd.Timestamp(admin_date), index=d.index), last]
    if endpoint != "death":
        ends.append(death)                                               # death censors (cause-specific)
    d["fup_end"] = pd.concat(ends, axis=1).min(axis=1, skipna=True)
    event = outcome.notna() & (outcome > d["index_date"]) & (outcome <= d["fup_end"])
    d["event"] = event.astype(int)
    d["end_date"] = outcome.where(event, d["fup_end"])
    d["time_years"] = (d["end_date"] - d["index_date"]).dt.days / 365.25
    d = d[d["time_years"] > 0].copy()
    d["time_years"] = d["time_years"].clip(upper=max_years)
    # competing death for Fine-Gray: died without the endpoint, with death ending follow-up
    d["status"] = np.where(d["event"] == 1, 1, np.where(_dates(d, "death_date") <= d["end_date"], 2, 0))
    if endpoint == "death":
        d["status"] = d["event"]
    return d


def landmark(ar: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep patients still at risk at `days` after the index date and restart the clock there."""
    t = days / 365.25
    return ar[ar["time_years"] > t].assign(time_years=lambda x: x["time_years"] - t)


def _design(d: pd.DataFrame, covs: list[str]) -> tuple[pd.DataFrame, list[str]]:
    covs = [EXPOSURE] + [c for c in covs if c in d and d[c].nunique(dropna=True) > 1]
    x = d[["time_years", "event"] + covs].apply(pd.to_numeric, errors="coerce").dropna()
    return x[x["time_years"] > 0], covs


def cox(d: pd.DataFrame, covs: list[str], min_events: int = 5, min_n: int = 20) -> dict:
    """Cox model with Huber-White robust standard errors; HR per SD of log(PanAdipo)."""
    x, covs = _design(d, covs)
    out = {"n": len(x), "events": int(x["event"].sum())}
    if out["events"] < min_events or len(x) < min_n:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = CoxPHFitter().fit(x, "time_years", "event", robust=True).summary.loc[EXPOSURE]
    return {**out, "logHR": float(s["coef"]), "se": float(s["se(coef)"]), "HR": float(s["exp(coef)"]),
            "lo": float(s["exp(coef) lower 95%"]), "hi": float(s["exp(coef) upper 95%"]), "p": float(s["p"])}


def fine_gray(d: pd.DataFrame, covs: list[str]) -> dict:
    """Fine-Gray subdistribution HR (death before the endpoint as the competing event), fitted in R with
    survival::finegray and a weighted Cox model with robust SEs. Requires Rscript with the survival package."""
    if shutil.which("Rscript") is None:
        return {}
    x = d[["time_years", "status", EXPOSURE] + covs].apply(pd.to_numeric, errors="coerce").dropna()
    x = x[x["time_years"] > 0].reset_index(drop=True).assign(id=lambda z: np.arange(len(z)))
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        x.to_csv(f.name, index=False)
    res = subprocess.run(["Rscript", str(FINE_GRAY_R), f.name, ",".join([EXPOSURE] + covs)],
                         capture_output=True, text=True, check=True)
    Path(f.name).unlink()
    b, se = map(float, res.stdout.strip().splitlines()[-1].split(","))
    return {"n": len(x), "events": int((x["status"] == 1).sum()), "competing_deaths": int((x["status"] == 2).sum()),
            "logHR": b, "se": se, "HR": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se)}


# ----------------------------------------------------------------------------- missing BMI
def impute_bmi(d: pd.DataFrame, m: int, rng: np.random.Generator) -> list[pd.Series]:
    """Multiple imputation of missing BMI: m draws from a Bayesian linear regression of log(BMI) on age, sex,
    log(PanAdipo), the event indicator, the Nelson-Aalen cumulative hazard, and race and ethnicity (BMI is the only
    incomplete variable, so chained equations reduce to this single model)."""
    na = NelsonAalenFitter().fit(d["time_years"], d["event"]).cumulative_hazard_at_times(d["time_years"]).values
    X = pd.get_dummies(d[["age", "female", EXPOSURE, "event", "race_ethnicity"]].assign(na=na),
                       columns=["race_ethnicity"], drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    obs = d["bmi"].notna().values
    y, Xo, Xm = np.log(d.loc[obs, "bmi"].values), X.values[obs], X.values[~obs]
    beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
    dof = len(y) - Xo.shape[1]
    s2 = ((y - Xo @ beta) ** 2).sum() / dof
    L = np.linalg.cholesky(np.linalg.pinv(Xo.T @ Xo))
    draws = []
    for _ in range(m):
        sigma2 = s2 * dof / rng.chisquare(dof)
        b = beta + np.sqrt(sigma2) * L @ rng.standard_normal(len(beta))
        bmi = d["bmi"].copy()
        bmi.loc[~obs] = np.exp(Xm @ b + np.sqrt(sigma2) * rng.standard_normal(len(Xm)))
        draws.append(bmi)
    return draws


def rubin(est, var) -> tuple[float, float]:
    """Rubin's rules: pooled estimate and total standard error."""
    est, var = np.asarray(est, float), np.asarray(var, float)
    m = len(est)
    return float(est.mean()), float(np.sqrt(var.mean() + (1 + 1 / m) * est.var(ddof=1)))


# ----------------------------------------------------------------------------- discrimination and model checks
def harrell_c(d: pd.DataFrame, covs: list[str]) -> float:
    """Apparent Harrell's C of a Cox model with `covs`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CoxPHFitter().fit(d[["time_years", "event"] + covs], "time_years", "event")
    return float(concordance_index(d["time_years"], -m.predict_partial_hazard(d[covs]), d["event"]))


def rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell): linear term plus k-2 nonlinear terms."""
    k, cols = np.asarray(knots, float), [x]
    for j in range(len(k) - 2):
        cols.append((np.maximum(x - k[j], 0) ** 3
                     - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0]) ** 2)
    return np.column_stack(cols)


def model_checks(d: pd.DataFrame, covs: list[str]) -> dict:
    """Schoenfeld test for log(PanAdipo) (rank-transformed time) and a likelihood-ratio test of a 4-knot restricted
    cubic spline against the linear term (log-linearity)."""
    x, covs = _design(d, covs)
    if x["event"].sum() < 20:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin = CoxPHFitter().fit(x, "time_years", "event")
        ph = proportional_hazard_test(lin, x, time_transform="rank").summary.loc[EXPOSURE, "p"]
        B = rcs(x[EXPOSURE].values, np.percentile(x[EXPOSURE], [5, 35, 65, 95]))
        xs = x.drop(columns=EXPOSURE).assign(**{f"rcs{i}": B[:, i] for i in range(B.shape[1])})
        spl = CoxPHFitter().fit(xs, "time_years", "event")
    lr = 2 * (spl.log_likelihood_ - lin.log_likelihood_)
    return {"schoenfeld_p": float(ph), "nonlinearity_p": float(st.chi2.sf(lr, B.shape[1] - 1))}
