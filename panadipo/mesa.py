"""MESA Exam 6 cross-sectional biomarker models (Methods, "Cross-sectional cardiometabolic associations in MESA").

z(biomarker) ~ z(log PanAdipo) + covariates by ordinary least squares, so beta is the SD difference in the biomarker
per SD of log(PanAdipo). Continuous covariates are standardized; binary indicators are not.
  M1: age, sex                     M2: M1 + BMI                     M3: M1 + waist-hip ratio (alternative to M2)
  M4: M2 + race and ethnicity, field center, lipid-lowering, antihypertensive, and glucose-lowering medications
M1-M3 share one complete-case sample per biomarker (biomarker, PanAdipo, age, sex, BMI, and WHR non-missing); M4 also
requires its additional covariates. Variable names follow the MESA Exam 6 data dictionary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .stats import zscore

BIOMARKERS = {  # label -> MESA variable; log HOMA-IR is the primary biomarker
    "log HOMA-IR": "log_homa_ir", "Triglycerides": "trig6", "HDL": "hdl6", "LDL": "ldl6", "Total cholesterol": "chol6",
    "Fasting glucose": "glucose6", "HbA1c": "hba1c6", "SBP": "sbp6c", "DBP": "dbp6c", "eGFR (CKD-EPI)": "cepgfr6c",
}
BASE = ["age6c", "gender1"]
MODELS = {"M1": BASE, "M2": BASE + ["bmi6c"], "M3": BASE + ["waist_hip"]}
COMPLETE_CASE = ["panadipo_log", "age6c", "gender1", "bmi6c", "waist_hip"]
MEDICATIONS = ["lipid6c", "htnmed6c", "glu_lower"]


def prepare(df: pd.DataFrame, floor: float = 1e-4) -> pd.DataFrame:
    """Derived variables: WHR, log HOMA-IR (insulin x glucose / 405), log(PanAdipo), glucose-lowering medication, and
    indicators for race and ethnicity (race1c) and field center (reference: first level)."""
    d = df.copy()
    d["waist_hip"] = d["waistcm6"] / d["hipcm6"].replace(0, np.nan)
    d["log_homa_ir"] = np.log((d["insulin6"] * d["glucose6"] / 405).clip(lower=0.01))
    d["panadipo_log"] = np.log(d["score"].clip(lower=floor))
    known = d[["insln6c", "ohga6c"]].notna().any(axis=1)
    d["glu_lower"] = ((d["insln6c"] == 1) | (d["ohga6c"] == 1)).astype(float).where(known)
    for col, prefix in (("race1c", "race_"), ("field_center", "site_")):
        levels = sorted(d[col].dropna().unique())
        for lv in levels[1:]:
            d[f"{prefix}{lv}"] = (d[col] == lv).astype(float).where(d[col].notna())
    return d


def m4_covariates(d: pd.DataFrame) -> list[str]:
    dummies = [c for c in d.columns if c.startswith(("race_", "site_"))]
    return MODELS["M2"] + dummies + MEDICATIONS


def standardized_beta(d: pd.DataFrame, outcome: str, covs: list[str], binary: list[str]) -> dict:
    x = d[[outcome, "panadipo_log"] + covs].apply(pd.to_numeric, errors="coerce").dropna()
    covs = [c for c in covs if x[c].nunique() > 1]                     # e.g., an indicator constant in a subgroup
    if len(x) < 30:
        return {"n": len(x)}
    X = pd.DataFrame({"panadipo_log": zscore(x["panadipo_log"])})
    for c in covs:
        X[c] = x[c] if c in binary else zscore(x[c])
    fit = sm.OLS(zscore(x[outcome]), sm.add_constant(X, has_constant="add")).fit()
    lo, hi = fit.conf_int().loc["panadipo_log"]
    return {"n": len(x), "beta": fit.params["panadipo_log"], "lo": lo, "hi": hi, "p": fit.pvalues["panadipo_log"]}


def run_models(d: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    m4 = m4_covariates(d)
    binary = ["gender1"] + [c for c in m4 if c.startswith(("race_", "site_"))] + MEDICATIONS
    rows = []
    for label, col in BIOMARKERS.items():
        for model, covs in {**MODELS, "M4": m4}.items():
            cc = list(dict.fromkeys([col] + COMPLETE_CASE + (m4 if model == "M4" else [])))
            sub = d[cc].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < min_n:
                continue
            rows.append({"biomarker": label, "model": model, **standardized_beta(sub, col, covs, binary)})
    return pd.DataFrame(rows)
