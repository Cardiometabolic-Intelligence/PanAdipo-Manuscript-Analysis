"""Paired echocardiography-CCTA cohort: sample selection, EAT volume indexed to total heart volume, and age- and
sex-specific high EAT (Methods, "Cross-sectional radiomic body and organ analysis"; Supplemental Methods).

CT measures come from the Body and Organ Analysis (BOA) pipeline: adipose tissue is -190 to -30 HU; EAT lies within
the pericardium and paracardial fat in the mediastinum outside the pericardium (Supplemental Table 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HEART = ["myocardium_volume", "lv_volume", "rv_volume", "la_volume", "ra_volume"]   # LA includes the appendage

# Supplemental Table 1 / Figure 2B measures: column -> label
VOLUMES = {
    "eat_volume_heart_indexed": "EAT volume indexed to total heart volume",
    "eat_volume_bsa_indexed": "EAT volume indexed to BSA",
    "eat_fraction_pericardium": "EAT fraction of pericardial sac volume",
    "eat_volume": "EAT volume",
    "paracardial_volume": "Paracardial adipose tissue volume",
    "thoracic_volume": "Thoracic adipose tissue (ThAT; epicardial + paracardial) volume",
    "sat_volume": "Subcutaneous adipose tissue volume",
    "imat_volume": "Intermuscular adipose tissue volume",
    "pericardium_volume": "Pericardial sac volume",
    "lv_volume": "LV volume", "rv_volume": "RV volume", "la_volume": "LA volume", "ra_volume": "RA volume",
    "myocardium_volume": "Myocardial volume",
    "bmi": "Body mass index",
}
ATTENUATION = {"eat_mean_hu": "EAT mean attenuation", "eat_median_hu": "EAT median attenuation",
               "paracardial_mean_hu": "Paracardial mean attenuation", "sat_mean_hu": "SAT mean attenuation"}


def primary_sample(pairs: pd.DataFrame, max_days: int = 365) -> pd.DataFrame:
    """Primary held-out sample: pairs within `max_days` from patients not used for model training or validation, one
    pair per patient (shortest absolute interval; ties broken by the earliest echocardiogram)."""
    p = pairs.assign(abs_days=pairs["interval_days"].abs())
    p = p[(p["abs_days"] <= max_days) & ~p["development"].astype(bool)]
    return p.sort_values(["patient_id", "abs_days", "echo_date"]).drop_duplicates("patient_id").reset_index(drop=True)


def add_derived(d: pd.DataFrame, heart_ml=(200, 1500), min_structure_ml: float = 10) -> pd.DataFrame:
    """ThAT volume, total heart volume, EAT volume / total heart volume (missing for implausible cardiac
    segmentations, which are excluded rather than corrected), and EAT volume / BSA (mL/m², when `bsa` is given)."""
    hv = d[HEART].apply(pd.to_numeric, errors="coerce")
    total = hv.sum(axis=1, min_count=len(HEART))
    plausible = (hv.min(axis=1) >= min_structure_ml) & total.between(*heart_ml)
    out = d.assign(thoracic_volume=d["eat_volume"] + d["paracardial_volume"], heart_volume=total,
                   eat_volume_heart_indexed=(d["eat_volume"] / total).where(plausible))
    if "bsa" in d:
        out["eat_volume_bsa_indexed"] = d["eat_volume"] / pd.to_numeric(d["bsa"], errors="coerce")
    return out


def qc_flow(d: pd.DataFrame, heart_ml=(200, 1500), min_structure_ml: float = 10) -> pd.DataFrame:
    """Patients excluded from the heart-indexed analyses by the segmentation plausibility checks (after add_derived)."""
    hv = d[HEART].apply(pd.to_numeric, errors="coerce")
    eat, miss = d["eat_volume"].notna(), hv.isna().any(axis=1)
    small = ~miss & (hv.min(axis=1) < min_structure_ml)
    outside = ~miss & ~small & ~d["heart_volume"].between(*heart_ml)
    bmi = d["eat_volume_heart_indexed"].notna() & d[["age", "female", "bmi"]].notna().all(axis=1)
    rows = [("Primary held-out sample (one pair per patient)", len(d)), ("Missing EAT volume", int((~eat).sum())),
            ("Any heart structure not segmented", int((eat & miss).sum())),
            (f"Any heart structure <{min_structure_ml:g} mL", int((eat & small).sum())),
            (f"Total heart volume outside {heart_ml[0]}-{heart_ml[1]:,} mL", int((eat & outside).sum())),
            ("EAT volume indexed to heart volume available", int(d["eat_volume_heart_indexed"].notna().sum())),
            ("... and age, sex, and BMI available (R² and Figure 3 sample)", int(bmi.sum()))]
    return pd.DataFrame(rows, columns=["step", "patients"])


def age_group(age: pd.Series, cuts=(50, 65)) -> pd.Series:
    a, b = cuts
    return pd.cut(age, [-np.inf, a, b, np.inf], right=False, labels=[f"<{a}", f"{a}-{b - 1}", f">={b}"])


def high_eat(d: pd.DataFrame, col: str = "eat_volume_heart_indexed", q: float = 0.95, cuts=(50, 65)) -> pd.Series:
    """1 if `col` is at or above its q-th percentile within age (<50, 50-64, >=65 years) x sex groups of the sample."""
    g = [age_group(d["age"], cuts), d["female"]]
    thr = d.groupby(g, observed=True)[col].transform(lambda v: v.quantile(q))
    return (d[col] >= thr).astype(float).where(d[col].notna())
