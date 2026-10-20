"""Diagnosis code lists (Supplemental Table 2) and first-diagnosis dates from structured diagnosis records.

Codes are matched by prefix after removing dots (e.g., "E11" matches E11.9; "250.x0" is written out as 250.00-250.90).
ICD-9 lists apply to ICD-9 records and ICD-10 lists to ICD-10 records. For each endpoint, the first qualifying
diagnosis on or before the index echocardiogram defines prevalent disease (all available history), and the first
qualifying diagnosis after it defines the incident outcome (see panadipo.survival.at_risk).
"""

from __future__ import annotations

import re

import pandas as pd

ICD10 = {
    "t2d": ["E11", "O24.1"],
    "htn": ["I10", "I11", "I12", "I13", "I67.4", "O10", "O11"],
    "hld": ["E78"],
    "masld": ["K76.0", "K75.81"],
}
ICD9 = {
    "t2d": [f"250.{x}{y}" for x in range(10) for y in (0, 2)],     # type 1 codes 250.x1 and 250.x3 not included
    "htn": ["401", "402", "403", "404", "405"],
    "hld": ["272.0", "272.1", "272.2", "272.3", "272.4"],
    "masld": ["571.8"],
}
# Participants with any alcohol-related code at any time are excluded from the steatotic liver disease (MASLD/MASH)
# analyses in all three hospital cohorts (YNHHS: encounter diagnoses, problem list, and medical history; MIMIC-IV:
# hospital diagnoses)
ALCOHOL = {9: ["291", "303", "571.0", "571.1", "571.2", "571.3"], 10: ["F10", "K70"]}

# Baseline comorbidities shown in Table 1 (ICD-10 and ICD-9)
COMORBIDITIES = {
    "myocardial_infarction": ["I21", "I22", "I24.0", "I25.2", "410", "411.81", "412"],
    "heart_failure": ["I50", "I42", "428"],
    "chronic_coronary_syndrome": ["I20", "I25.1", "I25.4", "I25.5", "I25.6", "I25.8", "I25.9", "411", "413", "414"],
    "stroke": ["I63", "I60", "I61", "I62", "433", "434", "436", "431", "432"],
    "atrial_fibrillation": ["I48.0", "I48.1", "I48.2", "427.31"],
    "peripheral_vascular_disease": ["I70.2", "I72.0", "I72.1", "I72.3", "I72.4", "I72.5", "I72.6", "I72.8", "I74.2",
                                    "I74.3", "I74.4", "I74.5", "440.2", "440.3", "440.4", "443", "444"],
}

# code-defined steatotic liver disease: MASLD/MASH diagnosis codes
ENDPOINTS = {"t2d": "Type 2 diabetes", "htn": "Hypertension", "hld": "Hyperlipidemia",
             "masld": "Steatotic liver disease"}


def normalize(code) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).strip().upper())


def _matches(codes: pd.Series, prefixes: list[str]) -> pd.Series:
    if not prefixes:
        return pd.Series(False, index=codes.index)
    pattern = "^(?:" + "|".join(re.escape(normalize(p)) for p in prefixes) + ")"
    return codes.str.match(pattern, na=False)


def first_dates(dx: pd.DataFrame, classes: dict[int, dict[str, list[str]]] | None = None) -> pd.DataFrame:
    """First diagnosis date per patient and class.

    dx: long table with patient_id, date, icd_code, icd_version (9 or 10); in MIMIC-IV, the date is the admission date.
    classes: {icd_version: {class name: code prefixes}}; defaults to the four endpoints of Supplemental Table 2.
    Returns one row per patient with columns <class>_date.
    """
    classes = classes or {9: ICD9, 10: ICD10}
    d = dx.assign(code=dx["icd_code"].map(normalize), date=pd.to_datetime(dx["date"], errors="coerce"))
    hits = []
    for version, lists in classes.items():
        sub = d[d["icd_version"] == version]
        for name, prefixes in lists.items():
            hits.append(sub.loc[_matches(sub["code"], prefixes), ["patient_id", "date"]].assign(cls=f"{name}_date"))
    first = pd.concat(hits).dropna(subset=["date"]).groupby(["patient_id", "cls"])["date"].min()
    return first.unstack("cls").reset_index()


def alcohol_flag(dx: pd.DataFrame) -> pd.Series:
    """Patients with any alcohol-related code (excluded from the steatotic liver disease analyses)."""
    d = dx.assign(code=dx["icd_code"].map(normalize))
    hit = pd.Series(False, index=d.index)
    for version, prefixes in ALCOHOL.items():
        hit |= (d["icd_version"] == version) & _matches(d["code"], prefixes)
    return pd.Series(True, index=pd.Index(d.loc[hit, "patient_id"].unique(), name="patient_id"), name="alcohol")
