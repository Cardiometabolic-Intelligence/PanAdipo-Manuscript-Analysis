"""Cohort characteristics: Table 1, Supplemental Figure 1 (age, BMI, and score distributions), and the correlation of
PanAdipo with BMI in each cohort (Figure 3C).

Inputs: cohort_ynhhs_tte, cohort_ynhhs_pocus, cohort_mimic, cohort_mesa (one row per patient), ct_pairs.
Prevalent conditions are those with a first diagnosis date on or before the index date. In the hospital cohorts,
characteristics are summarized among records linked to the EHR (age and sex available; records without linked
diagnosis codes were excluded as part of quality control); the total count includes every scored study.
Outputs: table1.csv, supp_figure1_distributions.csv, figure3c_bmi_correlation.csv.
"""

import pandas as pd

from panadipo import ct, io, stats
from panadipo.survival import linked

COHORTS = {"cohort_ynhhs_tte": "YNHHS TTE", "cohort_ynhhs_pocus": "YNHHS POCUS", "cohort_mimic": "MIMIC-IV",
           "cohort_mesa": "MESA"}
CONDITIONS = {"htn": "Hypertension", "hld": "Hyperlipidemia", "t2d": "Type 2 diabetes",
              "masld": "Steatotic liver disease (MASLD/MASH codes)",
              "myocardial_infarction": "Myocardial infarction", "heart_failure": "Heart failure",
              "chronic_coronary_syndrome": "Chronic coronary syndrome", "stroke": "Stroke",
              "atrial_fibrillation": "Atrial fibrillation",
              "peripheral_vascular_disease": "Peripheral vascular disease"}
RACE = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Asian", "Other/Unknown"]


def n_pct(mask: pd.Series, denom: pd.Series | None = None) -> str:
    denom = mask.notna() if denom is None else denom
    k, n = int(mask[denom].sum()), int(denom.sum())
    return f"{k:,} ({100 * k / n:.1f}%)" if n else "-"


def describe(d: pd.DataFrame, hospital: bool = False) -> dict:
    r = {"Total counts (n)": f"{len(d):,}"}
    if hospital:
        d = linked(d)
        r["With linked ICD codes, n"] = f"{len(d):,}"
    r |= {"Age, mean ± SD (years)": f"{d['age'].mean():.1f} ± {d['age'].std():.1f}",
          "Female, n (%)": n_pct(d["female"] == 1, d["female"].notna())}
    for race in RACE:
        r[race] = n_pct(d["race_ethnicity"] == race, pd.Series(True, index=d.index))
    r["BMI, mean ± SD (kg/m²)"] = f"{d['bmi'].mean():.1f} ± {d['bmi'].std():.1f}"
    q = d["score"].quantile([0.5, 0.25, 0.75]).values
    r["PanAdipo, median [IQR]"] = f"{q[0]:.2f} [{q[1]:.2f}-{q[2]:.2f}]"
    idx = pd.to_datetime(d["index_date"], errors="coerce") if "index_date" in d else None
    for c, lab in CONDITIONS.items():
        if f"{c}_date" in d and idx is not None:
            prev = pd.to_datetime(d[f"{c}_date"], errors="coerce") <= idx
            if c == "masld" and "alcohol_excluded" in d:                 # alcohol-related codes not counted
                prev &= ~d["alcohol_excluded"].fillna(False).astype(bool)
            r[f"{lab}, n (%)"] = n_pct(prev, pd.Series(True, index=d.index))
        else:
            r[f"{lab}, n (%)"] = "-"
    return r


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    cohorts = {lab: io.read(a.data, name) for name, lab in COHORTS.items() if io.exists(a.data, name)}
    cols = {lab: describe(d, hospital=lab != "MESA") for lab, d in cohorts.items()}
    if io.exists(a.data, "ct_pairs"):
        prim = ct.primary_sample(io.read(a.data, "ct_pairs"), cfg["ct"]["max_interval_days"])
        # as in the other YNHHS cohorts: age at the echocardiogram and BMI within ±365 days of it
        prim = prim.drop(columns=[c for c in ("age", "bmi") if f"{c}_at_echo" in prim])
        prim = prim.rename(columns={"echo_date": "index_date", "age_at_echo": "age", "bmi_at_echo": "bmi"})
        cols["YNHHS TTE-CCTA"] = describe(prim)
    io.write(pd.DataFrame(cols).fillna("-").rename_axis("Characteristic").reset_index(), a.out, "table1")

    rows, rho = [], []
    for lab, d in cohorts.items():
        for var in ("age", "bmi", "score"):
            q = d[var].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).values
            rows.append({"cohort": lab, "variable": var, "n": int(d[var].notna().sum()),
                         **dict(zip(["p5", "p25", "median", "p75", "p95"], q))})
        b = d[d["bmi"].between(15, 60)]                                 # BMI 15-60 kg/m²
        r, p, n = stats.spearman(b["bmi"], b["score"])
        rho.append({"cohort": lab, "n": n, "rho": r, "p": p})
    io.write(pd.DataFrame(rows), a.out, "supp_figure1_distributions")
    io.write(pd.DataFrame(rho), a.out, "figure3c_bmi_correlation")


if __name__ == "__main__":
    main()
