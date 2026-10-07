"""PanAdipo and newly documented cardiometabolic disease and mortality (Results; Figure 5; Supplemental Tables 9-11;
Supplemental Figure 3).

Landmark Cox models (panadipo.survival) in YNHHS TTE, YNHHS POCUS, and MIMIC-IV for type 2 diabetes, hypertension,
hyperlipidemia, and MASLD/MASH; HR per SD of log(PanAdipo) adjusted for age and sex, and additionally for BMI
(complete case). Cohort log HRs are pooled with REML random effects. Sensitivity analyses at the 90-day landmark:
Fine-Gray models with death as a competing event, age- and sex-adjusted models in the BMI-complete sample, and
multiple imputation of missing BMI. All-cause mortality is analyzed per cohort, including MESA.

Inputs: cohort_ynhhs_tte, cohort_ynhhs_pocus, cohort_mimic, cohort_mesa (docs/DATA.md).
Outputs: supp_table9_pooled.csv, supp_table9_cohorts.csv (Figure 5B), supp_table10_sensitivity.csv,
supp_table11_mortality.csv, supp_figure3_flow.csv, model_checks.csv.
"""

import numpy as np
import pandas as pd

from panadipo import io, meta
from panadipo import survival as sv
from panadipo.outcomes import ENDPOINTS

HOSPITAL = {"cohort_ynhhs_tte": "YNHHS TTE", "cohort_ynhhs_pocus": "YNHHS POCUS", "cohort_mimic": "MIMIC-IV"}


def pooled(fits: list[dict]) -> dict:
    p = meta.pool([f.get("logHR", np.nan) for f in fits], [f.get("se", np.nan) for f in fits])
    return {"n": sum(f["n"] for f in fits), "events": sum(f["events"] for f in fits), **p}


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    L = cfg["longitudinal"]
    lm90 = L["primary_landmark_days"]
    floor = cfg["score"]["log_floor_longitudinal"]
    cohorts = {lab: sv.add_exposure(io.read(a.data, name), floor) for name, lab in HOSPITAL.items()}
    risk = {(e, c): sv.at_risk(d, e, L["max_followup_years"]).dropna(subset=[sv.EXPOSURE])
            for e in ENDPOINTS for c, d in cohorts.items()}

    # Supplemental Table 9 and Figure 5B: cohort-specific and pooled HRs
    pooled_rows, cohort_rows = [], []
    for e, elab in ENDPOINTS.items():
        for lm in L["landmarks_days"]:
            for adj, covs in sv.ADJUSTMENTS.items():
                fits = [sv.cox(sv.landmark(risk[e, c], lm), covs) for c in cohorts]
                p = pooled(fits)
                pooled_rows.append({"outcome": elab, "landmark_days": lm, "adjustment": adj,
                                    **{k: p[k] for k in ("n", "events", "HR", "lo", "hi", "I2", "I2_lo", "I2_hi",
                                                         "Q", "Q_p", "tau2")}})
                for c, f, w in zip(cohorts, fits, p["weights"]):
                    cohort_rows.append({"outcome": elab, "landmark_days": lm, "adjustment": adj, "cohort": c,
                                        "n": f["n"], "events": f["events"], "HR": f.get("HR"), "lo": f.get("lo"),
                                        "hi": f.get("hi"), "reml_weight": w})
    io.write(pd.DataFrame(pooled_rows), a.out, "supp_table9_pooled")
    io.write(pd.DataFrame(cohort_rows).query("landmark_days == @lm90"), a.out, "supp_table9_cohorts")

    # Supplemental Table 10: sensitivity analyses at the 90-day landmark
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for e, elab in ENDPOINTS.items():
        for adj, covs in sv.ADJUSTMENTS.items():
            land = {c: sv.landmark(risk[e, c], lm90) for c in cohorts}
            main_ = pooled([sv.cox(land[c], covs) for c in cohorts])
            fg = [sv.fine_gray(land[c], covs) for c in cohorts]
            row = {"outcome": elab, "adjustment": adj, "n": main_["n"], "events": main_["events"],
                   "HR": main_["HR"], "lo": main_["lo"], "hi": main_["hi"],
                   "I2": main_["I2"], "I2_lo": main_["I2_lo"], "I2_hi": main_["I2_hi"]}
            if all(fg):                      # skipped without Rscript
                p = meta.pool([f["logHR"] for f in fg], [f["se"] for f in fg])
                row.update({"competing_deaths": sum(f["competing_deaths"] for f in fg), "fine_gray_HR": p["HR"],
                            "fine_gray_lo": p["lo"], "fine_gray_hi": p["hi"]})
            if adj == "Age, sex":            # same model in the BMI-complete sample
                fits = [sv.cox(land[c].dropna(subset=["bmi"]), covs) for c in cohorts]
                label = "bmi_complete"
            else:                            # BMI multiply imputed (Rubin's rules within cohort)
                fits = []
                for c in cohorts:
                    dc = land[c].dropna(subset=["age", "female"]).reset_index(drop=True)
                    est = [sv.cox(dc.assign(bmi=b), covs) for b in sv.impute_bmi(dc, L["mi_imputations"], rng)]
                    q, se = sv.rubin([f["logHR"] for f in est], [f["se"] ** 2 for f in est])
                    fits.append({"logHR": q, "se": se})
                label = "bmi_imputed"
            p = meta.pool([f.get("logHR", np.nan) for f in fits], [f.get("se", np.nan) for f in fits])
            row.update({f"{label}_HR": p["HR"], f"{label}_lo": p["lo"], f"{label}_hi": p["hi"]})
            rows.append(row)
    io.write(pd.DataFrame(rows), a.out, "supp_table10_sensitivity")

    # Supplemental Table 11: all-cause mortality per cohort (MESA included)
    mort = dict(cohorts)
    if io.exists(a.data, "cohort_mesa"):
        mort["MESA"] = sv.add_exposure(io.read(a.data, "cohort_mesa"), floor)
    rows = []
    for c, d in mort.items():
        ar = sv.at_risk(d, "death", L["max_followup_years"]).dropna(subset=[sv.EXPOSURE])
        for lm in L["landmarks_days"]:
            for adj, covs in sv.ADJUSTMENTS.items():
                f = sv.cox(sv.landmark(ar, lm), covs)
                rows.append({"cohort": c, "landmark_days": lm, "adjustment": adj,
                             **{k: f.get(k) for k in ("n", "events", "HR", "lo", "hi")}})
    io.write(pd.DataFrame(rows), a.out, "supp_table11_mortality")

    # Supplemental Figure 3: participant flow at the 90-day landmark, and model checks
    flow, checks = [], []
    for e, elab in ENDPOINTS.items():
        for c, d in cohorts.items():
            ar, land = risk[e, c], sv.landmark(risk[e, c], lm90)
            idx = pd.to_datetime(d["index_date"])
            prevalent = int((pd.to_datetime(d[f"{e}_date"], errors="coerce") <= idx).sum())
            alcohol = int(d.get("alcohol_excluded", pd.Series(False, index=d.index)).fillna(False).astype(bool).sum()) \
                if e == "masld" else 0
            a_s = land.dropna(subset=sv.AGE_SEX)
            a_sb = a_s.dropna(subset=["bmi"])
            fu = a_s["time_years"] + lm90 / 365.25
            flow.append({"outcome": elab, "cohort": c, "start": len(d), "prevalent_excluded": prevalent,
                         "alcohol_excluded": alcohol, "at_risk": len(ar),
                         "excluded_before_landmark": len(ar) - len(land), "at_landmark": len(land),
                         "analytic_age_sex": len(a_s), "analytic_bmi": len(a_sb),
                         "followup_median_years": fu.median(), "followup_p25": fu.quantile(0.25),
                         "followup_p75": fu.quantile(0.75)})
            for adj, covs in sv.ADJUSTMENTS.items():
                checks.append({"outcome": elab, "cohort": c, "adjustment": adj, **sv.model_checks(land, covs)})
    io.write(pd.DataFrame(flow), a.out, "supp_figure3_flow")
    io.write(pd.DataFrame(checks), a.out, "model_checks")


if __name__ == "__main__":
    main()
