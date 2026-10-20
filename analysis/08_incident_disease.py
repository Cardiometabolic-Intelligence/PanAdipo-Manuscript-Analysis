"""PanAdipo and newly documented cardiometabolic disease and mortality (Results; Figure 5; Supplemental Tables 10-16;
Supplemental Figure 4).

Landmark Cox models (panadipo.survival) in YNHHS TTE, YNHHS POCUS, and MIMIC-IV for type 2 diabetes, hypertension,
hyperlipidemia, and code-defined steatotic liver disease (MASLD/MASH codes; participants with alcohol-related codes
excluded in all three cohorts); HR per SD of log(PanAdipo) adjusted for age and sex, and additionally for BMI
(complete case). Records without EHR linkage (no age or sex) are excluded before log(PanAdipo) is standardized. Cohort
log HRs are pooled with REML random effects, with the modified Hartung-Knapp-Sidik-Jonkman (HKSJ) interval.
Sensitivity analyses at the 90-day landmark: Fine-Gray models with death as a competing event, age- and sex-adjusted
models in the BMI-complete sample, multiple imputation of missing BMI (Supplemental Table 11); a one-stage Cox model on
the stacked cohorts (cohort strata, robust variance clustered on the participant, as patients in both YNHHS cohorts
share one identifier), exclusion of POCUS patients who are also in YNHHS TTE, leave-one-cohort-out pooling, and BMI
measured at or before the index study (Supplemental Table 12); and the timing of the BMI measurement (Supplemental
Table 13). All-cause mortality is analyzed per cohort, including MESA.

Inputs: cohort_ynhhs_tte, cohort_ynhhs_pocus, cohort_mimic, cohort_mesa (docs/DATA.md).
Outputs: supp_table10_pooled.csv, supp_table10_cohorts.csv (Figure 5B), supp_table11_sensitivity.csv,
supp_table12_sensitivity.csv, supp_table13_bmi_timing.csv, supp_table14_hksj.csv, supp_table15_model_checks.csv,
supp_table16_mortality.csv, supp_figure4_flow.csv.
"""

import numpy as np
import pandas as pd

from panadipo import io, meta
from panadipo import survival as sv
from panadipo.outcomes import ENDPOINTS

HOSPITAL = {"cohort_ynhhs_tte": "YNHHS TTE", "cohort_ynhhs_pocus": "YNHHS POCUS", "cohort_mimic": "MIMIC-IV"}
SYSTEM = {"YNHHS TTE": "YNHHS", "YNHHS POCUS": "YNHHS", "MIMIC-IV": "MIMIC-IV"}


def pooled(fits: list[dict]) -> dict:
    p = meta.pool([f.get("logHR", np.nan) for f in fits], [f.get("se", np.nan) for f in fits])
    return {"n": sum(f["n"] for f in fits), "events": sum(f["events"] for f in fits), **p}


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    L = cfg["longitudinal"]
    lm90 = L["primary_landmark_days"]
    floor = cfg["score"]["log_floor_longitudinal"]
    raw = {lab: io.read(a.data, name) for name, lab in HOSPITAL.items()}
    cohorts = {c: sv.add_exposure(sv.linked(d), floor) for c, d in raw.items()}
    risk = {(e, c): sv.at_risk(d, e, L["max_followup_years"]).dropna(subset=[sv.EXPOSURE])
            for e in ENDPOINTS for c, d in cohorts.items()}

    # Supplemental Tables 10 and 14, Figure 5B: cohort-specific and pooled HRs (REML and HKSJ)
    pooled_rows, cohort_rows = [], []
    for e, elab in ENDPOINTS.items():
        for lm in L["landmarks_days"]:
            for adj, covs in sv.ADJUSTMENTS.items():
                fits = [sv.cox(sv.landmark(risk[e, c], lm), covs) for c in cohorts]
                p = pooled(fits)
                pooled_rows.append({"outcome": elab, "landmark_days": lm, "adjustment": adj,
                                    **{k: p.get(k) for k in ("n", "events", "HR", "lo", "hi", "p", "HKSJ_lo",
                                                             "HKSJ_hi", "HKSJ_p", "I2", "I2_lo", "I2_hi", "Q", "Q_p",
                                                             "tau2")}})
                for c, f, w in zip(cohorts, fits, p.get("weights", [np.nan] * len(fits))):
                    cohort_rows.append({"outcome": elab, "landmark_days": lm, "adjustment": adj, "cohort": c,
                                        "n": f["n"], "events": f["events"], "HR": f.get("HR"), "lo": f.get("lo"),
                                        "hi": f.get("hi"), "reml_weight": w})
    pool_t = pd.DataFrame(pooled_rows)
    io.write(pool_t.drop(columns=["HKSJ_lo", "HKSJ_hi", "HKSJ_p"]), a.out, "supp_table10_pooled")
    io.write(pd.DataFrame(cohort_rows).query("landmark_days == @lm90"), a.out, "supp_table10_cohorts")
    io.write(pool_t[["outcome", "landmark_days", "adjustment", "n", "events", "HR", "lo", "hi", "p", "HKSJ_lo",
                     "HKSJ_hi", "HKSJ_p", "I2"]], a.out, "supp_table14_hksj")

    # Supplemental Table 11: competing risk, BMI-complete, and multiple-imputation sensitivity analyses at 90 days
    rng = np.random.default_rng(cfg["seed"])
    land = {(e, c): sv.landmark(risk[e, c], lm90) for e in ENDPOINTS for c in cohorts}
    rows = []
    for e, elab in ENDPOINTS.items():
        for adj, covs in sv.ADJUSTMENTS.items():
            main_ = pooled([sv.cox(land[e, c], covs) for c in cohorts])
            fg = [sv.fine_gray(land[e, c], covs) for c in cohorts]
            row = {"outcome": elab, "adjustment": adj, "n": main_["n"], "events": main_["events"],
                   "HR": main_["HR"], "lo": main_["lo"], "hi": main_["hi"],
                   "I2": main_["I2"], "I2_lo": main_["I2_lo"], "I2_hi": main_["I2_hi"]}
            if all(fg):                      # skipped without Rscript
                p = meta.pool([f["logHR"] for f in fg], [f["se"] for f in fg])
                row.update({"competing_deaths": sum(f["competing_deaths"] for f in fg), "fine_gray_HR": p["HR"],
                            "fine_gray_lo": p["lo"], "fine_gray_hi": p["hi"]})
            if adj == "Age, sex":            # same model in the BMI-complete sample
                fits = [sv.cox(land[e, c].dropna(subset=["bmi"]), covs) for c in cohorts]
                label = "bmi_complete"
            else:                            # BMI multiply imputed (Rubin's rules within cohort)
                fits = []
                for c in cohorts:
                    dc = land[e, c].dropna(subset=["age", "female"]).reset_index(drop=True)
                    est = [sv.cox(dc.assign(bmi=b), covs)
                           for b in sv.impute_bmi(dc, L["mi_imputations"], rng, L["mi_extended"])]
                    q, se = sv.rubin([f["logHR"] for f in est], [f["se"] ** 2 for f in est])
                    fits.append({"logHR": q, "se": se})
                label = "bmi_imputed"
            p = meta.pool([f.get("logHR", np.nan) for f in fits], [f.get("se", np.nan) for f in fits])
            row.update({f"{label}_HR": p["HR"], f"{label}_lo": p["lo"], f"{label}_hi": p["hi"]})
            rows.append(row)
    io.write(pd.DataFrame(rows), a.out, "supp_table11_sensitivity")

    # Supplemental Table 12: one-stage model, POCUS-TTE overlap, leave-one-cohort-out, BMI at or before the index study
    in_tte = set(cohorts["YNHHS TTE"]["patient_id"])
    pre = None
    if all("bmi_pre_index" in d for d in cohorts.values()):
        pre = {(e, c): sv.landmark(sv.at_risk(d.assign(bmi=d["bmi_pre_index"]), e, L["max_followup_years"])
                                   .dropna(subset=[sv.EXPOSURE]), lm90) for e in ENDPOINTS for c, d in cohorts.items()}
    rows = []
    for e, elab in ENDPOINTS.items():
        for adj, covs in sv.ADJUSTMENTS.items():
            fits = {c: sv.cox(land[e, c], covs) for c in cohorts}
            prim = pooled(list(fits.values()))
            one = sv.one_stage_cox({c: land[e, c] for c in cohorts}, covs, SYSTEM)
            no_overlap = pooled([sv.cox(land[e, c][~land[e, c]["patient_id"].isin(in_tte)], covs)
                                 if c == "YNHHS POCUS" else fits[c] for c in cohorts])
            row = {"outcome": elab, "adjustment": adj, "primary_HR": prim["HR"], "primary_lo": prim["lo"],
                   "primary_hi": prim["hi"], "primary_n": prim["n"],
                   "one_stage_HR": one["HR"], "one_stage_lo": one["lo"], "one_stage_hi": one["hi"],
                   "one_stage_n": one["n"], "one_stage_participants": one["participants"],
                   "excl_pocus_in_tte_HR": no_overlap["HR"], "excl_pocus_in_tte_lo": no_overlap["lo"],
                   "excl_pocus_in_tte_hi": no_overlap["hi"], "excl_pocus_in_tte_n": no_overlap["n"]}
            for drop in cohorts:             # two cohorts: tau² is poorly estimated, near the fixed-effect estimate
                p = pooled([f for c, f in fits.items() if c != drop])
                row.update({f"excluding_{drop}_HR": p["HR"], f"excluding_{drop}_lo": p["lo"],
                            f"excluding_{drop}_hi": p["hi"]})
            if pre is not None and adj == "Age, sex, BMI":
                p = pooled([sv.cox(pre[e, c], covs) for c in cohorts])
                row.update({"bmi_pre_index_HR": p["HR"], "bmi_pre_index_lo": p["lo"], "bmi_pre_index_hi": p["hi"],
                            "bmi_pre_index_n": p["n"]})
            rows.append(row)
    io.write(pd.DataFrame(rows), a.out, "supp_table12_sensitivity")

    # Supplemental Table 13: BMI date minus index date, and BMI measured after the end of follow-up
    if all("bmi_days" in d for d in cohorts.values()):
        rows = []
        for c, d in cohorts.items():
            dd = pd.to_numeric(d["bmi_days"], errors="coerce")[d["bmi"].notna()].dropna()
            row = {"cohort": c, "with_bmi": len(dd), "median_days": dd.median(), "p25_days": dd.quantile(0.25),
                   "p75_days": dd.quantile(0.75), "on_index_day": (dd == 0).mean(), "before_index": (dd < 0).mean(),
                   "after_index": (dd > 0).mean(), f"after_landmark_{lm90}d": (dd > lm90).mean()}
            for e, elab in ENDPOINTS.items():  # BMI-adjusted 90-day landmark sample of each endpoint
                q = land[e, c].dropna(subset=sv.AGE_SEX_BMI + ["bmi_days"])
                after = q["bmi_days"] > (q["time_years"] * 365.25 + lm90)
                row[f"after_end_of_followup_{e}"] = f"{int(after.sum())}/{len(q)}"
            rows.append(row)
        io.write(pd.DataFrame(rows), a.out, "supp_table13_bmi_timing")

    # Supplemental Table 16: all-cause mortality per cohort (MESA included)
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
    io.write(pd.DataFrame(rows), a.out, "supp_table16_mortality")

    # Supplemental Figure 4: participant flow at the 90-day landmark; Supplemental Table 15: model checks
    flow, checks = [], []
    for e, elab in ENDPOINTS.items():
        for c, d in cohorts.items():
            ar, lnd = risk[e, c], land[e, c]
            idx = pd.to_datetime(d["index_date"])
            prevalent = int((pd.to_datetime(d[f"{e}_date"], errors="coerce") <= idx).sum())
            alc = d.get("alcohol_excluded", pd.Series(False, index=d.index)).fillna(False).astype(bool)
            alcohol = int((alc & ~(pd.to_datetime(d[f"{e}_date"], errors="coerce") <= idx)).sum()) if e == "masld" \
                else 0
            a_s = lnd.dropna(subset=sv.AGE_SEX)
            a_sb = a_s.dropna(subset=["bmi"])
            fu = a_s["time_years"] + lm90 / 365.25
            flow.append({"outcome": elab, "cohort": c, "scored": len(raw[c]), "unlinked_excluded": len(raw[c]) - len(d),
                         "start": len(d), "prevalent_excluded": prevalent, "alcohol_excluded": alcohol,
                         "at_risk": len(ar), "excluded_before_landmark": len(ar) - len(lnd), "at_landmark": len(lnd),
                         "analytic_age_sex": len(a_s), "analytic_bmi": len(a_sb),
                         "followup_median_years": fu.median(), "followup_p25": fu.quantile(0.25),
                         "followup_p75": fu.quantile(0.75)})
            for adj, covs in sv.ADJUSTMENTS.items():
                checks.append({"outcome": elab, "cohort": c, "adjustment": adj, **sv.model_checks(lnd, covs)})
    io.write(pd.DataFrame(flow), a.out, "supp_figure4_flow")
    io.write(pd.DataFrame(checks), a.out, "supp_table15_model_checks")


if __name__ == "__main__":
    main()
