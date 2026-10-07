"""Incremental discrimination of PanAdipo for newly documented MASLD/MASH at the 90-day landmark (Supplemental Table
10, continued; added in revision, not prespecified).

Within each cohort's BMI-complete landmark population: apparent Harrell's C of a Cox model with age, sex, and BMI,
without and with standardized log(PanAdipo). 95% CIs for the change from patient-level bootstrap replicates in which
both models are refitted. Calibration and absolute risks were not assessed.

Inputs: cohort_ynhhs_tte, cohort_ynhhs_pocus, cohort_mimic. Output: supp_table10_delta_c.csv.
"""

import numpy as np
import pandas as pd

from panadipo import io
from panadipo import survival as sv

HOSPITAL = {"cohort_ynhhs_tte": "YNHHS TTE", "cohort_ynhhs_pocus": "YNHHS POCUS", "cohort_mimic": "MIMIC-IV"}
BASE = sv.AGE_SEX_BMI


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    L = cfg["longitudinal"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for name, lab in HOSPITAL.items():
        d = sv.add_exposure(io.read(a.data, name), cfg["score"]["log_floor_longitudinal"])
        land = sv.landmark(sv.at_risk(d, "masld", L["max_followup_years"]), L["primary_landmark_days"])
        m = land[["time_years", "event", sv.EXPOSURE] + BASE].dropna().reset_index(drop=True)
        c0, c1 = sv.harrell_c(m, BASE), sv.harrell_c(m, BASE + [sv.EXPOSURE])
        reps = []
        for _ in range(L["delta_c_boot"]):
            b = m.iloc[rng.integers(0, len(m), len(m))].reset_index(drop=True)
            if b["event"].sum() >= 5:
                reps.append(sv.harrell_c(b, BASE + [sv.EXPOSURE]) - sv.harrell_c(b, BASE))
        lo, hi = np.percentile(reps, [2.5, 97.5])
        rows.append({"cohort": lab, "patients": len(m), "events": int(m["event"].sum()), "c_age_sex_bmi": c0,
                     "c_plus_panadipo": c1, "delta_c": c1 - c0, "lo": lo, "hi": hi, "replicates": len(reps)})
    io.write(pd.DataFrame(rows), a.out, "supp_table10_delta_c")


if __name__ == "__main__":
    main()
