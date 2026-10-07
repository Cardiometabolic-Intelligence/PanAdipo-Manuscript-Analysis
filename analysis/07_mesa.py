"""Cross-sectional associations of PanAdipo with cardiometabolic biomarkers in MESA Exam 6 (Figure 4; Supplemental
Table 8).

Standardized beta per SD of log(PanAdipo) from OLS models M1-M4 (panadipo.mesa) in the full cohort, the normoglycemic
subgroup (dm036c == 0: fasting glucose <100 mg/dL without treated diabetes), and participants with BMI <25 kg/m².
Log HOMA-IR is the primary biomarker; the others are secondary or exploratory and not adjusted for multiplicity.

Input: mesa (one row per participant; `score` = mean PanAdipo over the participant's decoded clips, plus Exam 6
variables). Output: figure4_mesa_biomarkers.csv.
"""

import pandas as pd

from panadipo import io, mesa


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    d = mesa.prepare(io.read(a.data, "mesa"), cfg["score"]["log_floor_mesa"])
    groups = {"Full cohort": d, "Normoglycemic": d[d["dm036c"] == 0],
              f"BMI <{cfg['mesa']['lean_bmi']} kg/m2": d[d["bmi6c"] < cfg["mesa"]["lean_bmi"]]}
    res = pd.concat([mesa.run_models(g).assign(cohort=name) for name, g in groups.items()], ignore_index=True)
    io.write(res[["cohort", "biomarker", "model", "n", "beta", "lo", "hi", "p"]], a.out, "figure4_mesa_biomarkers")


if __name__ == "__main__":
    main()
