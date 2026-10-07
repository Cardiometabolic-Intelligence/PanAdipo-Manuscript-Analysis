"""Multivariable comparators fitted in the development set and evaluated in the YNHHS TTE test set (Supplemental
Table 4).

Comparators are fitted on a development case-control sample (all label-positive training and validation studies and
randomly selected label-negative studies; PanEcho outputs averaged over all clips of each study):
  (1) logistic regression on age, sex, and BMI;
  (2) L2-penalized logistic regression on the PanEcho outputs (penalty tuned by patient-grouped 5-fold
      cross-validation on AUROC);
  (3) PanEcho outputs plus age, sex, and BMI (L2-penalized).
Predictors are standardized on the development sample; case-control sampling affects only the intercept. PanAdipo is
not refitted. AUROC, AUPRC, and PanAdipo-minus-comparator differences with 95% CIs from paired patient-level
bootstrap replicates; models with BMI use the BMI-complete test studies.

Inputs: dev_case_control (patient_id, study_id, label, age, female, bmi, panecho_*), tte_test_studies (same columns),
tte_test_clips. Output: supp_table4_comparators.csv.
"""

import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from panadipo import io, stats
from panadipo.scores import study_scores

DEMO = ["age", "female", "bmi"]


def fit(dev: pd.DataFrame, cols: list[str], penalized: bool):
    X, y = dev[cols].to_numpy(float), dev["label"].astype(int).to_numpy()
    if penalized:
        folds = list(GroupKFold(5).split(X, y, dev["patient_id"]))
        clf = LogisticRegressionCV(Cs=10, cv=folds, scoring="roc_auc", max_iter=5000)      # L2 (ridge) penalty
    else:
        clf = LogisticRegression(C=1e6, max_iter=5000)                                     # effectively unpenalized
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_pipeline(StandardScaler(), clf).fit(X, y)


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    dev = io.read(a.data, "dev_case_control")
    test = io.read(a.data, "tte_test_studies")
    test = test.merge(study_scores(io.read(a.data, "tte_test_clips")).rename("PanAdipo"), left_on="study_id",
                      right_index=True)
    pe = sorted(set(c for c in dev if c.startswith("panecho_")) & set(c for c in test if c.startswith("panecho_")))
    specs = {"Age, sex, and BMI (logistic)": (DEMO, False),
             f"PanEcho outputs (n={len(pe)}) (L2 logistic)": (pe, True),
             "PanEcho outputs + age, sex, and BMI (L2 logistic)": (pe + DEMO, True)}
    rows = []
    for sample, need in (("All test studies", pe), ("BMI-complete test studies", pe + DEMO)):
        t = test.dropna(subset=need + ["PanAdipo"]).reset_index(drop=True)
        models = ["PanAdipo"]
        for name, (cols, pen) in specs.items():
            if set(cols) <= set(need):
                t[name] = fit(dev.dropna(subset=cols), cols, pen).predict_proba(t[cols].to_numpy(float))[:, 1]
                models.append(name)

        def fn(b):
            return [f(b["label"], b[m]) for m in models for f in (stats.auroc, stats.auprc)]

        reps = stats.cluster_bootstrap(t, "patient_id", fn, cfg["n_boot"], cfg["seed"])
        point = fn(t)
        for i, m in enumerate(models):
            lo, hi = stats.ci(reps[:, 2 * i:2 * i + 2])
            r = {"test sample": sample, "model": m, "studies": len(t), "positives": int(t["label"].sum()),
                 "auroc": point[2 * i], "auroc_lo": lo[0], "auroc_hi": hi[0],
                 "auprc": point[2 * i + 1], "auprc_lo": lo[1], "auprc_hi": hi[1]}
            if i:
                dlo, dhi = stats.ci(reps[:, :2] - reps[:, 2 * i:2 * i + 2])
                r.update({"delta_auroc": point[0] - point[2 * i], "delta_auroc_lo": dlo[0], "delta_auroc_hi": dhi[0],
                          "delta_auprc": point[1] - point[2 * i + 1], "delta_auprc_lo": dlo[1],
                          "delta_auprc_hi": dhi[1]})
            rows.append(r)
    io.write(pd.DataFrame(rows), a.out, "supp_table4_comparators")


if __name__ == "__main__":
    main()
