"""Discrimination of the report-derived label in the temporally held-out YNHHS TTE test set (Results; Figure 1B;
Supplemental Methods).

Study scores are means of clip scores: all clips (the view-agnostic score used throughout); the standard on-axis clips
of each view; and combinations of on-axis PLAX, mid-papillary PSAX, and A4C (the mean of the view means, in studies
with every view of the combination). AUROC and AUPRC with 95% CIs from patient-level bootstrap replicates.

Second revision:
* Label sources (comments R1-2, R2-1): the test-set label applies the phrase rule to the same report entries retrieved
  from the echocardiography reporting system (the development label source) and from the institutional imaging data
  warehouse; discrimination is reported for each source's positives against the studies negative in both, and the
  agreement of the two sources is tabulated in the development and test studies with text in both.
* View subsets (R1-4): the number of studies (and positives) with each view, the clips classified as other (off-axis)
  views, and, for every view and combination, the all-view score recomputed in exactly the same studies with the
  paired bootstrap difference.
* Test-retest agreement (R2-2; for the Reviewers): first two test-set studies of patients with repeat studies, by
  interval; repeat studies were clinically indicated and often years apart, so this is not a repeatability study.

Inputs: tte_test_clips (patient_id, study_id, view, score), tte_test_studies (patient_id, study_id, label; optional
label_reporting_system, label_warehouse, text_reporting_system, text_warehouse, study_date), dev_label_sources
(optional). Outputs: figure1b_discrimination.csv, view_subsets.csv, view_counts.csv, label_sources.csv,
label_concordance.csv, r2r_test_retest.csv.
"""

import numpy as np
import pandas as pd

from panadipo import io, stats
from panadipo.scores import study_scores, view_scores

COMBINATIONS = {"PLAX + PSAX": ["plax_plax", "psax_pap"], "PLAX + A4C": ["plax_plax", "a4c"],
                "PSAX + A4C": ["psax_pap", "a4c"], "PLAX + PSAX + A4C": ["plax_plax", "psax_pap", "a4c"]}
SOURCES = ["label_reporting_system", "label_warehouse", "text_reporting_system", "text_warehouse"]


def scores(clips: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    out = {"All views": study_scores(clips)}
    out.update(view_scores(clips, cfg["views"]["standard"]).to_dict("series"))
    per_view = view_scores(clips, {v: [v] for v in cfg["views"]["key"]})
    for name, views in COMBINATIONS.items():
        out[name] = per_view[views].dropna().mean(axis=1)
    return out


def discrimination(d: pd.DataFrame, n_boot: int, seed: int, y: str = "label") -> dict:
    fn = lambda b: [stats.auroc(b[y], b["s"]), stats.auprc(b[y], b["s"])]  # noqa: E731
    lo, hi = stats.ci(stats.cluster_bootstrap(d, "patient_id", fn, n_boot, seed))
    return {"studies": len(d), "patients": d["patient_id"].nunique(), "positives": int(d[y].sum()),
            "auroc": stats.auroc(d[y], d["s"]), "auroc_lo": lo[0], "auroc_hi": hi[0],
            "auprc": stats.auprc(d[y], d["s"]), "auprc_lo": lo[1], "auprc_hi": hi[1]}


def view_subsets(studies: pd.DataFrame, sc: dict[str, pd.Series], cfg: dict) -> pd.DataFrame:
    """Each view-based score vs the all-view score in the same studies (paired patient-level bootstrap)."""
    allv = sc["All views"].rename("all")
    rows = []
    for name, s in sc.items():
        if name == "All views":
            continue
        d = studies.merge(s.rename("s"), left_on="study_id", right_index=True).dropna(subset=["s"])
        d = d.merge(allv, left_on="study_id", right_index=True).reset_index(drop=True)

        def f(b):
            a1, a2 = stats.auroc(b["label"], b["s"]), stats.auroc(b["label"], b["all"])
            return [a1, a2, a1 - a2]

        reps = stats.cluster_bootstrap(d, "patient_id", f, cfg["n_boot"], cfg["seed"])
        lo, hi = stats.ci(reps)
        est, diff = f(d), reps[:, 2][np.isfinite(reps[:, 2])]
        p = max(2 * min((diff <= 0).mean(), (diff >= 0).mean()), 1 / len(diff))
        rows.append({"score": name, "studies": len(d), "positives": int(d["label"].sum()),
                     "auroc_view": est[0], "auroc_view_lo": lo[0], "auroc_view_hi": hi[0],
                     "auroc_all_views_same_studies": est[1], "auroc_all_lo": lo[1], "auroc_all_hi": hi[1],
                     "difference": est[2], "difference_lo": lo[2], "difference_hi": hi[2], "p": p})
    return pd.DataFrame(rows)


def view_counts(clips: pd.DataFrame, studies: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Studies (and label-positive studies) with clips of each standard view, and clips of other (off-axis) views."""
    lab = studies.set_index("study_id")["label"]
    rows = []
    for v, classes in cfg["views"]["standard"].items():
        ids = clips.loc[clips["view"].isin(classes), "study_id"].unique()
        rows.append({"view": v, "studies": len(ids), "positives": int(lab.reindex(ids).sum()),
                     "clips": int(clips["view"].isin(classes).sum())})
    std = [c for cl in cfg["views"]["standard"].values() for c in cl]
    other = ~clips["view"].isin(std)
    rows.append({"view": "Other (off-axis)", "studies": clips.loc[other, "study_id"].nunique(),
                 "positives": int(lab.reindex(clips.loc[other, "study_id"].unique()).sum()),
                 "clips": int(other.sum())})
    return pd.DataFrame(rows).assign(total_clips=len(clips))


def label_sources(studies: pd.DataFrame, allv: pd.Series, cfg: dict) -> pd.DataFrame:
    d = studies.merge(allv.rename("s"), left_on="study_id", right_index=True)
    rs, wh = d["label_reporting_system"].fillna(0).astype(int), d["label_warehouse"].fillna(0).astype(int)
    neg = d[(rs == 0) & (wh == 0)]
    specs = [("Reporting-system positives vs studies negative in both sources", d[rs == 1], "label_reporting_system"),
             ("Warehouse-only positives vs studies negative in both sources", d[(wh == 1) & (rs == 0)],
              "label_warehouse"),
             ("Union (primary reference label)", None, "label")]
    rows = []
    for name, pos, y in specs:
        sub = d if pos is None else pd.concat([pos, neg])
        res = discrimination(sub.reset_index(drop=True), cfg["n_boot"], cfg["seed"], y)
        rows.append({"reference": name, **res, "positive_rate": res["positives"] / res["studies"]})
    for src in ("reporting_system", "warehouse"):                      # positive rate among studies with that text
        t = d[d[f"text_{src}"].fillna(False).astype(bool)]
        rows.append({"reference": f"Positive rate among studies with {src.replace('_', '-')} text", "studies": len(t),
                     "positives": int(t[f"label_{src}"].sum()), "positive_rate": t[f"label_{src}"].mean()})
    return pd.DataFrame(rows)


def concordance(name: str, d: pd.DataFrame) -> dict:
    both = d[d["text_reporting_system"].fillna(False).astype(bool) & d["text_warehouse"].fillna(False).astype(bool)]
    a, b = both["label_reporting_system"].astype(int), both["label_warehouse"].astype(int)
    n = lambda x, y: int(((a == x) & (b == y)).sum())  # noqa: E731
    return {"partition": name, "studies_with_both_sources": len(both), "positive_in_both": n(1, 1),
            "reporting_system_only": n(1, 0), "warehouse_only": n(0, 1), "negative_in_both": n(0, 0)}


def test_retest(studies: pd.DataFrame, allv: pd.Series, cfg: dict) -> pd.DataFrame:
    d = studies.merge(allv.rename("s"), left_on="study_id", right_index=True)
    d = d.assign(date=pd.to_datetime(d["study_date"])).dropna(subset=["date", "s"]).sort_values(["patient_id", "date"])
    d = d.groupby("patient_id").head(2)
    d = d[d.groupby("patient_id")["study_id"].transform("size") == 2]
    g = d.groupby("patient_id")
    a, b = g.nth(0).set_index("patient_id"), g.nth(1).set_index("patient_id")
    pair = pd.DataFrame({"x1": np.log(a["s"]), "x2": np.log(b["s"]), "days": (b["date"] - a["date"]).dt.days})
    pair = pair.dropna().reset_index()
    rows = []
    for lab, m in (("<=30 days", pair["days"] <= 30), ("<=90 days", pair["days"] <= 90),
                   ("<=180 days", pair["days"] <= 180), ("Any interval", pair["days"] >= 0)):
        q = pair[m].reset_index(drop=True)
        if len(q) < 10:
            continue
        fn = lambda z: [stats.spearman(z["x1"], z["x2"])[0], stats.icc_2_1(z["x1"], z["x2"])]  # noqa: E731
        lo, hi = stats.ci(stats.cluster_bootstrap(q, "patient_id", fn, cfg["n_boot"], cfg["seed"]))
        ba = stats.bland_altman(q["x2"], q["x1"])
        rows.append({"interval": lab, "patients": len(q), "median_days": q["days"].median(),
                     "rho": stats.spearman(q["x1"], q["x2"])[0], "rho_lo": lo[0], "rho_hi": hi[0],
                     "icc_log": stats.icc_2_1(q["x1"], q["x2"]), "icc_lo": lo[1], "icc_hi": hi[1],
                     "bias_log": ba["bias"], "loa_low": ba["loa_low"], "loa_high": ba["loa_high"]})
    return pd.DataFrame(rows)


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    clips, studies = io.read(a.data, "tte_test_clips"), io.read(a.data, "tte_test_studies")
    sc = scores(clips, cfg)
    rows = []
    for name, s in sc.items():
        d = studies.merge(s.rename("s"), left_on="study_id", right_index=True).dropna(subset=["s"])
        kind = "view" if name in cfg["views"]["standard"] else "combination" if name in COMBINATIONS else "all"
        res = discrimination(d.reset_index(drop=True), cfg["n_boot"], cfg["seed"])
        rows.append({"score": name, "kind": kind, **res})
    io.write(pd.DataFrame(rows), a.out, "figure1b_discrimination")
    io.write(view_subsets(studies, sc, cfg), a.out, "view_subsets")
    io.write(view_counts(clips, studies, cfg), a.out, "view_counts")
    if set(SOURCES) <= set(studies):
        io.write(label_sources(studies, sc["All views"], cfg), a.out, "label_sources")
        conc = [concordance("Test", studies)]
        if io.exists(a.data, "dev_label_sources"):
            conc.insert(0, concordance("Development (training and validation)", io.read(a.data, "dev_label_sources")))
        io.write(pd.DataFrame(conc), a.out, "label_concordance")
    if "study_date" in studies:
        io.write(test_retest(studies, sc["All views"], cfg), a.out, "r2r_test_retest")


if __name__ == "__main__":
    main()
