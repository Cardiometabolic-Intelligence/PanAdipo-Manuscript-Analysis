"""Discrimination of the report-derived label in the temporally held-out YNHHS TTE test set (Results; Figure 1B).

Study scores are means of clip scores: all clips (the view-agnostic score used throughout); the standard on-axis clips
of each view; and combinations of on-axis PLAX, mid-papillary PSAX, and A4C (the mean of the view means, in studies
with every view of the combination). AUROC and AUPRC with 95% CIs from patient-level bootstrap replicates.

Inputs: tte_test_clips (patient_id, study_id, view, score), tte_test_studies (patient_id, study_id, label).
Output: figure1b_discrimination.csv.
"""

import pandas as pd

from panadipo import io, stats
from panadipo.scores import study_scores, view_scores

COMBINATIONS = {"PLAX + PSAX": ["plax_plax", "psax_pap"], "PLAX + A4C": ["plax_plax", "a4c"],
                "PSAX + A4C": ["psax_pap", "a4c"], "PLAX + PSAX + A4C": ["plax_plax", "psax_pap", "a4c"]}


def scores(clips: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    out = {"All views": study_scores(clips)}
    out.update(view_scores(clips, cfg["views"]["standard"]).to_dict("series"))
    per_view = view_scores(clips, {v: [v] for v in cfg["views"]["key"]})
    for name, views in COMBINATIONS.items():
        out[name] = per_view[views].dropna().mean(axis=1)
    return out


def discrimination(d: pd.DataFrame, n_boot: int, seed: int) -> dict:
    fn = lambda b: [stats.auroc(b["label"], b["s"]), stats.auprc(b["label"], b["s"])]  # noqa: E731
    lo, hi = stats.ci(stats.cluster_bootstrap(d, "patient_id", fn, n_boot, seed))
    return {"studies": len(d), "patients": d["patient_id"].nunique(), "positives": int(d["label"].sum()),
            "auroc": stats.auroc(d["label"], d["s"]), "auroc_lo": lo[0], "auroc_hi": hi[0],
            "auprc": stats.auprc(d["label"], d["s"]), "auprc_lo": lo[1], "auprc_hi": hi[1]}


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    clips, studies = io.read(a.data, "tte_test_clips"), io.read(a.data, "tte_test_studies")
    rows = []
    for name, s in scores(clips, cfg).items():
        d = studies.merge(s.rename("s"), left_on="study_id", right_index=True).dropna(subset=["s"])
        kind = "view" if name in cfg["views"]["standard"] else "combination" if name in COMBINATIONS else "all"
        res = discrimination(d.reset_index(drop=True), cfg["n_boot"], cfg["seed"])
        rows.append({"score": name, "kind": kind, **res})
    io.write(pd.DataFrame(rows), a.out, "figure1b_discrimination")


if __name__ == "__main__":
    main()
