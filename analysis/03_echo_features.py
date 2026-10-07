"""Overlap of PanAdipo with conventional echocardiographic features in the YNHHS TTE test set (Figure 1C).

Study-level Spearman correlation of the PanAdipo study score with each feature available both as a report
measurement (measured_<feature>) and as a PanEcho output (panecho_<feature>; study mean of clip outputs), with 95% CIs
from study-level bootstrap replicates.

Inputs: tte_test_clips, tte_test_studies (study_id, measured_*, panecho_*).
Output: figure1c_feature_correlations.csv.
"""

import numpy as np
import pandas as pd

from panadipo import io, stats
from panadipo.scores import study_scores


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    studies = io.read(a.data, "tte_test_studies")
    d = studies.merge(study_scores(io.read(a.data, "tte_test_clips")).rename("score"), left_on="study_id",
                      right_index=True)
    features = [c.removeprefix("measured_") for c in d if c.startswith("measured_") and f"panecho_{c[9:]}" in d]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for f in features:
        for source in ("measured", "panecho"):
            x = d[["score", f"{source}_{f}"]].apply(pd.to_numeric, errors="coerce").dropna().values
            r = stats.spearman(x[:, 0], x[:, 1])[0]
            reps = [stats.spearman(*x[rng.integers(0, len(x), len(x))].T)[0] for _ in range(cfg["n_boot_secondary"])]
            lo, hi = np.nanpercentile(reps, [2.5, 97.5])
            label = {"measured": "Report measurement", "panecho": "PanEcho output"}[source]
            rows.append({"feature": f, "source": label,
                         "studies": len(x), "rho": r, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    io.write(out.sort_values(["source", "abs_rho"], ascending=[False, False]), a.out, "figure1c_feature_correlations")


if __name__ == "__main__":
    main()
