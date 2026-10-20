"""Detection of age- and sex-specific high EAT on CCTA (Abstract; Results; Supplemental Table 8).

High EAT: EAT volume indexed to total heart volume at or above the 95th percentile within age (<50, 50-64, >=65
years) and sex groups of the primary held-out sample. AUROC overall and within each age-sex group; within each group,
the PanAdipo threshold maximizing the Youden index with its sensitivity, specificity, PPV, and NPV. 95% CIs for the
AUROC and 95% intervals for the threshold from bootstrap replicates (because thresholds take observed score values, an
interval can be bounded by the point estimate). Supplemental Table 8 (continued; comment R1-7): confusion-matrix
counts at each threshold with Wilson 95% CIs. Thresholds were selected in this sample (apparent, optimistic
estimates) and are hypothesis-generating.

Input: ct_pairs. Outputs: supp_table8_high_eat.csv, supp_table8_counts.csv.
"""

import numpy as np
import pandas as pd

from panadipo import ct, io, stats


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    c = cfg["ct"]
    d = ct.add_derived(ct.primary_sample(io.read(a.data, "ct_pairs"), c["max_interval_days"]),
                       c["heart_volume_ml"], c["min_structure_ml"])
    d = d.dropna(subset=["score", "age", "female"])
    d["y"] = ct.high_eat(d, q=c["high_eat_percentile"], cuts=c["age_groups"])
    d = d.dropna(subset=["y"])
    sex = d["female"].map({1: "Female", 0: "Male"})
    d["group"] = sex + ", age " + ct.age_group(d["age"], c["age_groups"]).astype(str)
    rng = np.random.default_rng(cfg["seed"])
    rows, counts = [], []
    for name, g in [("All held-out patients", d)] + list(d.groupby("group")):
        y, s = g["y"].to_numpy(int), g["score"].to_numpy(float)
        boots = [(stats.auroc(y[i], s[i]), stats.youden_threshold(y[i], s[i]))
                 for i in (rng.integers(0, len(y), len(y)) for _ in range(cfg["n_boot_secondary"]))
                 if 0 < y[i].sum() < len(i)]
        auc_b, thr_b = np.array(boots).T
        r = {"group": name, "patients": len(g), "high_eat": int(y.sum()), "prevalence": y.mean(),
             "auroc": stats.auroc(y, s), "auroc_lo": np.percentile(auc_b, 2.5), "auroc_hi": np.percentile(auc_b, 97.5)}
        if name != "All held-out patients":
            thr = stats.youden_threshold(y, s)
            r.update({"threshold": thr, "threshold_lo": np.percentile(thr_b, 2.5),
                      "threshold_hi": np.percentile(thr_b, 97.5), **stats.operating_point(y, s, thr)})
            counts.append({"group": name, "patients": len(g), "threshold": thr, **stats.confusion(y, s, thr)})
        rows.append(r)
    io.write(pd.DataFrame(rows), a.out, "supp_table8_high_eat")
    io.write(pd.DataFrame(counts), a.out, "supp_table8_counts")


if __name__ == "__main__":
    main()
