"""PanAdipo and paired CCTA body and organ analysis in patients held out from model development (Results; Figures 2B,
3A, 3B; Supplemental Tables 5 and 6; Supplemental Figure 2).

Primary sample: one echocardiogram-CCTA pair per patient (closest CCTA within 365 days) among patients not used for
model training or validation. Spearman rho with 95% CIs from patient-level bootstrap replicates; Holm-adjusted P
across the panel of Supplemental Table 5; partial Spearman correlations adjust the ranks of both variables for the
ranks of the covariates.

Input: ct_pairs (docs/DATA.md). Outputs: figure2b_ct_correlations.csv, supp_table5_ct_depots.csv,
supp_table6_strata.csv (Figure 3B), figure3a_r2.csv, supp_figure2_interval.csv.
"""

import numpy as np
import pandas as pd

from panadipo import ct, io, stats

ST5 = ["eat_fraction_pericardium", "eat_volume_heart_indexed", "eat_volume", "paracardial_volume", "thoracic_volume",
       "sat_volume", "imat_volume", "eat_mean_hu", "eat_median_hu", "bmi"]
PARTIALS = {  # Supplemental Table 5
    "eat_fraction_pericardium": ["sat_volume", "bmi"], "eat_volume_heart_indexed": ["sat_volume", "bmi"],
    "eat_volume": ["sat_volume", "bmi"], "paracardial_volume": ["eat_volume", "sat_volume", "bmi"],
    "eat_mean_hu": ["eat_volume", "kvp"], "eat_median_hu": ["eat_volume", "kvp"], "bmi": ["age", "female"]}
LABELS = {**ct.VOLUMES, **ct.ATTENUATION}


def strata(d: pd.DataFrame, cfg: dict) -> dict[str, dict[str, pd.Series]]:
    """{stratum: {level: boolean mask}} for Supplemental Table 6."""
    c = cfg["ct"]
    b1, b2 = c["bmi_groups"]
    race = d["race_ethnicity"].where(d["race_ethnicity"].map(d["race_ethnicity"].value_counts()) >= c["min_group_n"],
                                     "Other or unknown")
    groups = {
        "Age (years)": ct.age_group(d["age"], c["age_groups"]),
        "Sex": d["female"].map({1: "Female", 0: "Male"}),
        "BMI (kg/m2)": pd.cut(d["bmi"], [-np.inf, b1, b2, np.inf], right=False,
                              labels=[f"<{b1}", f"{b1}-<{b2}", f">={b2}"]),
        "Race and ethnicity": race,
        "Echo vendor": d["echo_vendor"],
        "CT scanner model": d["ct_scanner"],
        "Tube voltage (kVp)": pd.cut(d["kvp"], [0, 95, 115, 125, 200], labels=["70-90", "100-110", "120", "130-140"]),
    }
    out = {k: {str(lv): v == lv for lv in (v.cat.categories if hasattr(v, "cat") else sorted(v.dropna().unique()))}
           for k, v in groups.items()}
    days = d["interval_days"].abs()
    out["Echocardiogram-CCTA interval"] = {f"<={n} days": days <= n for n in c["close_interval_days"]}
    eff = d["effusion"]
    out["Pericardial effusion"] = {"Excluding echo studies with reported effusion": eff != "present",
                                   "Restricted to echo studies reporting no or trace effusion": eff == "none_trace",
                                   "Echo studies with reported effusion": eff == "present"}
    return out


def main() -> None:
    a = io.cli(__doc__)
    cfg = io.config()
    nb, nb2, seed = cfg["n_boot"], cfg["n_boot_secondary"], cfg["seed"]
    d = ct.add_derived(ct.primary_sample(io.read(a.data, "ct_pairs"), cfg["ct"]["max_interval_days"]),
                       cfg["ct"]["heart_volume_ml"], cfg["ct"]["min_structure_ml"])
    print(f"primary held-out sample: {len(d):,} patients")

    # Supplemental Table 5: panel, Holm P, and partial correlations
    rows = []
    for i, f in enumerate(ST5):
        r = stats.spearman_ci(d, "score", f, nb, seed + i)
        row = {"measure": LABELS[f], "patients": r["patients"], **{k: r[k] for k in ("rho", "lo", "hi", "p")}}
        if f in PARTIALS:
            pr = stats.spearman_ci(d, "score", f, nb, seed + 10 * i, covariates=PARTIALS[f])
            row.update({"partial_rho": pr["rho"], "partial_lo": pr["lo"], "partial_hi": pr["hi"],
                        "adjusted_for": ", ".join(PARTIALS[f])})
        rows.append(row)
    t5 = pd.DataFrame(rows)
    t5.insert(6, "holm_p", stats.holm(t5["p"]))
    io.write(t5, a.out, "supp_table5_ct_depots")

    # Figure 2B: unadjusted and age-, sex-, and BMI-adjusted (BMI itself: age and sex) rho for every measure
    rows = []
    for i, f in enumerate(c for c in LABELS if c in d):
        r = stats.spearman_ci(d, "score", f, nb, seed + i)
        adj = ["age", "female"] + ([] if f == "bmi" else ["bmi"])
        pr = stats.spearman_ci(d, "score", f, nb2, seed + i, covariates=adj)
        rows.append({"measure": LABELS[f], "kind": "attenuation" if f in ct.ATTENUATION else "volume or fraction",
                     "patients": r["patients"], "rho": r["rho"], "lo": r["lo"], "hi": r["hi"],
                     "adjusted_rho": pr["rho"], "adjusted_lo": pr["lo"], "adjusted_hi": pr["hi"]})
    f2b = pd.DataFrame(rows).assign(abs_rho=lambda x: x["rho"].abs())         # rows ordered by |rho| within panel
    io.write(f2b.sort_values(["kind", "abs_rho"], ascending=[False, False]).drop(columns="abs_rho"), a.out,
             "figure2b_ct_correlations")

    # Supplemental Table 6 / Figure 3B: rho with indexed EAT across strata
    y = "eat_volume_heart_indexed"
    r = stats.spearman_ci(d, "score", y, nb, seed)
    rows = [{"stratum": "All", "level": "Primary held-out sample", "patients": r["patients"], "rho": r["rho"],
             "lo": r["lo"], "hi": r["hi"]}]
    for name, levels in strata(d, cfg).items():
        for lv, mask in levels.items():
            if mask.sum() >= 20:
                r = stats.spearman_ci(d[mask], "score", y, nb2, seed)
                rows.append({"stratum": name, "level": lv, "patients": r["patients"], "rho": r["rho"], "lo": r["lo"],
                             "hi": r["hi"]})
    io.write(pd.DataFrame(rows), a.out, "supp_table6_strata")

    # Figure 3A: variance in indexed EAT explained (OLS R²) by log10(PanAdipo), by age + sex + BMI, and by both
    m = d.dropna(subset=[y, "score", "age", "female", "bmi"]).reset_index(drop=True)
    yv, base = m[y].to_numpy(float), m[["age", "female", "bmi"]].to_numpy(float)
    X = {"PanAdipo": np.log10(m[["score"]].to_numpy(float)), "Age, sex, BMI": base,
         "Age, sex, BMI + PanAdipo": np.column_stack([base, np.log10(m["score"])])}

    def r2(Xm, yy):
        Xm = np.column_stack([np.ones(len(yy)), Xm])
        return 1 - ((yy - Xm @ np.linalg.lstsq(Xm, yy, rcond=None)[0]) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()

    rng = np.random.default_rng(seed)
    reps = np.array([[r2(Xm[i], yv[i]) for Xm in X.values()]
                     for i in (rng.integers(0, len(yv), len(yv)) for _ in range(nb2))])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    io.write(pd.DataFrame({"model": list(X), "patients": len(yv), "r2": [r2(Xm, yv) for Xm in X.values()],
                           "lo": lo, "hi": hi}), a.out, "figure3a_r2")

    # Supplemental Figure 2: echocardiogram date minus CCTA date (days), all pairs within 365 days
    pairs = io.read(a.data, "ct_pairs")
    bins = np.arange(-365, 370, 15)
    counts, _ = np.histogram(pairs.loc[pairs["interval_days"].abs() <= 365, "interval_days"], bins)
    hist = pd.DataFrame({"days_from": bins[:-1], "days_to": bins[1:], "pairs": counts})
    io.write(hist, a.out, "supp_figure2_interval")

if __name__ == "__main__":
    main()
