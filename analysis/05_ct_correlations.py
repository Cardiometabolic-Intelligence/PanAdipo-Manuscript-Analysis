"""PanAdipo and paired CCTA body and organ analysis in patients held out from model development (Results; Figures 2B,
3A, 3B; Supplemental Tables 5-7; Supplemental Figure 3).

Primary sample: one echocardiogram-CCTA pair per patient (closest CCTA within 365 days) among patients not used for
model training or validation. EAT volume is indexed to total heart volume (implausible cardiac segmentations are
excluded rather than corrected; flow in ct_qc_flow) and to BSA. Spearman rho with 95% CIs from patient-level bootstrap
replicates; Holm-adjusted P across the panel of Supplemental Table 5; partial Spearman correlations adjust the ranks of
both variables for the ranks of the covariates.

Second revision (comment R1-3): Supplemental Table 6 compares indexing approaches (absolute, heart volume-indexed, and
BSA-indexed EAT volume; absolute EAT volume with total heart volume as a covariate; linear regression of log EAT volume
with log heart volume), benchmarks the CT EAT measures against BMI in the same patients, and gives reciprocal partial
correlations with paracardial fat. Figure 3A (R²) and 3B (subgroups) use one sample, the patients with indexed EAT
volume, age, sex, and BMI; Supplemental Table 7 gives the subgroups in all patients with indexed EAT volume.

Input: ct_pairs (docs/DATA.md). Outputs: figure2b_ct_correlations.csv, supp_table5_ct_depots.csv,
supp_table6_ct_indexing.csv, supp_table7_strata.csv, figure3a_r2.csv, figure3b_strata.csv, ct_qc_flow.csv,
supp_figure3_interval.csv.
"""

import numpy as np
import pandas as pd

from panadipo import ct, io, stats

ST5 = ["eat_fraction_pericardium", "eat_volume_heart_indexed", "eat_volume", "eat_volume_bsa_indexed",
       "paracardial_volume", "thoracic_volume", "sat_volume", "imat_volume", "eat_mean_hu", "eat_median_hu", "bmi"]
PARTIALS = {  # Supplemental Table 5
    "eat_fraction_pericardium": ["sat_volume", "bmi"], "eat_volume_heart_indexed": ["sat_volume", "bmi"],
    "eat_volume": ["sat_volume", "bmi"], "eat_volume_bsa_indexed": ["age", "female", "bmi"],
    "paracardial_volume": ["eat_volume", "sat_volume", "bmi"],
    "eat_mean_hu": ["eat_volume", "kvp"], "eat_median_hu": ["eat_volume", "kvp"], "bmi": ["age", "female"]}
LABELS = {**ct.VOLUMES, **ct.ATTENUATION}


def strata(d: pd.DataFrame, cfg: dict) -> dict[str, dict[str, pd.Series]]:
    """{stratum: {level: boolean mask}} for Supplemental Table 7 and Figure 3B."""
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
    for i, f in enumerate(c for c in ST5 if c in d):
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

    # Supplemental Table 6: indexing approaches, heart volume as a covariate, BMI benchmarks, reciprocal partials
    io.write(indexing(d, nb, seed), a.out, "supp_table6_ct_indexing")
    io.write(ct.qc_flow(d, cfg["ct"]["heart_volume_ml"], cfg["ct"]["min_structure_ml"]), a.out, "ct_qc_flow")

    # Supplemental Table 7 (all patients with indexed EAT) and Figure 3B (patients with BMI): rho across strata
    y = "eat_volume_heart_indexed"
    d["high_eat"] = ct.high_eat(d, q=cfg["ct"]["high_eat_percentile"], cuts=cfg["ct"]["age_groups"])
    m = d.dropna(subset=[y, "score", "age", "female", "bmi"]).reset_index(drop=True)
    for name, sample in (("supp_table7_strata", d), ("figure3b_strata", m)):
        io.write(strata_rows(sample, cfg, y, nb, nb2, seed), a.out, name)

    # Figure 3A: variance in indexed EAT explained (OLS R²) by log10(PanAdipo), by age + sex + BMI, and by both, and
    # the least-squares fit of indexed EAT on log10(PanAdipo), all in the same patients
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
    slope, intercept = np.polyfit(np.log10(m["score"]), yv, 1)
    io.write(pd.DataFrame({"model": list(X), "patients": len(yv), "r2": [r2(Xm, yv) for Xm in X.values()],
                           "lo": lo, "hi": hi, "fit_slope": slope, "fit_intercept": intercept,
                           "rho": stats.spearman(m["score"], m[y])[0], "high_eat": int(m["high_eat"].sum())}), a.out,
             "figure3a_r2")

    # Supplemental Figure 3: echocardiogram date minus CCTA date (days), all pairs within 365 days
    pairs = io.read(a.data, "ct_pairs")
    bins = np.arange(-365, 370, 15)
    counts, _ = np.histogram(pairs.loc[pairs["interval_days"].abs() <= 365, "interval_days"], bins)
    hist = pd.DataFrame({"days_from": bins[:-1], "days_to": bins[1:], "pairs": counts})
    io.write(hist, a.out, "supp_figure3_interval")


def strata_rows(d: pd.DataFrame, cfg: dict, y: str, nb: int, nb2: int, seed: int) -> pd.DataFrame:
    r = stats.spearman_ci(d, "score", y, nb, seed)
    rows = [{"stratum": "All", "level": "Primary held-out sample", "patients": r["patients"], "rho": r["rho"],
             "lo": r["lo"], "hi": r["hi"]}]
    for name, levels in strata(d, cfg).items():
        for lv, mask in levels.items():
            if mask.sum() >= 20:
                r = stats.spearman_ci(d[mask], "score", y, nb2, seed)
                rows.append({"stratum": name, "level": lv, "patients": r["patients"], "rho": r["rho"], "lo": r["lo"],
                             "hi": r["hi"]})
    return pd.DataFrame(rows)


def indexing(d: pd.DataFrame, nb: int, seed: int) -> pd.DataFrame:
    """Supplemental Table 6 (comment R1-3)."""
    import statsmodels.api as sm
    eat = {"EAT volume": "eat_volume", "EAT volume indexed to heart volume": "eat_volume_heart_indexed",
           "EAT volume indexed to BSA": "eat_volume_bsa_indexed"}
    rows = []

    def add(label, x, y, covs=(), adjusted="-", also=()):
        sub = d.dropna(subset=[x, y, *covs, *also])
        r = stats.spearman_ci(sub, x, y, nb, seed + len(rows), covariates=list(covs) or None)
        rows.append({"correlation": label, "adjusted_for": adjusted, "patients": r["patients"], "estimate": r["rho"],
                     "lo": r["lo"], "hi": r["hi"], "p": r["p"]})

    for lab, y in eat.items():
        add(f"PanAdipo vs {lab}", "score", y)
    add("PanAdipo vs EAT volume", "score", "eat_volume", ["heart_volume"], "Total heart volume")
    add("PanAdipo vs EAT volume", "score", "eat_volume", ["heart_volume", "age", "female", "bmi"],
        "Total heart volume, age, sex, BMI")
    add("PanAdipo vs EAT volume", "score", "eat_volume", ["paracardial_volume"], "Paracardial volume")
    add("PanAdipo vs EAT volume indexed to heart volume", "score", "eat_volume_heart_indexed", ["paracardial_volume"],
        "Paracardial volume")
    add("PanAdipo vs paracardial volume", "score", "paracardial_volume", ["eat_volume"], "EAT volume")
    add("PanAdipo vs total heart volume", "score", "heart_volume")
    if "bsa" in d:
        add("PanAdipo vs BSA", "score", "bsa")
    for lab, y in eat.items():                        # BMI benchmark and PanAdipo in the same patients
        add(f"BMI vs {lab}", "bmi", y)
        add(f"PanAdipo vs {lab}, same patients as BMI benchmark", "score", y, also=["bmi"])
    add("PanAdipo vs BMI", "score", "bmi")
    # percentage difference in EAT volume per SD of log(PanAdipo), with log heart volume (robust HC3 SEs)
    m = d.dropna(subset=["eat_volume", "heart_volume", "score", "age", "female", "bmi", "eat_volume_heart_indexed"])
    m = m[m["eat_volume"] > 0]
    z = stats.zscore(np.log(m["score"])).to_numpy()
    for lab, X in (("log heart volume", np.column_stack([z, np.log(m["heart_volume"])])),
                   ("log heart volume, age, sex, BMI",
                    np.column_stack([z, np.log(m["heart_volume"]), m[["age", "female", "bmi"]].to_numpy(float)]))):
        f = sm.OLS(np.log(m["eat_volume"].to_numpy(float)), sm.add_constant(X)).fit(cov_type="HC3")
        b, (lo, hi) = f.params[1], f.conf_int()[1]
        rows.append({"correlation": "EAT volume per SD of log(PanAdipo), linear regression of log EAT volume "
                                    "(% difference)", "adjusted_for": lab, "patients": len(m),
                     "estimate": 100 * (np.exp(b) - 1), "lo": 100 * (np.exp(lo) - 1), "hi": 100 * (np.exp(hi) - 1),
                     "p": f.pvalues[1]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
