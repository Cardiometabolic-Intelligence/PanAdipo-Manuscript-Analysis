"""Write synthetic input tables in the documented format (docs/DATA.md) so every analysis script can be run end to
end without access to patient data. Values are simulated from a latent adiposity trait; the numbers they produce have
no clinical meaning.

    python examples/make_synthetic_data.py --out example_data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = ["plax_plax", "psax_pap", "psax_avz", "psax_mv", "a2c", "a3c", "a4c", "a5c", "subcostal", "a4c_laocc"]
FEATURES = ["ivsd", "lvpwd", "lvidd", "lvids", "lvef", "la_volume", "tapse", "rvsp"]
RACE = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Asian", "Other/Unknown"]
RACE_P = [0.65, 0.15, 0.1, 0.03, 0.07]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def people(n: int, rng, age=(65, 15)):
    z = rng.standard_normal(n)                                     # latent epicardial adiposity
    return pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in range(n)], "z": z,
        "age": np.clip(rng.normal(*age, n), 18, 95).round(1), "female": rng.integers(0, 2, n),
        "bmi": np.clip(29 + 3 * (0.45 * z + 0.9 * rng.standard_normal(n)) * 2.2, 15, 70).round(1),
        "race_ethnicity": rng.choice(RACE, n, p=RACE_P),
        "score": sigmoid(-2.6 + 1.1 * z + 0.4 * rng.standard_normal(n)),
    })


def tte_test(rng, n=1500):
    p = people(n, rng)
    studies = p.loc[p.index.repeat(rng.integers(1, 3, n))].reset_index(drop=True)
    studies["study_id"] = [f"S{i:06d}" for i in range(len(studies))]
    studies["label"] = (rng.random(len(studies)) < sigmoid(-6 + 1.6 * studies["z"])).astype(int)
    for f in FEATURES:
        true = 0.25 * studies["z"] + rng.standard_normal(len(studies))
        measured = true + 0.4 * rng.standard_normal(len(studies))
        studies[f"measured_{f}"] = measured.where(rng.random(len(studies)) > 0.2)
        studies[f"panecho_{f}"] = true + 0.3 * rng.standard_normal(len(studies))
    for k in range(12):                                            # PanEcho outputs without a report measurement
        studies[f"panecho_task{k:02d}"] = 0.1 * studies["z"] + rng.standard_normal(len(studies))
    studies.loc[rng.random(len(studies)) < 0.3, "bmi"] = np.nan
    clips = studies.loc[studies.index.repeat(rng.integers(8, 30, len(studies))), ["patient_id", "study_id", "z"]]
    clips["view"] = rng.choice(VIEWS, len(clips))
    clips["score"] = sigmoid(-2.6 + 1.1 * clips["z"] + 0.8 * rng.standard_normal(len(clips)))
    dev = people(3000, rng)
    dev["study_id"] = [f"D{i:06d}" for i in range(len(dev))]
    dev["label"] = (rng.random(len(dev)) < sigmoid(-2.5 + 1.6 * dev["z"])).astype(int)    # case-control enriched
    for c in [c for c in studies if c.startswith("panecho_")]:
        dev[c] = (0.25 if c[8:] in FEATURES else 0.1) * dev["z"] + rng.standard_normal(len(dev))
    drop = ["z"]
    return clips.drop(columns=drop), studies.drop(columns=drop + ["score"]), dev.drop(columns=drop + ["score"])


def ct_pairs(rng, n=2500):
    p = people(n, rng, age=(59, 14))
    z = p["z"]
    heart = {c: np.clip(rng.normal(m, m / 5, n), 5, None) for c, m in
             (("myocardium_volume", 120), ("lv_volume", 140), ("rv_volume", 150), ("la_volume", 90), ("ra_volume", 80))}
    total = sum(heart.values())
    eat = np.exp(np.log(0.12 * total) + 0.35 * z + 0.15 * rng.standard_normal(n))
    d = p.assign(**heart, eat_volume=eat, pericardium_volume=total + eat + rng.normal(60, 10, n),
                 paracardial_volume=np.exp(np.log(0.8 * eat) + 0.2 * rng.standard_normal(n)),
                 sat_volume=np.exp(8 + 0.25 * z + 0.4 * rng.standard_normal(n)),
                 imat_volume=np.exp(5 + 0.2 * z + 0.4 * rng.standard_normal(n)),
                 eat_mean_hu=-80 - 4 * z + rng.normal(0, 5, n), paracardial_mean_hu=-85 + rng.normal(0, 6, n),
                 sat_mean_hu=-100 + rng.normal(0, 6, n),
                 kvp=rng.choice([80, 100, 120, 140], n, p=[0.3, 0.25, 0.37, 0.08]),
                 echo_vendor=rng.choice(["Philips", "Siemens", "GE"], n, p=[0.78, 0.19, 0.03]),
                 ct_scanner=rng.choice(["SOMATOM Force", "SOMATOM Definition Edge", "LightSpeed VCT"], n,
                                       p=[0.93, 0.05, 0.02]),
                 effusion=rng.choice(["none_trace", "present", None], n, p=[0.87, 0.02, 0.11]),
                 interval_days=rng.integers(-365, 366, n), development=rng.random(n) < 0.06,
                 echo_date=pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 2500, n), unit="D"))
    d["eat_median_hu"] = d["eat_mean_hu"] - 3
    d["age_at_echo"] = d["age"] + d["interval_days"] / 365.25
    d["bmi_at_echo"] = d["bmi"].where(rng.random(n) > 0.05)
    d["eat_fraction_pericardium"] = d["eat_volume"] / d["pericardium_volume"]
    d["paracardial_fraction_mediastinum"] = np.clip(0.3 + 0.03 * z + rng.normal(0, 0.05, n), 0, 1)
    extra = d.sample(frac=0.1, random_state=1).assign(interval_days=lambda x: rng.integers(-365, 366, len(x)))
    return pd.concat([d, extra]).drop(columns="z")                 # some patients with several pairs


def mesa(rng, n=1500):
    p = people(n, rng, age=(74, 8.5))
    z, e = p["z"], lambda s=1: s * rng.standard_normal(n)
    d = pd.DataFrame({"mesaid": np.arange(n) + 10000, "score": p["score"] * 0.6, "age6c": p["age"],
                      "gender1": 1 - p["female"], "bmi6c": p["bmi"].clip(15, 55), "race1c": rng.choice([1, 2, 3, 4], n),
                      "field_center": rng.choice([3, 4, 5, 6, 7, 8], n),
                      "waistcm6": 95 + 6 * z + e(10), "hipcm6": 105 + 3 * z + e(8),
                      "insulin6": np.exp(2 + 0.25 * z + e(0.5)), "glucose6": 95 + 5 * z + e(15),
                      "trig6": np.exp(4.7 + 0.15 * z + e(0.4)), "hdl6": 55 - 4 * z + e(14), "ldl6": 105 + e(30),
                      "chol6": 185 + e(35), "hba1c6": 5.8 + 0.1 * z + e(0.6), "sbp6c": 125 + 2 * z + e(18),
                      "dbp6c": 68 + 1.5 * z + e(10), "cepgfr6c": 75 + e(15),
                      "dm036c": rng.choice([0, 1, 2, 3], n, p=[0.5, 0.25, 0.05, 0.2]),
                      "lipid6c": rng.integers(0, 2, n), "htnmed6c": rng.integers(0, 2, n),
                      "insln6c": (rng.random(n) < 0.05).astype(int), "ohga6c": (rng.random(n) < 0.15).astype(int)})
    d.loc[rng.random(n) < 0.6, "insulin6"] = np.nan                # insulin measured in a subsample
    d.loc[d["field_center"].isin([3, 4]), "hba1c6"] = np.nan       # HbA1c not measured at two centers
    return d


def cohort(rng, n: int, start: str, death_rate: float, mimic: bool = False):
    p = people(n, rng)
    idx = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 365 * 3, n), unit="D")
    d = p.assign(index_date=idx)
    years = lambda rate: np.minimum(rng.exponential(1 / rate, n), 9)   # noqa: E731  (>8 years = never)
    for e, base, beta, prev in (("t2d", 0.03, 0.25, 0.25), ("htn", 0.12, 0.15, 0.6), ("hld", 0.1, 0.15, 0.45),
                                ("masld", 0.01, 0.35, 0.05)):
        t = years(base * np.exp(beta * p["z"]))
        is_prev = rng.random(n) < prev
        t[is_prev] = -rng.uniform(0.01, 10, is_prev.sum())
        d[f"{e}_date"] = (idx + pd.to_timedelta(t * 365.25, unit="D")).where(t < 8)
    for c, prev in (("myocardial_infarction", 0.1), ("heart_failure", 0.25), ("chronic_coronary_syndrome", 0.25),
                    ("stroke", 0.1), ("atrial_fibrillation", 0.15), ("peripheral_vascular_disease", 0.04)):
        d[f"{c}_date"] = idx.where(rng.random(n) < prev)
    death = years(death_rate * np.exp(-0.1 * p["z"] + 0.04 * (p["age"] - 65)))
    d["death_date"] = (idx + pd.to_timedelta(death * 365.25, unit="D")).where(death < 8)
    d["last_known_date"] = idx + pd.to_timedelta(rng.uniform(0.2, 6, n) * 365.25, unit="D")
    d.loc[rng.random(n) < 0.3, "bmi"] = np.nan
    if mimic:
        d["alcohol_excluded"] = rng.random(n) < 0.05
    return d.drop(columns="z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="example_data")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips, studies, dev = tte_test(rng)
    tables = {"tte_test_clips": clips, "tte_test_studies": studies, "dev_case_control": dev, "ct_pairs": ct_pairs(rng),
              "mesa": mesa(rng),
              "cohort_ynhhs_tte": cohort(rng, 2000, "2022-07-01", 0.04),
              "cohort_ynhhs_pocus": cohort(rng, 3000, "2020-01-01", 0.12),
              "cohort_mimic": cohort(rng, 1500, "2017-01-01", 0.05, mimic=True),
              "cohort_mesa": cohort(rng, 1500, "2017-01-01", 0.03)}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
