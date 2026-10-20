"""Known-answer tests of the label rules, code lists, and statistical helpers (run with: pytest)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from panadipo import ct, labels, meta, outcomes, stats
from panadipo import survival as sv


def test_label_rule():
    findings = pd.Series(["Prominent epicardial fat pad.", "Normal study.", None,
                          "PROMINENT EPICARDIAL adipose tissue noted"])
    conclusions = pd.Series([None, "prominent epicardial fat depot", None, None])
    assert labels.prominent_epicardial_fat(findings, conclusions).tolist() == [1, 1, 0, 1]
    eff = labels.pericardial_effusion(pd.Series(["Trivial pericardial effusion.", "Small pericardial effusion.", "x"]))
    assert eff.tolist() == ["none_trace", "present", None]


def test_type2_diabetes_codes():
    dx = pd.DataFrame({"patient_id": ["a", "b", "c", "d"], "date": ["2020-01-01"] * 4,
                       "icd_code": ["250.00", "250.01", "250.42", "E11.65"], "icd_version": [9, 9, 9, 10]})
    first = outcomes.first_dates(dx).set_index("patient_id")
    assert first["t2d_date"].notna().to_dict() == {"a": True, "c": True, "d": True}      # type 1 code 250.01 excluded
    assert set(outcomes.alcohol_flag(pd.DataFrame({"patient_id": ["x", "y"], "icd_code": ["K70.30", "K76.0"],
                                                   "icd_version": [10, 10]})).index) == {"x"}


def test_partial_spearman_removes_shared_covariate():
    rng = np.random.default_rng(0)
    c = rng.normal(size=5000)
    x, y = c + 0.5 * rng.normal(size=5000), c + 0.5 * rng.normal(size=5000)
    assert stats.spearman(x, y)[0] > 0.7
    assert abs(stats.partial_spearman(x, y, c)[0]) < 0.05


def test_holm():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_reml_homogeneous_equals_fixed_effect():
    y, se = np.log([1.2, 1.2, 1.2]), np.array([0.1, 0.2, 0.15])
    p = meta.pool(y, se)
    assert p["tau2"] == 0 and p["I2"] == 0
    assert p["HR"] == pytest.approx(1.2)
    w = 1 / se ** 2
    assert p["se"] == pytest.approx(np.sqrt(1 / w.sum()))


def test_landmark_design():
    d = pd.DataFrame({"patient_id": list("abcde"), "index_date": pd.to_datetime(["2020-01-01"] * 5), "score": 0.1,
                      "age": 60, "female": 1, "bmi": 30,
                      "t2d_date": pd.to_datetime(["2019-06-01", "2020-02-01", "2021-01-01", None, None]),
                      "death_date": pd.to_datetime([None, None, None, "2020-03-01", None]),
                      "last_known_date": pd.to_datetime(["2024-01-01"] * 5)})
    ar = sv.at_risk(d, "t2d")
    assert "a" not in set(ar["patient_id"])                                 # prevalent at index
    assert ar.set_index("patient_id")["event"].to_dict() == {"b": 1, "c": 1, "d": 0, "e": 0}
    assert ar.set_index("patient_id").loc["d", "status"] == 2               # competing death
    lm = sv.landmark(ar, 90)
    assert set(lm["patient_id"]) == {"c", "e"}                              # outcome or death within 90 days
    assert lm.set_index("patient_id").loc["c", "time_years"] == pytest.approx((366 - 90) / 365.25)


def test_heart_index_and_high_eat():
    d = pd.DataFrame({"eat_volume": [100.0, 100.0], "paracardial_volume": [50.0, 50.0], "myocardium_volume": [120, 5],
                      "lv_volume": [150, 150], "rv_volume": [150, 150], "la_volume": [80, 80], "ra_volume": [100, 100]})
    out = ct.add_derived(d)
    assert out["eat_volume_heart_indexed"].iloc[0] == pytest.approx(100 / 600)
    assert np.isnan(out["eat_volume_heart_indexed"].iloc[1])               # structure <10 mL: implausible
    g = pd.DataFrame({"age": [40] * 20, "female": [1] * 20, "eat_volume_heart_indexed": np.arange(20.0)})
    assert ct.high_eat(g).sum() == 1


def test_hksj_never_narrower_than_reml():
    y, se = np.log([1.2, 1.25, 1.22]), np.array([0.1, 0.2, 0.15])         # homogeneous: q* < 1, truncated at 1
    p = meta.pool(y, se)
    t = st.t.ppf(0.975, 2)
    assert p["HKSJ_hi"] == pytest.approx(np.exp(p["logHR"] + t * p["se"]))
    assert p["HKSJ_lo"] < p["lo"] and p["HKSJ_hi"] > p["hi"]


def test_icc_and_wilson():
    a = np.arange(10.0)
    assert stats.icc_2_1(a, a) == pytest.approx(1.0)
    assert stats.icc_2_1(a, a + 5) < 0.6                                   # absolute agreement penalizes bias
    c = stats.confusion([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["sensitivity_lo"] < 0.5 < c["sensitivity_hi"]


def test_linked_records_and_bsa_index():
    d = pd.DataFrame({"age": [60, np.nan, 70], "female": [1, 0, np.nan]})
    assert len(sv.linked(d)) == 1
    out = ct.add_derived(pd.DataFrame({"eat_volume": [100.0], "paracardial_volume": [50.0], "myocardium_volume": [120],
                                       "lv_volume": [150], "rv_volume": [150], "la_volume": [80], "ra_volume": [100],
                                       "bsa": [2.0]}))
    assert out["eat_volume_bsa_indexed"].iloc[0] == pytest.approx(50.0)
    assert out["heart_volume"].iloc[0] == pytest.approx(600.0)
