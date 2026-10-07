# Input tables

The analysis scripts read analysis-ready tables (`.csv` or `.parquet`) from one folder (`--data`). No patient-level
data are distributed with this repository. `examples/make_synthetic_data.py` writes synthetic tables in exactly this
format.

Conventions: one row per unit described; dates in ISO format (`YYYY-MM-DD`); `female` is 1/0; `score` is the PanAdipo
output (0-1) aggregated as described; empty cells are missing values.

## YNHHS TTE test set

**`tte_test_clips`**: one row per clip of the temporally held-out test set.

| Column | Description |
|---|---|
| `patient_id`, `study_id` | patient and study identifiers |
| `view` | PanEcho view-classifier class (e.g., `plax_plax`, `psax_pap`, `a4c`) |
| `score` | clip-level PanAdipo output (`python -m panadipo.predict`) |

**`tte_test_studies`**: one row per test study.

| Column | Description |
|---|---|
| `patient_id`, `study_id` | identifiers |
| `label` | report-derived label (1/0; `panadipo.labels.prominent_epicardial_fat` applied to every available report text) |
| `age`, `female`, `bmi` | at the study (BMI: closest measurement within ±365 days) |
| `measured_<feature>` | report measurement (Figure 1C), e.g., `measured_ivsd` |
| `panecho_<feature>` | PanEcho output averaged over the study's clips; features with both a `measured_` and a `panecho_` column are compared in Figure 1C, and all `panecho_` columns enter the comparators (Supplemental Table 4) |

**`dev_case_control`**: development case-control sample for the comparators: all label-positive training and
validation studies and randomly selected label-negative studies; same columns as `tte_test_studies` (`patient_id`,
`study_id`, `label`, `age`, `female`, `bmi`, `panecho_*`).

## Paired echocardiography-CCTA cohort

**`ct_pairs`**: one row per echocardiogram, paired with its closest CCTA (the primary sample is selected in code:
pairs within 365 days, patients outside model development, one pair per patient).

| Column | Description |
|---|---|
| `patient_id`, `echo_date` | identifiers and echocardiogram date |
| `interval_days` | echocardiogram date minus CCTA date (days) |
| `development` | True if the patient contributed to model training or validation |
| `score` | study score (mean of all clip outputs) |
| `age`, `female`, `bmi`, `race_ethnicity` | at the CCTA (used in the CT analyses) |
| `age_at_echo`, `bmi_at_echo` | optional: age at the echocardiogram and BMI within ±365 days of it (used in Table 1, as in the other YNHHS cohorts) |
| `<condition>_date` | optional first diagnosis dates for the Table 1 conditions (prevalent if on or before `echo_date`) |
| `echo_vendor`, `ct_scanner`, `kvp` | from DICOM headers (manufacturer, scanner model, tube voltage) |
| `effusion` | pericardial effusion in the echocardiography report: `none_trace`, `present`, or empty (`panadipo.labels.pericardial_effusion`) |
| BOA measures | see below |

BOA measures (Body and Organ Analysis output; adipose tissue −190 to −30 HU):

| Column | BOA feature |
|---|---|
| `eat_volume` | `tissues / epicardial_adipose_tissue / volume_ml` |
| `paracardial_volume` | `tissues / paracardial_adipose_tissue / volume_ml` |
| `sat_volume` | `tissues / subcutaneous_adipose_tissue / volume_ml` |
| `imat_volume` | `tissues / intermuscular_adipose_tissue / volume_ml` |
| `eat_mean_hu`, `eat_median_hu` | `tissues / epicardial_adipose_tissue / mean_hu`, `median_hu` |
| `paracardial_mean_hu`, `sat_mean_hu` | `tissues / paracardial_adipose_tissue / mean_hu`, `subcutaneous_adipose_tissue / mean_hu` |
| `eat_fraction_pericardium` | `body_regions / pericardium / frac_fat_hu` |
| `paracardial_fraction_mediastinum` | `body_regions / mediastinum / frac_fat_hu` |
| `pericardium_volume` | `body_regions / pericardium / volume_ml` |
| `myocardium_volume`, `lv_volume`, `rv_volume`, `la_volume`, `ra_volume` | `total / heart_myocardium`, `heart_ventricle_left`, `heart_ventricle_right`, `heart_atrium_left`, `heart_atrium_right` / `volume_ml` |

Thoracic adipose tissue volume (EAT + paracardial) and EAT volume indexed to total heart volume are derived in
`panadipo.ct.add_derived`.

## MESA Exam 6

**`mesa`**: one row per participant with a PanAdipo score; `score` is the mean output over the participant's decoded
clips. Other columns use MESA Exam 6 variable names: `age6c`, `gender1`, `bmi6c`, `waistcm6`, `hipcm6`, `insulin6`,
`glucose6`, `trig6`, `hdl6`, `ldl6`, `chol6`, `hba1c6`, `sbp6c`, `dbp6c`, `cepgfr6c`, `dm036c`, `race1c`, `lipid6c`,
`htnmed6c`, `insln6c`, `ohga6c`, and `field_center`.

## Longitudinal cohorts

**`cohort_ynhhs_tte`, `cohort_ynhhs_pocus`, `cohort_mimic`, `cohort_mesa`**: one row per patient; the index study is
the patient's earliest echocardiogram (MESA: the Exam 6 echocardiogram; used for Table 1 and mortality).

| Column | Description |
|---|---|
| `patient_id`, `index_date` | identifiers and index study date |
| `score` | study score: mean of all clip outputs (POCUS: mean of on-axis PLAX, mid-papillary PSAX, and A4C clips) |
| `age`, `female`, `race_ethnicity` | at the index study |
| `bmi` | closest measurement within ±365 days of the index study (empty if none) |
| `death_date` | date of death (empty if alive) |
| `last_known_date` | last recorded contact (empty if not used) |
| `t2d_date`, `htn_date`, `hld_date`, `masld_date` | first diagnosis date of each endpoint, any time (`panadipo.outcomes.first_dates`); dates on or before `index_date` mark prevalent disease |
| `<condition>_date` | optional first dates of the Table 1 comorbidities (`panadipo.outcomes.COMORBIDITIES`) |
| `alcohol_excluded` | MIMIC-IV only: any alcohol-related code (`panadipo.outcomes.alcohol_flag`); excluded from MASLD/MASH analyses |

For MESA, prevalent conditions are those recorded at Exam 6 (set the date to `index_date`).
