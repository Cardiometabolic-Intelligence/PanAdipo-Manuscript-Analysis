# PanAdipo

Code for **PanAdipo**, a video-based deep learning model that derives a score of prominent epicardial adiposity from
transthoracic echocardiography (TTE) and point-of-care ultrasound (POCUS), and for the statistical analyses of the
manuscript:

> Aminorroaya A, Coppi A, McNamara RL, Lima JAC, Krumholz HM, Antoniades C, Khera R, Oikonomou EK.
> *AI-Enabled Echocardiography Identifies Prominent Epicardial Adipose Tissue Associated With an Adverse
> Cardiometabolic Profile.*

The repository contains code only. Patient-level data cannot be redistributed: YNHHS imaging and EHR data are
institutional; MIMIC-IV is available through PhysioNet credentialed access; MESA is available through NHLBI BioLINCC and
BioData Catalyst. A synthetic data generator lets you run every analysis end to end.

## Layout

```
configs/
  train.yaml            fine-tuning settings (early-stopping metric and patience are configurable)
  analysis.yaml         seed, bootstrap replicates, view groups, CT and landmark settings
panadipo/               the package
  labels.py             report-derived label and pericardial effusion rules
  outcomes.py           ICD code lists (Supplemental Table 2) and first-diagnosis dates
  data.py, model.py     clip loading and augmentation; PanEcho backbone + linear head
  train.py              fine-tuning (distributed, mixed precision, configurable early stopping)
  predict.py, scores.py inference and study-level scores
  gradcam.py            Grad-CAM++ maps
  stats.py, meta.py     bootstrap, (partial) Spearman, AUROC/AUPRC, Holm, Wilson CIs, ICC; REML random-effects
                        pooling with the modified Hartung-Knapp-Sidik-Jonkman interval
  ct.py, mesa.py        CT sample, heart volume- and BSA-indexed EAT, segmentation QC; MESA biomarker models
  survival.py           linked records, landmark Cox, one-stage stratified Cox, Fine-Gray (R), multiple imputation,
                        model checks
analysis/               one script per manuscript section (01-09), each writing CSV result tables
examples/               synthetic data generator
docs/DATA.md            format of the input tables
tests/                  known-answer tests
```

## Installation

```bash
pip install -e .            # statistical analyses (Python >= 3.10)
pip install -e ".[dl]"      # + training, inference, and Grad-CAM++ (PyTorch, OpenCV, grad-cam)
```

The Fine-Gray sensitivity analysis calls R (`Rscript`) with the `survival` package; it is skipped if R is not
installed. The manuscript used Python 3.11.14 (pandas 2.3.3, NumPy 2.3.3, SciPy 1.17.1, statsmodels 0.14.6, lifelines
0.30.3, scikit-learn 1.8.0), R 4.6.1 (survival 3.8.9), PyTorch 2.7.1, and grad-cam 1.5.5.

## Quick start without patient data

```bash
python examples/make_synthetic_data.py --out example_data
./run_all.sh example_data results          # or run a single script: python analysis/05_ct_correlations.py --data example_data --out results
pytest
```

The synthetic tables follow [docs/DATA.md](docs/DATA.md); their results have no clinical meaning.

## Model

PanAdipo fine-tunes the public [PanEcho](https://github.com/CarDS-Yale/PanEcho) video encoder (loaded through
`torch.hub`) end to end, with a dropout layer and a single linear output. Clips are preprocessed with the PanEcho
pipeline and sampled to 16 frames at 224 × 224 pixels.

### Training

Clip tables are CSV files with `fpath` and `label` (one row per clip; patient-level partitions). Settings come from
[configs/train.yaml](configs/train.yaml); every setting can be overridden on the command line.

```bash
# 6 GPUs x 24 clips (effective batch 144), early stopping on validation AUPRC with patience 5 (configs/train.yaml)
torchrun --nproc_per_node 6 -m panadipo.train --config configs/train.yaml \
    --train_csv <train clips> --val_csv <validation clips> --out_dir <run folder>
```

Early stopping, checkpoint selection, and the learning-rate scheduler follow one validation metric. Both the metric
and the patience can be changed:

```bash
--monitor auprc --patience 5     # validation AUPRC; stop after 5 epochs without improvement
--monitor auroc --patience 7     # validation AUROC; stop after 7 epochs without improvement
--epochs 50                      # maximum number of epochs
```

Other settings: AdamW with learning rate 1e-4 for the head and 1e-5 for the backbone (`--head_lr`,
`--backbone_lr_divisor`), weight decay 1e-4, dropout 0.3, positive-class weight 15 in the binary cross-entropy loss
(`--pos_weight`), learning rate reduced 10-fold after 3 epochs without improvement (`--reduce_lr_patience`), mixed
precision (`--no-amp` to disable), and PanEcho-style augmentation (`--no-augment` to disable). The run folder receives
`best_model.pt`, `training_log.csv` (per-epoch loss, AUROC, and AUPRC), and `config.json`.

### Inference and Grad-CAM++

```bash
python -m panadipo.predict --checkpoint <best_model.pt> --clips_csv <clips with fpath> --out_csv <clip scores>
python -m panadipo.gradcam --checkpoint <best_model.pt> --out_dir <folder> --fpaths <clip> [<clip> ...]
```

The study score is the mean of all clip scores of a study (`panadipo.scores.study_scores`); in POCUS, only on-axis
PLAX, mid-papillary PSAX, and A4C clips are averaged. Because of the positive-class weight, the score is not a
calibrated probability and is analyzed as a continuous score. Model weights will be released with the publication.

## Where each result comes from

| Manuscript item | Script | Output table(s) |
|---|---|---|
| Table 1; Supplemental Figure 1; Figure 3C | `analysis/01_cohorts.py` | `table1`, `supp_figure1_distributions`, `figure3c_bmi_correlation` |
| Report-label AUROC/AUPRC overall, by view, and by view combination; Figure 1B; label sources, view counts, and view scores vs the all-view score in the same studies (Supplemental Methods); test-retest agreement (Response to Reviewers) | `analysis/02_report_label.py` | `figure1b_discrimination`, `view_subsets`, `view_counts`, `label_sources`, `label_concordance`, `r2r_test_retest` |
| Correlation with measured and PanEcho-derived echo features; Figure 1C | `analysis/03_echo_features.py` | `figure1c_feature_correlations` |
| Multivariable comparators; Supplemental Table 4 | `analysis/04_comparators.py` | `supp_table4_comparators` |
| CT depot correlations and partial correlations; indexing approaches and BMI benchmarks; subgroups; R²; Figures 2B, 3A, 3B; Supplemental Tables 5-7; Supplemental Figure 3 | `analysis/05_ct_correlations.py` | `figure2b_ct_correlations`, `supp_table5_ct_depots`, `supp_table6_ct_indexing`, `supp_table7_strata`, `figure3a_r2`, `figure3b_strata`, `ct_qc_flow`, `supp_figure3_interval` |
| Age- and sex-specific high EAT with confusion-matrix counts; Supplemental Table 8 | `analysis/06_ct_high_eat.py` | `supp_table8_high_eat`, `supp_table8_counts` |
| MESA biomarkers (M1-M4); Figure 4; Supplemental Table 9; Central Illustration (MESA panel) | `analysis/07_mesa.py` | `figure4_mesa_biomarkers` |
| Newly documented disease and mortality; Figure 5; Supplemental Tables 10-16; Supplemental Figure 4 | `analysis/08_incident_disease.py` | `supp_table10_pooled`, `supp_table10_cohorts`, `supp_table11_sensitivity`, `supp_table12_sensitivity`, `supp_table13_bmi_timing`, `supp_table14_hksj`, `supp_table15_model_checks`, `supp_table16_mortality`, `supp_figure4_flow` |
| Incremental C-statistic for steatotic liver disease and hyperlipidemia; Supplemental Table 11 (continued) | `analysis/09_incremental_c.py` | `supp_table11_delta_c` |
| Grad-CAM++ maps; Figures 1A, 2A; Supplemental Figure 2 | `python -m panadipo.gradcam` | PNG strips and map arrays |

## Statistical methods in brief

* **Confidence intervals**: 95% percentile intervals from 2,000 patient-level bootstrap replicates (1,000 for
  subgroups, thresholds, and R²); seed and replicate counts in [configs/analysis.yaml](configs/analysis.yaml).
* **Report label**: AUROC and AUPRC of the study score; view-specific scores average the standard on-axis clips of a
  view; combinations average the view scores in studies with every view of the combination, and each is compared with
  the all-view score in the same studies (paired bootstrap). The test-set label is the phrase rule applied to the same
  report entries retrieved from the reporting system and from the imaging data warehouse.
* **CT**: Spearman correlations with Holm-adjusted P values; partial Spearman correlations regress the ranks of both
  variables on the ranks of the covariates. EAT volume is reported unindexed and indexed to total heart volume
  (implausible cardiac segmentations excluded) and to BSA. High EAT is EAT volume / total heart volume at or above the
  95th percentile within age (<50, 50-64, ≥65 years) and sex groups; thresholds maximize the Youden index, with
  Wilson CIs for sensitivity, specificity, PPV, and NPV. Figure 3A and 3B use the patients with age, sex, and BMI.
* **MESA**: OLS of the z-scored biomarker on z-scored log(PanAdipo) with sequential adjustment (M1 age, sex; M2 +BMI;
  M3 +WHR; M4 = M2 + race and ethnicity, field center, and medications); log HOMA-IR is the primary biomarker.
* **Longitudinal**: records without EHR linkage excluded; patients with prevalent disease excluded (and, for steatotic
  liver disease, patients with alcohol-related codes); landmark at 30, 90 (primary), or 365 days; Cox models with
  robust SEs per SD of log(PanAdipo), adjusted for age and sex and additionally for BMI; cohort estimates pooled with
  REML random effects with I² (Q-profile CI) and the modified Hartung-Knapp-Sidik-Jonkman interval. Sensitivity
  analyses: Fine-Gray with death as a competing event; age- and sex-adjusted models in the BMI-complete sample;
  multiple imputation of missing BMI (20 imputations, Rubin's rules; predictors include baseline hypertension,
  hyperlipidemia, heart failure, and prior-year healthcare contact); a one-stage Cox model stratified by cohort with
  variance clustered on the participant; exclusion of POCUS patients also in YNHHS TTE; leave-one-cohort-out pooling;
  and BMI measured at or before the index study.

## License

The code in this repository is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3), as
for [PanEcho](https://github.com/CarDS-Yale/PanEcho). PanAdipo model weights, when released, will be licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA 4.0), matching the PanEcho weights on which they build.

## Contact

Evangelos K. Oikonomou, MD, DPhil, Yale School of Medicine (evangelos.oikonomou@yale.edu).
