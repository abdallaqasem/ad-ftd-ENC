ad-ftd-ENC — elastic-net classification of Alzheimer's disease vs frontotemporal dementia
A reproducible machine-learning pipeline for the AD versus FTD differential diagnosis, developed toward analysis of the National Alzheimer's Coordinating Center (NACC) Uniform Data Set.

Distinguishing Alzheimer's disease from frontotemporal dementia is one of the harder routine problems in cognitive neurology: the two overlap in presentation, and the task is attribution rather than detection. This repository holds the modelling pipeline for a regularised (elastic-net) logistic classifier built for that question, together with a nested cross-validation estimate, a fitted probability calibrator, and a transportability test against a simulated external cohort.

Data governance
No patient-level data is in this repository and none ever will be. NACC records are held under an institutional Data Use Agreement and are analysed only in an approved offline environment.

So that the work stays inspectable anyway, the repository ships a synthetic NACC-schema generator. It produces mock records with the same variable names, scales and coding conventions as the Uniform Data Set, and it reproduces two features of real NACC data that matter for modelling:

Form-version missingness — UDS v2 records carry MMSE but not MoCA, v3 records the reverse, with a minority having both. This is the instrument-harmonisation problem the real analysis has to solve.
Modality-specific missingness — 1-5% for clinical variables, 18% for MRI volumes, 25% for fluid biomarkers.
The entire pipeline can be cloned and run end to end by anyone, with no data access and no agreement in place.

The two synthetic cohorts
data/generate_data.py defines two site profiles, and every difference between them is declared in SITE_PROFILES rather than buried in code. This is what makes the transportability test below interpretable.

Development site	External site
FTD prevalence	22%	35% (specialist referral centre)
Mean age	baseline	3 years younger
Plasma p-tau217	baseline	x1.25 (different immunoassay platform)
Plasma NfL	baseline	x0.85 (different immunoassay platform)
MRI volumes	baseline	x1.02 (different scanner / segmentation)
UDS form version	35% v2, 50% v3, 15% both	v3 only — MMSE entirely absent
Fluid-biomarker missingness	25%	45%
Note the last two rows: the external cohort contains no MMSE at all, so a model trained where MMSE was available must cope with that column being wholly imputed. That is the realistic version of the harmonisation problem.

What the pipeline does
Generate (data/generate_data.py) — two synthetic cohorts, 12 columns each: an identifier, the AD/FTD label, and 10 predictors (age, education, BMI, MMSE, MoCA, CDR-SB, plasma p-tau217, NfL, hippocampal and frontal volumes).

Load and clean (src/data_loader.py) — read the cohort, drop the identifier, drop records with no diagnosis.

Split and preprocess (src/preprocess.py) — stratified 80/20 split, then StandardScaler followed by KNNImputer (k=5). Scaling precedes imputation deliberately: KNN imputation uses Euclidean distance, so features must be on a comparable scale first.

Nested cross-validation (src/evaluate.py) — a 5-fold outer loop around the hyperparameter search, estimating how the whole procedure generalises rather than how one arbitrary train/test split happened to score.

Fit and calibrate (src/train_model.py) — elastic-net logistic regression via LogisticRegressionCV (10-fold inner CV, 20 regularisation strengths, 7 L1/L2 mixing ratios, saga, log-loss, balanced class weights), then a Platt/sigmoid calibrator fitted on top with 5-fold CV. Balanced class weights deliberately distort predicted probabilities in order to help discrimination on a 78/22 split, which is exactly why the calibrator is needed.

Evaluate on the held-out internal test set — ROC-AUC, PR-AUC, Brier score, classification report, confusion matrix, and per-feature coefficients as log-odds and odds ratios.

Transportability test against the simulated external cohort, reported both uncalibrated and calibrated.

Metrics are written to output/metrics.json and coefficients to output/coefficients.csv.

Example output
Synthetic data throughout — these numbers are not diagnostic results. Runtime about one minute on a laptop.


=== Nested cross-validation (training set only) ===
  outer fold 1/5: ROC-AUC 0.9609  PR-AUC 0.9188  Brier 0.0805
  outer fold 2/5: ROC-AUC 0.9626  PR-AUC 0.8699  Brier 0.0768
  outer fold 3/5: ROC-AUC 0.9733  PR-AUC 0.9290  Brier 0.0561
  outer fold 4/5: ROC-AUC 0.9816  PR-AUC 0.9630  Brier 0.0508
  outer fold 5/5: ROC-AUC 0.9720  PR-AUC 0.9266  Brier 0.0607
  nested CV: ROC-AUC 0.9701 (SD 0.0085)  Brier 0.0650 (SD 0.0130)

selected C = 0.2336   selected l1_ratio = 0.1000
Evaluation	ROC-AUC	PR-AUC	Brier
Internal held-out test, uncalibrated	0.9669	0.9269	0.0624
Internal held-out test, calibrated	0.9690	0.9318	0.0454
Simulated external cohort, uncalibrated	0.9612	0.9391	0.0725
Simulated external cohort, calibrated	0.9624	0.9407	0.0770
Coefficients from the final model (positive = toward FTD):


 Feature  Log_Odds_Coef  Odds_Ratio
     age        -1.2754      0.2793
 ptau217        -1.2664      0.2818
frontalV        -1.0082      0.3649
   hipoV         0.8735      2.3953
     nfl         0.7410      2.0981
    moca         0.5964      1.8155
    mmse         0.4167      1.5169
   cdrsb        -0.3542      0.7017
    educ        -0.0589      0.9428
     bmi         0.0139      1.0140
What this run shows
The signs are clinically coherent. Younger age, lower p-tau217, smaller frontal volume, larger hippocampal volume and higher NfL all push toward FTD. The model recovered the structure built into the generator.
Calibration helps internally and does not transport. The calibrator cut the internal Brier score by 27% (0.0624 to 0.0454), but on the external cohort it made calibration slightly worse (0.0725 to 0.0770), because it was fitted at 22% prevalence and the external site runs at 35%.
Discrimination transports; calibration does not. ROC-AUC barely moved under a shift in prevalence, assay scaling, scanner and form version — but the probability scale did. This is the single most important practical lesson in the cross-cohort literature, and it is why a model moved between centres needs recalibration rather than retraining.
Education and BMI were shrunk nearly to zero, which is correct: they are noise in the generator.
Scope and limitations
Everything here runs on synthetic data. The numbers demonstrate that the pipeline executes and that the evaluation structure is sound. They say nothing about real diagnostic performance.
The "external" cohort is simulated, not real. It is a transportability test under a declared shift, which is a much stronger test than re-drawing from the same distribution, but it is not a substitute for validation in an independent real-world cohort. That remains future work.
Variable harmonisation across UDS form versions is performed separately, outside this repository, as part of the NACC analysis. The generator simulates the problem; the pipeline does not solve it.
Preprocessing is fitted on the whole training set before the inner hyperparameter folds run, so scaling and imputation leak slightly across those inner folds. The outer splits are clean, so the reported estimates are not optimistically biased by tuning.
The nested loop uses a reduced search grid (10 regularisation strengths, 3 mixing ratios, 5 inner folds) to keep runtime near a minute; the final model is fitted with the full grid.
Quickstart
Requires Python 3.10+ and Git.


git clone https://github.com/bobalqasem-crypto/ad-ftd-ENC.git
cd ad-ftd-ENC
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/generate_data.py     # writes data/mock9.parquet and data/mock_external.parquet
python src/train_model.py        # nested CV, fit, calibrate, internal and external evaluation
<details> <summary>Step-by-step setup if you are new to Git or the terminal</summary>
Install Python (3.10 or newer), Git and VS Code.

Open VS Code, then Terminal -> New Terminal.

Run each command from the Quickstart block above, one line at a time, waiting for each to finish.

On Windows, if activating the virtual environment is blocked, first run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass.

</details>
Repository layout
Path	Contents
data/generate_data.py	Synthetic NACC-schema generator and the two declared site profiles
src/data_loader.py	Parquet loading, feature/target split
src/preprocess.py	Stratified split; scaling and KNN imputation
src/evaluate.py	Nested cross-validation and metric reporting
src/train_model.py	Training, calibration, internal and external evaluation
output/	Generated metrics and coefficients (git-ignored)
Related work
A companion systematic review with meta-analysis compares the discriminative performance of clinically available markers for the same differential, organised by level of care. Its results supply the feature prior for this classifier.

Licence and contact
MIT. Abdalla Qasem — Gazi University Faculty of Medicine, Ankara · ORCID 0009-0005-4664-7883
