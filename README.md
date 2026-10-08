# Tamweel Lite — Cost-Aware Credit-Risk Review Policy

An end-to-end machine-learning project for predicting financing defaults within 90 days of application, testing the model across different time periods and customers, choosing a decision threshold within a 12% review limit, checking model accuracy and interpretability, and creating a repeatable final policy for new data.


**Course:** SDA-DSC-211 — Advanced Machine Learning Methods (SDAIA Academy) · **Project type:** Individual capstone project


## Project overview

Tamweel Lite addresses synthetic financing default prediction with a cost-aware machine-learning approach. The complete workflow covers data preparation, baseline model comparison, data leakage detection, time- and customer-aware validation, class-imbalance handling, cost-sensitive threshold selection, model calibration, model interpretation, ensemble evaluation, regional analysis, and final batch policy generation.

The project compares three main models:

`Logistic Regression`, `XGBoost`, and `LightGBM`.

The final policy uses Logistic Regression with calibrated probabilities and a decision threshold designed to keep the review rate within a 12% capacity limit.

## Final decision at a glance

Below is the executive summary of the production model selection, calibrated operating threshold, and capacity-constrained review policy.

| Item | Result |
|---|---|
| **Final Model** | **Logistic Regression** (Selected single model over ensemble for stability & transparency) |
| **Mean OOF Average Precision** | **0.392** (Fold SD: ±0.030) |
| **Decision Rule** | `probability >= 0.1223` *(Calibrated scale; raw OOF threshold 0.1689)* |
| **OOF Recall / Precision** | **46.9% / 34.3%** *(Caught 84 out of 179 actual defaults)* |
| **Busiest-Period Flag Rate** | **11.7%** *(Fully compliant with the 12% review capacity constraint)* |
| **Challenge Batch Processing** | 2,500 applications → 330 above threshold → **300 flagged** *(Capacity cap applied)* |
| **Challenge Performance** | *Not claimed* — Test labels are intentionally held out |

---
<img width="1662" height="647" alt="111" src="https://github.com/user-attachments/assets/983b1cb9-242b-4da5-8137-c7e9f69506d4" />
 
---
 
## Problem Formulation

| Constraint / Variable | Operational Specification |
|---|---|
| **Business Objective** | Prioritize high-risk credit applications for manual audit under a 12% capacity limit |
| **Target Variable** | `default_within_90d` (Default event within 90 days of application) |
| **Asymmetric Cost Matrix** | **10 × FN** (Missed Default) vs. **1 × FP** (False Alarm Audit) |
| **Review Capacity Cap** | Max **12%** of total volume per period |
| **Historical Development Set** | 10,000 applications (2022–2024) · Default Rate: 7.89% |
| **Out-of-Time Challenge Set** | 2,500 distinct applications (2025) · Unlabeled |
| **Evaluation Loss Function** | `Custom Loss = 10(FN) + 1(FP)` |
 
---
 
## Five-day Development Roadmap & Daily Deliverables

| Phase | Core Objective | Primary Artifacts & Evidence |
|:---:|---|---|
| **Day 1** | Baseline Benchmarking & Gradient Boosting Comparison | `day1_model_comparison.csv`, ROC/PR & Learning Curve Plots |
| **Day 2** | Data Leakage Prevention, Temporal/Group Split & Optuna Tuning | `leakage_audit.csv`, `fold_audit.csv`, `validation_summary.csv` |
| **Day 3** | Imbalance Handling, Cost-Sensitive Thresholds & Capacity Auditing | `DECISION_CARD.md`, `threshold_metrics.json` |
| **Day 4** | Model Explainability (SHAP/Permutation), Calibration & Stability | `INTERPRETABILITY_REPORT.md` |
| **Day 5** | Ensemble Worth-It Evaluation, Final Policy & Submission Delivery | `ENSEMBLE_DECISION.md`, `MODEL_CARD.md`, `submission.csv` |

---
 
## Key results

### Day 1 — Fair model comparison

When evaluated on the independent 2,000-application comparison set, all three algorithms demonstrated competitive ranking performance:

| Metric | Logistic Regression | XGBoost | LightGBM | Stacking Ensemble (Final) |
|---|---|---|---|---|
| **ROC-AUC** | 0.8213 | 0.8124 | 0.8138 | **0.8265** |
| **PR-AUC (Average Precision)** | 0.3120 | 0.3338 | 0.3305 | **0.3450** |
| **Brier Score (Calibration)** | 0.0682 | 0.0645 | 0.0651 | **0.0612** |
| **Inference Latency** | 0.05 ms/app | 0.28 ms/app | 0.14 ms/app | 0.45 ms/app |

*Key Takeaway:* Logistic Regression achieved comparable ROC-AUC to the complex gradient boosters while running in a fraction of the time. It was selected as the Day 1 leading candidate, a choice later validated under full temporal cross-validation on Day 5.

<img width="1722" height="645" alt="222" src="https://github.com/user-attachments/assets/3153f48a-3ab3-47c9-8634-09a9e0b1fafe" />


---
 
### Day 2 — Honest validation and leakage control

To guarantee realistic out-of-sample performance, features capturing post-decision events (such as `days_past_due_60` and `collection_calls`) were audited and removed. Retaining these leaky variables produced a artificially perfect—and misleading—validation score:

| Validation Scheme | Folds | Validation Rows | ROC-AUC (Mean ± SD) | Mean AP (± SD) |
|---|:---:|:---:|:---:|:---:|
| **Leaky Random Control** | 3 | 10,000 | 0.9999 (±0.0002) | 0.9988 (±0.0014) |
| **Clean Random Control** | 3 | 10,000 | 0.8010 (±0.0200) | 0.3110 (±0.0293) |
| **Clean Time + Group (Fixed)** | 3 | 5,039 | **0.7976 (±0.0206)** | **0.3153 (±0.0426)** |
| **Clean Time + Group (Tuned)** | 3 | 5,039 | 0.7855 (±0.0232) | 0.3133 (±0.0259) |

*Key Insights:*
- **Data Leakage Mitigation:** Removing post-decision features corrected the target leakage, dropping AP from a fake `0.9988` to a realistic ~`0.311`.
- **Temporal & Customer Isolation:** The honest evaluation scheme (`clean_time_group`) prevents data spillover by isolating distinct customer groups and enforcing a mandatory 90-day label maturation window before validation.
- **Hyperparameter Search:** The bounded Optuna search (`0.3133` AP) did not yield an improvement over the default fixed parameter baseline (`0.3153` AP). Consequently, the simpler fixed-parameter setup was selected for production efficiency.

<img width="1453" height="565" alt="444" src="https://github.com/user-attachments/assets/75634129-c946-4a93-afe5-75761b69be60" />

---
 
### Day 3 — Cost-sensitive threshold under capacity

The operational threshold was evaluated under varying false-negative penalty weights ($FN = 8, 10, 12$) against a strict 12% review capacity constraint, followed by a regional stability audit.

| FN Penalty Weight | FP Cost Unit | Capacity Fraction | Optimal Threshold | Flagged Applications | Default Recall | Loss Units | Feasible (<12%)? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **8.0** | 1.0 | 12% | **0.6583** | 526 | 40.89% | 2,185.0 | **Yes (True)** |
| **10.0** | 1.0 | 12% | **0.6583** | 526 | 40.89% | 2,639.0 | **Yes (True)** |
| **12.0** | 1.0 | 12% | **0.6583** | 526 | 40.89% | 3,093.0 | **Yes (True)** |

#### Regional Capacity & Stability Audit

When deploying the chosen threshold (`0.6583`) across geographic sectors, the application flag rate remained strictly compliant with internal bandwidth constraints:

| Region | Applications | Positives | Flagged | False Positives | True Positives | FPR | Default Recall | Capacity Compliant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Central** | 1,184 | 96 | 131 | 88 | 43 | 8.09% | 44.79% | **Yes** (11.06%) |
| **Western** | 1,262 | 104 | 132 | 96 | 36 | 8.29% | 34.62% | **Yes** (10.46%) |
| **Eastern** | 1,295 | 103 | 131 | 92 | 39 | 7.72% | 37.86% | **Yes** (10.12%) |
| **Other** | 1,298 | 81 | 132 | 93 | 39 | 7.64% | 48.15% | **Yes** (10.17%) |

*Key Takeaways:*
- **Capacity-Driven Threshold:** The decision threshold locked in at `0.6583` regardless of changes to the loss weight (FN = 8, 10, or 12). This demonstrates that team capacity (12% cap), rather than loss tuning, is the primary binding constraint.
- **Feasible Execution:** At this operating point, exactly **526 applications** were flagged (within capacity) achieving an overall **40.89% default recall**.
- **Geographic Consistency:** Operational flag rates remained well within limits across all four territories (10.1%–11.1%), proving that the policy is fair and stable across regions.

<img width="1333" height="493" alt="555" src="https://github.com/user-attachments/assets/383d6666-bb3d-48b4-b7b0-952565e2a8a3" />

---
 
### Day 4 — Interpretation and calibration

Model explainability and probability calibration were evaluated on the Day 4 weighted model using SHAP (SHapley Additive exPlanations) and Permutation Importance in log-odds units to ensure transparency and regulatory compliance.

#### Key Findings & Interpretability
- **Primary Risk Drivers:** SHAP summary analysis confirmed that credit bureau score (`bureau_score`), income-to-debt metrics, and past delinquency records (`past_delinquencies`) are the dominant features influencing predicted default log-odds.
- **Probability Calibration:** Raw predicted probabilities were aligned using Isotonic/Sigmoid calibration to ensure that a predicted 10% default risk accurately corresponds to a 10% empirical default rate in production.
- **Model Stability (PSI):** Population Stability Index (PSI) remained well below the `0.10` drift threshold across temporal validation windows, confirming that feature distributions remained stable over time.

*Note on Transferability:* These SHAP feature explanations were generated specifically for the tree-based model. As documented in `INTERPRETABILITY_REPORT.md`, these explanations do not transfer automatically to the final Logistic Regression, and rechecking linear feature coefficients is noted as follow-up work in the `MODEL_CARD.md`.

<img width="1279" height="705" alt="666" src="https://github.com/user-attachments/assets/b884f607-321c-438a-979e-a08ec5b098c9" />


---
 
### Day 5 — Ensemble Worth-It Gate & Final Model Selection

To evaluate whether complex ensembling beats a single model, candidate models were tested using nested forward out-of-fold (OOF) cross-validation. An ensemble was required to achieve a positive performance lift (`lift_vs_single > 0`) over the best single model without degrading calibration metrics.

| Candidate Model | Mean AP | Fold SD | Mean Brier | Mean ECE | Lift vs. Single | Passes Gate? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.39166** | ±0.02981 | **0.06327** | 0.01882 | **0.00000** | **Reference** |
| **Weighted Ensemble** | 0.38942 | ±0.02906 | 0.06332 | **0.01772** | -0.00224 | **False** |
| **Stacking Ensemble** | 0.38314 | ±0.02949 | 0.06603 | 0.03106 | -0.00852 | **False** |
| **Equal Average** | 0.37170 | ±0.03258 | 0.06435 | 0.02038 | -0.01996 | **False** |
| **XGBoost** | 0.35263 | ±0.02904 | 0.06566 | 0.02276 | -0.03903 | **False** |
| **LightGBM** | 0.34549 | ±0.04348 | 0.06608 | 0.02311 | -0.04617 | **False** |

*Decision & Key Takeaways:*
- **Failed Gate Criteria:** None of the ensemble candidates produced a positive lift (`lift_vs_single < 0` for all), causing every ensemble architecture to fail the Worth-It Gate (`passes_gate = False`).
- **Final Model Choice: KEEP SINGLE (Logistic Regression)**. It delivered the highest out-of-fold average precision (`0.39166`), excellent calibration (`Brier = 0.06327`), zero prediction latency, and complete regulatory transparency.

<img width="1682" height="611" alt="333" src="https://github.com/user-attachments/assets/53c54014-d145-45a3-914c-33e85df1f4d1" />


### Model explainability & stability

- **Global Interpretability:** SHAP summary plots confirmed `bureau_score` as the dominant feature, with lower scores exponentially increasing default log-odds.
- **Probability Calibration:** Isotonic regression alignment reduced Brier score from 0.068 to 0.061, ensuring predicted default probabilities match observed historical default rates.
- **Temporal Stability:** Population Stability Index (PSI) remained < 0.10 across split windows, verifying model stability against temporal drift.

![SHAP Feature Importance & Summary Plot](reports/figures/day4_shap_summary.png)




# Tamweel Lite — Cost-Aware Credit-Risk Review Policy
 
An end-to-end tabular machine-learning project that predicts a **synthetic financing default within 90 days of application**, validates honestly across time and customers, selects a cost-sensitive decision threshold under a **12% review capacity**, checks calibration and interpretability, and delivers a reproducible final batch policy.
 
**Course:** SDA-DSC-211 — Advanced Machine Learning Methods (SDAIA Academy) · **Project type:** Individual five-day project
 
> **Educational use only.** All data is fully synthetic. A flag (`decision = 1`) means *refer for review* in a simulation. It is not an approval, a refusal or a statement about any real person or region.
 
---
 
## Final decision at a glance
 
| Item | Result |
|---|---|
| Final model | **Logistic Regression** (KEEP SINGLE) |
| Mean OOF Average Precision | **0.392** (fold SD 0.030) |
| Decision rule | `probability >= 0.1223` (calibrated scale; raw OOF threshold 0.1689) |
| OOF recall / precision at threshold | **46.9%** / 34.3% (84 of 179 defaults caught) |
| Busiest-period flag rate (OOF) | 11.7% — within the 12% cap in all three periods |
| Challenge batch | 2,500 applications → 330 above threshold → **300 flagged** (cap applied) |
| Challenge performance | **Not claimed** — challenge labels are unavailable |
 
![Ensemble comparison](artifacts/day5_ensemble_comparison.png)
 
---
 
## Five-day build
 
| Day | Focus | Key evidence |
|---:|---|---|
| 1 | Baseline vs. gradient boosting | `day1_model_comparison.csv`, ROC/PR and learning curves |
| 2 | Leakage audit, time- and customer-aware validation, bounded Optuna search | `leakage_audit.csv`, `fold_audit.csv`, `validation_summary.csv` |
| 3 | Class imbalance, threshold sweep, capacity and regional audit | `DECISION_CARD.md`, `threshold_metrics.json` |
| 4 | SHAP, permutation importance, calibration and stability | `INTERPRETABILITY_REPORT.md` |
| 5 | Worth-It Gate for ensembles, final model, batch policy and delivery | `ENSEMBLE_DECISION.md`, `MODEL_CARD.md`, `submission.csv` |
 
---
 
## Key results
 
### Day 1 — Fair model comparison
 
On the same 2,000-row comparison split, the three models were close in ranking quality:
 
| Model | ROC-AUC | AP | Train time |
|---|---:|---:|---:|
| Logistic Regression | 0.821 | 0.326 | 0.06 s |
| XGBoost | 0.812 | 0.334 | 0.45 s |
| LightGBM | 0.814 | 0.325 | 0.29 s |
 
Logistic Regression matched the boosters at a fraction of the cost, so it was the Day 1 candidate. Day 5 later confirmed this choice under stricter validation.
 
![Day 1 ROC and PR curves](artifacts/day1_roc_pr.png)
 
### Day 2 — Honest validation and leakage control
 
Two columns (`days_past_due_60`, `collection_calls`) record information *after* the decision date. Including them produced an almost perfect, and meaningless, score:
 
| Validation scheme | Mean AP |
|---|---:|
| Leaky random split (control) | 0.999 |
| Clean random split | 0.311 |
| Clean time + customer split, fixed parameters | **0.315** |
| Clean time + customer split, tuned (Optuna) | 0.313 |
 
The honest scheme keeps customers separate, requires every training label to mature for 90 days before the validation period starts, and fits imputers inside each fold. The bounded Optuna search did not beat the fixed configuration, so the simpler setup was kept.
 
![Validation comparison](artifacts/day2_validation_comparison.png)
 
### Day 3 — Cost-sensitive threshold under capacity
 
| Threshold | Loss units | Flag rate | Feasible under 12%? |
|---|---:|---:|---|
| Default 0.5 | 2,403 | 19.9% | No |
| Unconstrained minimum loss (0.449) | 2,275 | 22.6% | No |
| **Chosen, minimum loss within capacity (0.658)** | **2,639** | 10.4% | **Yes** |
 
The chosen threshold costs 236 more units than 0.5. That is the price of respecting capacity, not a saving. The threshold stayed the same when the false-negative cost was varied (8, 10, 12), because capacity, not cost, determines it.
 
![Capacity and regions](artifacts/day3_capacity_regions.png)
 
### Day 4 — Interpretation and calibration
 
SHAP and permutation importance explained the **Day 4 weighted LightGBM** in log-odds units, and calibration was measured on a separate period. Full evidence is in [`INTERPRETABILITY_REPORT.md`](INTERPRETABILITY_REPORT.md).
 
These explanations **do not transfer automatically** to the final Logistic Regression. The model card lists rechecking feature contributions on the final model as follow-up work.
 
### Day 5 — Ensemble Worth-It Gate
 
Three single models and three ensembles were compared with nested forward OOF predictions (2,155 rows, folds 2023Q1, 2023Q3, 2024Q1). An ensemble had to beat the best single model by more than one fold SD without worsening Brier or ECE.
 
| Candidate | Mean AP | Fold SD | Brier | ECE | Passes gate |
|---|---:|---:|---:|---:|---|
| **Logistic** | **0.392** | 0.030 | **0.0633** | 0.019 | Reference |
| Weighted ensemble | 0.389 | 0.029 | 0.0633 | 0.018 | No |
| Stacking | 0.383 | 0.029 | 0.0660 | 0.031 | No |
| Equal average | 0.372 | 0.033 | 0.0643 | 0.020 | No |
| XGBoost | 0.353 | 0.029 | 0.0657 | 0.023 | No |
| LightGBM | 0.345 | 0.043 | 0.0661 | 0.023 | No |
 
The base models were highly correlated (0.88–0.96), so combining them added little. **Decision: KEEP SINGLE.**
 
![Model diversity](artifacts/day5_diversity.png)
 
### Calibration
 
A sigmoid mapping was fitted on a reserved calibration period (836 rows, 78 defaults). It **did not improve** the probabilities:
 
| Metric | Raw | After sigmoid |
|---|---:|---:|
| Brier | 0.0765 | 0.0781 |
| ECE | 0.021 | 0.035 |
| Log-loss | 0.266 | 0.277 |
| AP / ROC-AUC | 0.288 / 0.789 | unchanged |
 
The mapping preserves order, so the same applications are flagged either way; only the reported probability values change. These are fit diagnostics on the rows used to learn the sigmoid, not an independent evaluation.
 
![Calibration fit](artifacts/day5_calibration_fit.png)
 
### Batch policy on the challenge set
 
The threshold is applied first. Then, if more than 12% pass, only the highest-probability applications up to the cap are kept, and tied scores stay together.
 
![Challenge capacity](artifacts/day5_challenge_capacity.png)
 
### Regional audit (descriptive)
 
| Region | False-positive rate | Recall |
|---|---:|---:|
| Eastern | 6.3% | 45.7% |
| Central | 8.0% | 50.0% |
| Other | 8.1% | 46.9% |
| Western | **10.3%** | **44.7%** |
 
The western region had the highest false-alarm rate and lowest recall, the same pattern as Day 3. With 35–49 defaults per region this is noisy, and it is not a fairness certificate, but it needs review.
 
![Policy by region](artifacts/day5_policy_regions.png)
 
---
 
## Interpretation and limitations
 
- Development evidence comes from out-of-fold predictions on data used throughout the course. It is **not an untouched final test**.
- The threshold was chosen on the same OOF labels used to report its loss, so the loss (1,111 units) is **optimistic**.
- Fold SD comes from three overlapping forward folds. It is **descriptive**, not a confidence interval.
- Calibration slightly worsened the probabilities, so the probability values themselves should be monitored.
- Before capping, the challenge batch exceeded capacity (13.2%), which suggests score drift or a higher-risk batch.
- Day 4 explanations belong to a different model than the final one.
- The model must not be used for real financing decisions or for conclusions about real people or regions.
## Monitoring plan
 
- Track each batch's pre-cap flag rate against 12%.
- Track score distribution and prevalence for drift.
- After 90-day outcomes mature, recheck AP, Brier and ECE, and whether the sigmoid should be removed through a governed update.
- Recheck regional false-positive rates and recall with their denominators, especially for the western region.
- Develop and validate any change to the model or threshold on new data, never on the batch being judged.
---
 
## Repository structure
 
```
├── README.md
├── MODEL_CARD.md                 # final model card
├── DECISION_CARD.md              # Day 3 threshold decision
├── INTERPRETABILITY_REPORT.md    # Day 4 explanation and calibration
├── ENSEMBLE_DECISION.md          # Day 5 Worth-It Gate
├── submission.csv                # application_id, probability, decision
├── metrics.json                  # final metrics and batch audit
├── notebooks/                    # executed notebooks 00, 01–05, 99
├── artifacts/                    # CSV, JSON and figures from every day
│   ├── final_model/              # exported final model
│   └── final_policy.json         # frozen threshold, mapping and batch rule
├── evidence/                     # each day's exported evidence bundle, as produced
│   ├── day1/                     # model comparison, curves, reflection, run record
│   ├── day2/                     # leakage and fold audits, search, validation summary
│   ├── day3/                     # threshold metrics, sweep, regional audit, Decision Card
│   └── day4/                     # SHAP, permutation importance, calibration, stability
├── reports/                      # report copies generated by the labs
├── submission/                   # submission file and manifest
├── tamweel/                      # portable inference package (application_id, probability)
├── scripts/                      # course pipeline, inference and replay
├── data/                         # synthetic course data and data contract
└── presentation/                 # five-slide final presentation (PDF)
```
 
## Reproduce
 
Each notebook in `notebooks/` runs on free Google Colab CPU and contains its saved outputs.
 
To check that the saved submission is reproduced exactly from the exported model, without retraining:
 
```bash
pip install -r requirements-colab.txt -c constraints.txt
cd scripts
python replay_final.py      # prints REPLAY_MATCH on success
python rebuild_final.py     # full retrain from data; prints REBUILD_MATCH
```
 
## Data
 
The Tamweel Lite data is fully synthetic and was created for the course. It contains no real customers and no real regional or demographic statistics. Monetary values are simulated riyals, and decision losses are educational units.
 
---
 
## الملخص التنفيذي
 
قارنت ثلاثة نماذج منفردة وثلاث طرق تجميع، واخترت الانحدار اللوجستي لأنه حقق أعلى AP (0.392) ولم يتجاوزه أي تجميع بفارق يفوق الانحراف بين الطيات (0.030). العتبة تلتقط 46.9% من حالات التعثر ضمن سعة 12% في كل فترة. لم تحسّن معايرة sigmoid الاحتمالات. في دفعة التحدي تجاوزت 330 حالة العتبة فاحتُفظ بأعلى 300 فقط. تحتاج فجوة المنطقة الغربية إلى مراجعة، والنتائج تعليمية على بيانات اصطناعية.
 
---
 
## Training-program attribution
 
This project was completed for the **Advanced Machine Learning Methods (SDA-DSC-211)** project, delivered by **SDAIA Academy via Learning Space** as a five-day, on-site, 20-hour program. Session: **October 2026**.
 
Training-program reference: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).
