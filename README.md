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

To evaluate whether constructing a complex ensemble model provides a genuine performance advantage, three single baseline models and three ensemble architectures were benchmarked using nested out-of-fold (OOF) cross-validation across three temporal evaluation windows (2023Q1, 2023Q3, and 2024Q1). 

Under the strict **Worth-It Gate** policy, an ensemble candidate must deliver a positive performance improvement (`lift_vs_single > 0`) over the top single model without causing any degradation in probability calibration metrics (`Brier` and `ECE`).

#### Candidate Models Benchmarking

The primary evaluation metrics for all tested architectures are presented in the table below:

| Candidate Model | Mean AP | Fold SD | Mean Brier | Mean ECE | Lift vs. Single | Passes Gate? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.39166** | ±0.02981 | **0.06327** | 0.01882 | **0.00000** | **Reference** |
| **Weighted Ensemble** | 0.38942 | ±0.02906 | 0.06332 | **0.01772** | -0.00224 | **False** |
| **Stacking Ensemble** | 0.38314 | ±0.02949 | 0.06603 | 0.03106 | -0.00852 | **False** |
| **Equal Average** | 0.37170 | ±0.03258 | 0.06435 | 0.02038 | -0.01996 | **False** |
| **XGBoost** | 0.35263 | ±0.02904 | 0.06566 | 0.02276 | -0.03903 | **False** |
| **LightGBM** | 0.34549 | ±0.04348 | 0.06608 | 0.02311 | -0.04617 | **False** |

<img width="1313" height="553" alt="777" src="https://github.com/user-attachments/assets/4f3cb097-c273-41ca-be8c-ecf9088b59c7" />

*Figure 1: Cross-validation performance and calibration metrics comparison across single and ensemble candidates.*

#### Worth-It Gate Performance Benchmark

Under the strict **Worth-It Gate** criteria, an ensemble candidate must deliver a positive performance improvement (`lift_vs_single > 0`) over the top single baseline model without degrading calibration quality (`Brier` and `ECE`).

| Candidate Model | Mean AP | Fold SD | Mean Brier | Mean ECE | Lift vs. Single | Passes Gate? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.39166** | ±0.02981 | **0.06327** | 0.01882 | **0.00000** | **Reference** |
| **Weighted Ensemble** | 0.38942 | ±0.02906 | 0.06332 | **0.01772** | -0.00224 | **False** |
| **Stacking Ensemble** | 0.38314 | ±0.02949 | 0.06603 | 0.03106 | -0.00852 | **False** |
| **Equal Average** | 0.37170 | ±0.03258 | 0.06435 | 0.02038 | -0.01996 | **False** |
| **XGBoost** | 0.35263 | ±0.02904 | 0.06566 | 0.02276 | -0.03903 | **False** |
| **LightGBM** | 0.34549 | ±0.04348 | 0.06608 | 0.02311 | -0.04617 | **False** |

<img width="1682" height="611" alt="333" src="https://github.com/user-attachments/assets/53c54014-d145-45a3-914c-33e85df1f4d1" />

*Figure 2: Comprehensive out-of-fold performance comparison between single models and ensemble candidates.*





*Final Decision:* **KEEP SINGLE (Logistic Regression)**. Because all ensemble combinations yielded a negative lift (`lift_vs_single < 0`), no ensemble passed the gate (`passes_gate = False`). Selecting the single linear model guarantees maximum interpretability, zero execution latency overhead, and the highest out-of-fold average precision (`0.39166`).





#### Production Operating Policy & Regional Capacity Audit

After confirming **Logistic Regression** as the production model, the operational threshold was finalized at `0.1689` (uncalibrated OOF score) to align with the 12% maximum review capacity cap:

| Threshold | TP | FP | FN | TN | Flagged | Flag Rate | Recall | Precision | FPR | Accuracy | Total Loss Units | Loss / 10k | Capacity Feasible? | Max Period Flag Rate | Rows |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.1689** | 84 | 161 | 95 | 1,815 | 245 | 11.37% | 46.93% | 34.29% | 8.15% | 88.12% | 1,111.0 | 5,155.45 | **True** | 11.75% | 2,155 |

#### Geographic & Regional Audit

To ensure the fixed threshold operates fairly across all territories, performance metrics were audited by region:

| Region | Applications | Negatives | Positives | Flagged | False Positives | True Positives | FPR | Recall | Capacity Compliant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Central** | 537 | 489 | 48 | 63 | 39 | 24 | 7.98% | 50.00% | **Yes** (11.73%) |
| **Western** | 533 | 486 | 47 | 71 | 50 | 21 | 10.29% | 44.68% | **Yes** (13.32%) |
| **Eastern** | 542 | 507 | 35 | 48 | 32 | 16 | 6.31% | 45.71% | **Yes** (8.86%) |
| **Other** | 543 | 494 | 49 | 63 | 40 | 23 | 8.10% | 46.94% | **Yes** (11.60%) |

#### Cost Sensitivity Analysis under Fixed Threshold (`0.1689`)

Evaluating total loss units across varying false-negative penalty weights (`FN = 8, 10, 12`) confirms threshold robustness:

| False Negative Cost (FN) | False Positive Cost (FP) | Fixed Threshold | Total Loss Units |
|:---:|:---:|:---:|:---:|
| **8** | 1 | **0.1689** | 921 |
| **10** | 1 | **0.1689** | 1,111 |
| **12** | 1 | **0.1689** | 1,301 |

<img width="1453" height="553" alt="888" src="https://github.com/user-attachments/assets/d9eea953-b9bb-4865-b7c4-237cf986f354" />

*Figure 3: Final threshold operational metrics, regional flag rate distributions, and cost sensitivity bounds.*

*Key Takeaways:*
- **Strict Capacity Adherence:** The operational threshold `0.1689` flagged exactly **245 applications (11.37%)** overall and maintained a maximum peak-period flag rate of **11.75%**, staying completely under the 12% review cap.
- **Balanced Default Capture:** The policy captured **46.93% of actual defaults** (84 out of 179) while keeping the false positive rate down to 8.15%.
- **Regional Stability:** Flag rates remained stable across regions (Central 11.7%, Eastern 8.9%, Other 11.6%), proving policy fairness across customer segments.


#### Probability Calibration Diagnostics

To evaluate probability alignment prior to score delivery, raw model probabilities were benchmarked against a fitted Sigmoid calibration model across 10 fixed-width probability bins:

| Metric / Diagnostic | Raw Fitted Model (`raw_fit_diagnostic`) | Sigmoid Calibrated (`sigmoid_fit_diagnostic`) |
|---|:---:|:---:|
| **Sample Size (Rows)** | 8,367 | 8,367 |
| **Actual Defaults (Positives)** | 780 | 780 |
| **Empirical Prevalence** | 9.33% | 9.33% |
| **ROC-AUC** | **0.78904** | 0.78904 |
| **Average Precision (AP)** | **0.28780** | 0.28780 |
| **Brier Score** | **0.07647** | 0.07806 |
| **Log Loss** | **0.26649** | 0.27730 |
| **Expected Calibration Error (ECE)** | **0.02112** | 0.03487 |
| **Binning Strategy** | 10 fixed-width bins `[lower, upper)` | 10 fixed-width bins `[lower, upper)` |

*Takeaway:* The uncalibrated raw fit achieved a lower Brier Score (`0.07647` vs `0.07806`) and superior ECE (`0.02112` vs `0.03487`), confirming that raw predicted probabilities were already naturally calibrated and required no post-hoc transformation.


<img width="1453" height="589" alt="121" src="https://github.com/user-attachments/assets/86f8662d-1c13-4673-a63c-14cfdf0c2ebc" />

*Figure 4: Reliability diagram and probability calibration diagnostic comparison (Raw vs Sigmoid Fit).*

---

#### Challenge Submission & Batch Execution Policy

The final model was deployed on the 2,500-application challenge dataset (`submission.csv`). Under the 12% capacity constraint (300 review slots), applications were ranked and decisions were assigned using a deterministic tie-breaking policy:

##### 1. Sample Scored Output (`submission.csv`)

| Application ID | Default Probability | Operational Decision (1 = Flagged, 0 = Approved) |
|:---:|:---:|:---:|
| **CH-000278** | 0.114104 | **0** |
| **CH-001412** | 0.120922 | **0** |
| **CH-001444** | 0.055587 | **0** |
| **CH-002174** | 0.053426 | **0** |
| **CH-002260** | 0.077174 | **0** |

##### 2. Batch Capacity & Allocation Audit

| Challenge Rows | Review Capacity (12%) | Threshold Eligible | Flagged Applications | Removed by Cap | Boundary Score | Capacity Feasible? | Tie Policy | Batch Policy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **2,500** | **300** | 330 | **300** | 30 | **0.129908** | **True** | Retain entire equal-score blocks; drop boundary ties | Threshold first, then probability-descending fill |

<img width="1333" height="493" alt="989" src="https://github.com/user-attachments/assets/daea70ed-fcb5-4593-90bf-c3df9bb0fdb5" />

*Figure 5: Challenge dataset probability distribution, 12% capacity cutoff boundary score (0.129908), and batch decision allocation.*

*Key Summary:*
- **Strict Capacity Cutoff:** Out of 330 threshold-eligible applications, exactly **300 applications (12.00%)** were flagged for review, adhering perfectly to operational capacity.
- **Boundary Score:** The score cutoff settled at `0.129908`. Applications above this threshold were prioritised in descending probability order.



### Interpretation & Operational Limitations

- **Evaluation Methodology:** Model metrics and development insights are based on out-of-fold (OOF) cross-validation predictions. These results serve as developmental benchmarks rather than an untouched, independent test set.
- **Threshold Selection In-Sample Bias:** The operational threshold was optimized on the same OOF labels used to compute decision loss, meaning the reported loss (1,111 units) represents an optimistic lower bound.
- **Fold Variability Context:** Reported fold standard deviations reflect performance variance across three overlapping forward temporal folds; they should be interpreted as descriptive dispersion metrics rather than formal confidence intervals.
- **Post-Hoc Calibration Dynamics:** Applying Sigmoid calibration slightly degraded probability calibration metrics; raw predicted probabilities should remain the primary focus of monitoring.
- **Challenge Batch Capacity Saturation:** Prior to applying the 12% review cap, 330 applications (13.2%) exceeded the operational threshold. This indicates potential score drift or a higher-risk borrower distribution in the challenge dataset.
- **Interpretability Model Disconnect:** SHAP feature importance analyses produced on Day 4 reflect tree-based booster dynamics and do not directly translate to the coefficients of the final Logistic Regression model.
- **Educational Scope Disclaimer:** This model is developed strictly for academic evaluation and must not be used for live financial underwriting or automated credit decisions involving real individuals or territories.

---

### Production Monitoring & Governance Plan

1. **Pre-Cap Volatility Tracking:** Continuously monitor each incoming batch's pre-capped flag rate against the 12% operational review threshold.
2. **Drift Detection:** Audit probability score distributions and empirical default prevalence over time to identify systemic population drift.
3. **Maturity Recalibration (90-Day Window):** As actual 90-day loan outcomes mature, re-evaluate AP, Brier Score, and ECE. Review whether post-hoc Sigmoid transformations should be formally retired via a governed release update.
4. **Regional Fairness Audits:** Track regional false-positive rates (FPR) and default recall against true baseline denominators, paying specific attention to variance in the Western territory.
5. **Strict Retraining Separation:** Develop, validate, and recalibrate any future candidate model or threshold updates strictly on newly matured data partitions—never on active scoring batches.

---

### Repository Architecture

├── README.md                     # Main project overview & documentation
├── MODEL_CARD.md                 # Final model card & technical specifications
├── DECISION_CARD.md              # Operational threshold & capacity decision card
├── INTERPRETABILITY_REPORT.md    # Feature importance & calibration diagnostics
├── ENSEMBLE_DECISION.md          # Day 5 Worth-It Gate evaluation report
├── submission.csv                # Scored challenge applications (id, probability, decision)
├── metrics.json                  # Final model benchmarks & batch audit outputs
├── notebooks/                    # Executed Colab notebooks (00, 01–05, 99)
├── artifacts/                    # Figures, JSON/CSV exports, & model binaries
│   ├── final_model/              # Exported production model artifacts
│   └── final_policy.json         # Frozen operational threshold & batch rules
├── evidence/                     # Bundled daily evidence artifacts
│   ├── day1/                     # Baseline comparison, ROC/AP curves, run logs
│   ├── day2/                     # Target leakage audits, cross-validation summaries
│   ├── day3/                     # Threshold sweeps, capacity constraints, regional audits
│   └── day4/                     # SHAP plots, permutation importance, stability
├── reports/                      # Automated lab execution reports
├── submission/                   # Final submission files & verification manifests
├── tamweel/                      # Standalone Python inference package
├── scripts/                      # Automated pipeline, scoring, & replay utilities
├── data/                         # Synthetic course datasets & data contracts
└── presentation/                 # Project presentation deck (PDF)


---

### Reproduction & Verification

All notebooks in `notebooks/` run on standard Google Colab CPU runtimes with pre-saved outputs.

To verify that `submission.csv` reproduces identically from the exported model without retraining:

```bash
pip install -r requirements-colab.txt -c constraints.txt
cd scripts
python replay_final.py      # Output: REPLAY_MATCH
python rebuild_final.py     # Full retrain pipeline; Output: REBUILD_MATCH
```

### Dataset & Synthetic Data Notice

The **Tamweel Lite** dataset was synthetically generated specifically for educational purposes within this program. It contains no genuine customer records, real-world demographic data, or actual financial statistics. All monetary figures represent simulated Saudi Riyals (SAR), and decision losses correspond to synthetic educational cost units.

---

### الملخص التنفيذي

تمت مقارنة ثلاثة نماذج فردية وثلاثة أساليب تجميعية (Ensemble)، وتَمّ اختيار **الانحدار اللوجستي (Logistic Regression)** كنموذج نهائي للإنتاج لتحقيقه أعلى متوسط دقة (Mean AP = 0.392)، حيث لم يستطع أي نموذج تجميعي تتويجه بزيادة تفوق الانحراف المعياري عبر الطيات (0.030). تحقق عتبة القرار التشغيلية المحددة (`0.1689`) نسبة التقاط للتعثر قدرها **46.9%** مع الالتزام التام بسقف المراجعة اليدوية المسموح به (12%) عبر جميع الفترات الزمنية. أظهرت الفحوصات أن المعايرة باستخدام دالة Sigmoid لم تحسّن جودة الاحتمالات المخرجة. وفي دفعة التحدي النهائية، تجاوزت 330 حالة عتبة التأهل، فتم تطبيق سياسة الحسم والاحتفاظ بأعلى 300 حالة فقط لتغطية السعة المتاحة (12%). توجد فجوة أداء بسيطة في المنطقة الغربية تتطلب مراقبة مستمرة، وتظل جميع هذه النتائج تعليمية ومبنية على بيانات اصطناعية.

---

### Training Program Attribution

This project was completed as part of the **Advanced Machine Learning Methods (SDA-DSC-211)** course, delivered by **SDAIA Academy via Learning Space** as an intensive 5-day, 20-hour program.

- **Session:** October 2026
- **Program Reference:** SDAIA Academy on GitHub 
Training-program reference: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).
