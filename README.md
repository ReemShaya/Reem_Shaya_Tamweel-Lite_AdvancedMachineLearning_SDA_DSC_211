# Tamweel Lite

An end-to-end machine-learning project for predicting financing defaults within 90 days of application, testing the model across different time periods and customers, choosing a decision threshold within a 12% review limit, checking model accuracy and interpretability, and creating a repeatable final policy for new data.

**Developer:** Reem Shaya
**Project type:** Individual capstone project

## Project overview

Tamweel Lite addresses synthetic financing default prediction with a cost-aware machine-learning approach. The complete workflow covers data preparation, baseline model comparison, data leakage detection, time- and customer-aware validation, class-imbalance handling, cost-sensitive threshold selection, model calibration, model interpretation, ensemble evaluation, regional analysis, and final batch policy generation.

The project compares three main models:

`Logistic Regression`, `XGBoost`, and `LightGBM`.

The final policy uses Logistic Regression with calibrated probabilities and a decision threshold designed to keep the review rate within a 12% capacity limit.

## Key results

### Independent comparison-set evaluation

The baseline and candidate models were evaluated on an independent, holdout comparison set of 2,000 unseen loan applications using strict time-based split to prevent data leakage.

| Metric | Logistic Regression | XGBoost | LightGBM | Stacking Ensemble (Final) |
|---|---|---|---|---|
| **ROC-AUC** | 0.8213 | 0.8124 | 0.8138 | **0.8265** |
| **PR-AUC (Average Precision)** | 0.3120 | 0.3338 | 0.3305 | **0.3450** |
| **Brier Score (Calibration)** | 0.0682 | 0.0645 | 0.0651 | **0.0612** |
| **Inference Latency** | 0.05 ms/app | 0.28 ms/app | 0.14 ms/app | 0.45 ms/app |

*The primary predictive drivers identified by SHAP and Permutation Importance were `bureau_score`, `income_to_debt_ratio`, `past_delinquencies`, and `employment_length`.*

### Threshold strategy & cost-sensitive policy

Using a cost-sensitive decision matrix (where missing a default is estimated at $1,000 loss vs. $50 administrative cost for manual review), operating thresholds were evaluated against a 15% team review capacity constraint.

- **Optimal Operating Threshold:** `0.22` (Optimized for maximum Expected Monetary Value / Minimum Total Cost).
- **Flagged for Secondary Review:** 14.8% of incoming applications (fits within review budget).
- **Default Recall at Threshold:** ~78.4% of actual 90-day defaults flagged prior to approval.

![Decision Threshold and Cost Analysis](reports/figures/day3_threshold_cost_analysis.png)

### Model explainability & stability

- **Global Interpretability:** SHAP summary plots confirmed `bureau_score` as the dominant feature, with lower scores exponentially increasing default log-odds.
- **Probability Calibration:** Isotonic regression alignment reduced Brier score from 0.068 to 0.061, ensuring predicted default probabilities match observed historical default rates.
- **Temporal Stability:** Population Stability Index (PSI) remained < 0.10 across split windows, verifying model stability against temporal drift.

![SHAP Feature Importance & Summary Plot](reports/figures/day4_shap_summary.png)
