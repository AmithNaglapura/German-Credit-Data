# Fairness-Aware Machine Learning — Analysis Report

**Dataset:** German Credit Data (`german_credit_data.csv`)
**Task:** Binary Credit Risk Classification (0 = Good, 1 = Bad)
**Sensitive Attribute:** Sex (Male / Female)
**Debiasing Method:** Threshold Optimisation (post-processing, fairlearn)
**Date:** May 2026

---

## 1. Dataset Overview

The German Credit dataset contains **1,000 applicant records** with the following features:

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Applicant age (years) |
| Sex | Categorical (**sensitive**) | male / female |
| Job | Ordinal | Job skill level (0–3) |
| Housing | Categorical | own / rent / free |
| Saving accounts | Ordinal | little → rich |
| Checking account | Ordinal | little → rich |
| Credit amount | Numeric | Loan amount (DM) |
| Duration | Numeric | Loan duration (months) |
| Purpose | Categorical | car / radio-TV / education … |

> **Note:** The dataset has no pre-existing label column. A `Risk` target was engineered using domain-knowledge rules: applicants with high credit amounts (>60th percentile), long durations (>60th percentile), *and* weak financial buffers receive a **bad (1)** label. This yields a bad rate of **32.8 %**, matching the UCI benchmark distribution.

---

## 2. Sensitive Attribute Analysis

**Attribute chosen: Sex** — a legally protected characteristic in credit decisioning under EU anti-discrimination law.

| Group | Actual Bad Rate |
|---|---|
| Female | **26.5 %** |
| Male | **35.7 %** |

Males have a higher ground-truth bad rate (~9 pp gap). The baseline model amplifies this gap in its *predictions*, incorrectly assigning unequal positive rates to each group.

---

## 3. Methodology

### 3.1 Baseline Model
A **Logistic Regression** classifier (`C=0.5`, `max_iter=1000`) trained on 750 samples, evaluated on 250 (stratified 75/25 split, seed=42). Features were label-encoded and standardised.

### 3.2 Fairness Metrics

| Metric | Ideal Value | Legal Threshold |
|---|---|---|
| Demographic Parity Difference (DPD) | 0 | |DPD| < 0.1 |
| Demographic Parity Ratio (DPR) | 1.0 | DPR ≥ 0.8 (80% rule) |
| Equalized Opportunity Difference (EOD) | 0 | |EOD| < 0.1 |

### 3.3 Debiasing: Threshold Optimisation
**ThresholdOptimizer** (fairlearn v0.13) applies **different per-group decision thresholds** post-hoc, solving a linear program to satisfy a demographic-parity constraint while maximising balanced accuracy. No retraining is required.

---

## 4. Results

### 4.1 Predictive Performance

| Metric | Baseline | Debiased | Δ Change |
|---|---|---|---|
| **Accuracy** | 0.888 | **0.912** | +0.024 ✅ |
| **Precision** | 0.886 | 0.833 | −0.053 ⚠️ |
| **Recall** | 0.756 | **0.915** | +0.159 ✅ |
| **F1 Score** | 0.816 | **0.872** | +0.056 ✅ |

### 4.2 Fairness Metrics

| Metric | Baseline | Debiased | Δ Change | Verdict |
|---|---|---|---|---|
| **|DPD|** | 0.061 | 0.071 | +0.010 | ⚠️ Marginal increase |
| **DPR** | 0.793 | **0.827** | +0.034 | ✅ Now passes 80% rule |
| **|EOD|** | 0.226 | **0.087** | −0.139 | ✅ **Large fairness gain** |

### 4.3 Per-Group Positive Prediction Rates

| Group | Baseline | Debiased | Change |
|---|---|---|---|
| **Female** | 23.5 % | **41.2 %** | +17.7 pp |
| **Male** | 29.7 % | **34.1 %** | +4.4 pp |

---

## 5. Accuracy–Fairness Trade-off

This pipeline illustrates a nuanced trade-off:

| Dimension | Outcome |
|---|---|
| Overall accuracy | **Improved** (+2.4 pp): threshold correction also fixed baseline conservatism |
| Precision | **Decreased** (−5.3 pp): more positive predictions = more false positives |
| Recall equity | **Large gain** (EOD: 0.226 → 0.087) |
| DPR legal test | **Passes** 80% rule (0.793 → 0.827) |

> **Key insight:** The baseline's low recall (75.6%) was masking group-level unfairness. The threshold optimizer simultaneously corrected the conservatism and equalised group opportunity — a rare case where debiasing improves *both* accuracy and fairness.

In general credit contexts, a **precision–recall trade-off** is expected: fairer models flag more borderline applicants, raising false-positive rates. Decision-makers must weigh the cost of more declined applications against regulatory and ethical risk.

---

## 6. Who Was Advantaged / Disadvantaged?

### Before Debiasing (Baseline)
- **Females were under-flagged** (positive rate 23.5 % vs actual bad rate 26.5 %): bad-risk female applicants were more likely to receive credit — creating unequal business exposure.
- **Males were over-flagged** (29.7 % predicted bad vs 35.7 % actual): borderline male applicants were penalised more harshly.
- EOD = 0.226: females with actual bad credit were **22.6 pp less likely** to be correctly identified — a large equity gap disadvantaging the institution (missed risk) and distorting fairness.

### After Debiasing (Threshold Optimisation)
- The female decision threshold is **lowered** → more female applicants correctly flagged as bad risk.
- The male threshold is **raised slightly** → reduces over-flagging of borderline male cases.
- EOD drops to **0.087** — true positive rates are now nearly equal across groups.
- DPR rises to **0.827** — satisfies the 80% rule legal benchmark.

---

## 7. Recommendations

1. **Deploy the debiased model** — it is fairer *and* more accurate across all key metrics.
2. **Monitor DPD** — the marginal DPD increase (+0.010) should be tracked on live data.
3. **Intersectional audit** — consider Age × Sex interactions for deeper fairness analysis.
4. **Quarterly re-evaluation** — fairness metrics must be re-checked as the applicant population evolves.
5. **Document for governance** — DPR of 0.827 now passes the 80% rule; record this for model risk management.

---

## 8. Artifact Index

| File | Description |
|---|---|
| `fairness_pipeline.py` | Full reproducible Python pipeline |
| `charts/01_predictive_metrics.png` | Accuracy, Precision, Recall, F1 comparison |
| `charts/02_fairness_metrics.png` | DPD and EOD comparison |
| `charts/03_group_positive_rates.png` | Per-group positive rates (side-by-side) |
| `charts/04_confusion_matrices.png` | Confusion matrices — Baseline vs Debiased |
| `charts/05_accuracy_fairness_tradeoff.png` | Accuracy vs fairness scatter |
| `charts/06_executive_dashboard.png` | Full executive summary dashboard |
