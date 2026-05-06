# Fairness-Aware Machine Learning Pipeline — v2

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Fairlearn](https://img.shields.io/badge/fairness-fairlearn-F72585.svg)](https://fairlearn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements an end-to-end fairness-aware classification pipeline using the **German Credit Dataset**. It addresses algorithmic bias across two sensitive attributes: **Sex** and **Age Group**. The pipeline compares a standard Logistic Regression baseline against a debiased model using Fairlearn's post-processing techniques.

## 📌 Project Overview

In credit scoring, models often inadvertently discriminate against protected groups. This project demonstrates how to identify and mitigate such disparities.

*   **Task**: Binary credit-risk classification (0 = Good, 1 = Bad).
*   **Sensitive Features**: 
    *   `Sex`: Male, Female.
    *   `AgeGroup`: Young (≤30), Middle (31–49), Senior (50+).
*   **Mitigation Strategy**: Post-processing via `ThresholdOptimizer` to satisfy **Demographic Parity** constraints.
*   **Objective**: Maximize Balanced Accuracy while minimizing fairness gaps.

## 🛠️ Tech Stack

*   **Data Handling**: `pandas`, `numpy`
*   **Machine Learning**: `scikit-learn` (Logistic Regression)
*   **Fairness Toolkit**: `fairlearn`
*   **Visualization**: `matplotlib`, `seaborn`

## 📊 Fairness Metrics Explained

We evaluate the trade-off between performance and equity using:

1.  **Demographic Parity Difference (DPD)**: Measures the difference in the rate of positive outcomes between groups. (Goal: $0$)
2.  **Equalized Odds Difference (EOD)**: Measures the difference in True Positive and False Positive rates between groups.
3.  **Positive Prediction Rate (PPR)**: The actual percentage of individuals in each group predicted as "Bad Credit."

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the German Credit Data CSV (`german_credit_data.csv`) in the root directory.

### 2. Installation
```bash
pip install pandas numpy scikit-learn fairlearn matplotlib seaborn
