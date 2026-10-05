# Homework 2: Machine Learning for Regression

Course: **[DataTalks.Club Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/)**  
Student: **Diego Gonzalez** ([@NativelogXD](https://github.com/NativelogXD))

---

## Verified Solution Summary

| Question | Topic | Computed Value | Selected Option | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Q1** | Column with missing values | `horsepower` (877 missing / 8.77%) | `'horsepower'` | Verified |
| **Q2** | Median of `horsepower` | `254.0` | `254` | Verified |
| **Q3** | Missing value handling (0 vs mean) | RMSE 0: `2.205` vs RMSE Mean: `2.202` | `With mean` | Verified |
| **Q4** | Best regularization parameter $r$ | $r=0$ gives optimal validation RMSE `2.2053` | `0` | Verified |
| **Q5** | Seed stability analysis | $\sigma = 0.02878 \approx 0.029$ across seeds 0-9 | `0.029` | Verified |
| **Q6** | Test RMSE evaluation | Test RMSE = `2.23577 \approx 2.236` ($r=0.001$, seed 9) | `2.236` | Verified |

---

## Files in this Module

1. [`homework_2.ipynb`](./homework_2.ipynb) — Executable Jupyter Notebook with full step-by-step mathematical reasoning, Normal equation derivations, and output displays.
2. [`homework_2.py`](./homework_2.py) — Standalone reproducible Python script.
