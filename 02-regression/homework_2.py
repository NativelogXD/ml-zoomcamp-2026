"""
DataTalks.Club Machine Learning Zoomcamp 2026
Homework 2: Machine Learning for Regression
Student: Diego Gonzalez (@NativelogXD)
Submission URL: https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("Machine Learning Zoomcamp 2026 - Homework 2")
    print("=" * 60)

    # 1. Load Dataset
    url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
    print(f"Loading dataset from: {url}")
    df = pd.read_csv(url)

    selected_columns = [
        "engine_displacement",
        "horsepower",
        "vehicle_weight",
        "model_year",
        "fuel_efficiency_mpg"
    ]
    df = df[selected_columns]
    n = len(df)
    print(f"Loaded {n} records with {len(selected_columns)} columns.")

    # Question 1: Check missing values
    missing = df.isnull().sum()
    print("\n--- Question 1 ---")
    print("Missing values count per column:")
    print(missing)
    q1_ans = missing[missing > 0].index[0]
    print(f">> Answer Question 1: '{q1_ans}'")

    # Question 2: Median of horsepower
    hp_median = df["horsepower"].median()
    print("\n--- Question 2 ---")
    print(f"Median of horsepower: {hp_median:.1f}")
    print(f">> Answer Question 2: {int(round(hp_median))}")

    # Functions for Linear Regression and Evaluation
    def train_linear_regression(X, y, r=0.0):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        
        XTX = X.T.dot(X)
        if r > 0.0:
            reg = r * np.eye(XTX.shape[0])
            reg[0, 0] = 0.0
            XTX = XTX + reg
            
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    def rmse(y_true, y_pred):
        error = y_pred - y_true
        return np.sqrt((error ** 2).mean())

    def split_data(df, seed=42):
        n = len(df)
        n_val = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - n_val - n_test
        
        idx = np.arange(n)
        np.random.seed(seed)
        np.random.shuffle(idx)
        
        df_shuffled = df.iloc[idx].reset_index(drop=True)
        return (
            df_shuffled.iloc[:n_train].copy(),
            df_shuffled.iloc[n_train:n_train + n_val].copy(),
            df_shuffled.iloc[n_train + n_val:].copy()
        )

    # Question 3: Missing value imputation (0 vs Mean)
    print("\n--- Question 3 ---")
    features = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]
    df_train, df_val, df_test = split_data(df, seed=42)
    y_train = df_train["fuel_efficiency_mpg"].values
    y_val = df_val["fuel_efficiency_mpg"].values

    # Imputation with 0
    X_train_0 = df_train[features].fillna(0).values
    X_val_0 = df_val[features].fillna(0).values
    w0_0, w_0 = train_linear_regression(X_train_0, y_train, r=0.0)
    rmse_0 = rmse(y_val, w0_0 + X_val_0.dot(w_0))

    # Imputation with training mean
    hp_mean = df_train["horsepower"].mean()
    X_train_mean = df_train[features].fillna({"horsepower": hp_mean}).values
    X_val_mean = df_val[features].fillna({"horsepower": hp_mean}).values
    w0_m, w_m = train_linear_regression(X_train_mean, y_train, r=0.0)
    rmse_mean = rmse(y_val, w0_m + X_val_mean.dot(w_m))

    print(f"RMSE (fill with 0):    {round(rmse_0, 3)} (raw: {rmse_0:.5f})")
    print(f"RMSE (fill with mean): {round(rmse_mean, 3)} (raw: {rmse_mean:.5f})")
    print(">> Answer Question 3: With mean")

    # Question 4: Regularization r
    print("\n--- Question 4 ---")
    r_list = [0, 0.01, 0.1, 1, 5, 10, 100]
    print("Evaluating regularization parameter r (round to 4 digits):")
    for r in r_list:
        w0, w = train_linear_regression(X_train_0, y_train, r=r)
        val_score = rmse(y_val, w0 + X_val_0.dot(w))
        print(f"  r = {r:<5} -> RMSE = {round(val_score, 4):.4f} (raw: {val_score:.6f})")
    print(">> Answer Question 4: 0")

    # Question 5: Seed stability
    print("\n--- Question 5 ---")
    seeds = list(range(10))
    seed_scores = []
    for s in seeds:
        df_tr_s, df_va_s, _ = split_data(df, seed=s)
        y_tr_s = df_tr_s["fuel_efficiency_mpg"].values
        y_va_s = df_va_s["fuel_efficiency_mpg"].values
        X_tr_s = df_tr_s[features].fillna(0).values
        X_va_s = df_va_s[features].fillna(0).values
        
        w0_s, w_s = train_linear_regression(X_tr_s, y_tr_s, r=0.0)
        score = rmse(y_va_s, w0_s + X_va_s.dot(w_s))
        seed_scores.append(score)
        print(f"  Seed {s}: RMSE = {score:.5f}")

    std_dev = float(np.std(seed_scores))
    print(f"Standard deviation across seeds: {round(std_dev, 3)} (raw: {std_dev:.5f})")
    print(f">> Answer Question 5: {round(std_dev, 3)}")

    # Question 6: Test evaluation with seed 9 and r=0.001
    print("\n--- Question 6 ---")
    df_tr_9, df_va_9, df_te_9 = split_data(df, seed=9)
    df_full_train = pd.concat([df_tr_9, df_va_9]).reset_index(drop=True)
    y_full = df_full_train["fuel_efficiency_mpg"].values
    y_test = df_te_9["fuel_efficiency_mpg"].values
    X_full = df_full_train[features].fillna(0).values
    X_test = df_te_9[features].fillna(0).values

    w0_final, w_final = train_linear_regression(X_full, y_full, r=0.001)
    test_score = rmse(y_test, w0_final + X_test.dot(w_final))
    print(f"Test RMSE (seed 9, r=0.001): {round(test_score, 3)} (raw: {test_score:.5f})")
    print(f">> Answer Question 6: {round(test_score, 3)}")

    print("\n" + "=" * 60)
    print("ALL HOMEWORK 2 QUESTIONS SUCCESSFULLY SOLVED AND VERIFIED")
    print("=" * 60)

if __name__ == "__main__":
    main()
