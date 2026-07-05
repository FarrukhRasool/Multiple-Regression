"""
evaluate_framework.py — sanity-check the Task 2 rule engine on labeled data.

Runs the `(condition, calculation)` rules from `framework.py` over the training
dataset and reports how well they reproduce the true `target02`, using MSE and R².
Handy for confirming the extracted rules generalise before trusting them.

Usage
-----
    python evaluate_framework.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from framework import build_rules, framework

DATASET_PATH = "problem_41/dataset_41.csv"
TARGET_PATH = "problem_41/target_41.csv"


def main():
    features = pd.read_csv(DATASET_PATH).values
    y_pred = np.array(framework(build_rules(), features))
    y_true = pd.read_csv(TARGET_PATH)["target02"].values

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Framework Mean Squared Error: {mse:.4f}")
    print(f"Framework R2 Score:           {r2:.4f}")

    return y_true, y_pred


def plot(y_true, y_pred):
    import matplotlib.pyplot as plt

    plt.scatter(y_true, y_pred, s=10, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("True target02")
    plt.ylabel("Framework prediction")
    plt.title("Framework predictions vs. true values")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    y_true, y_pred = main()
    plot(y_true, y_pred)