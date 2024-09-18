import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import src.utility.plots_cfg as plt_c
from src.utility.plots_save import export_figs

plt_c.load_cfg()


def load_and_preprocess_data():
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "../../data/raw/daily-bike-share.csv")
    df = pd.read_csv(data_path)

    categorical_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    feature_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]
    target = "rentals"

    X = df[feature_cols].copy()
    y = df[target]

    X[categorical_cols] = X[categorical_cols].astype("category")
    X = pd.get_dummies(X, columns=categorical_cols, dtype=int)

    return train_test_split(X, y, test_size=0.2, random_state=123)


def load_model():
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "../../models/bike-share.joblib")

    return load(model_path)


def evaluate_model(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R2 Score:", r2)

    figs = []  # List to store the figures

    # Scatter plot of actual vs predicted
    fig1, ax1 = plt.subplots()
    plt.scatter(y_test, y_pred)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
    figs.append((fig1, "Scatter_plot_actual_vs_predicted.png"))

    # Histogram of Residual
    fig2, ax2 = plt.subplots()
    residuals = y_test - y_pred
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.show()
    figs.append((fig2, "Histogram_Residual.png"))

    # Line plot of actual vs predicted over time
    fig3, ax2 = plt.subplots()
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted", color="green")
    plt.xlabel("Time")
    plt.ylabel("Bike Rentals")
    plt.legend()
    plt.show()
    figs.append((fig3, "Line_plot_actual_vs_predicted.png"))


def save_figs(figures):
    script_dir = os.path.dirname(__file__)
    export_dir: str = os.path.join(script_dir, "../reports/figures/")

    for index, (fig, filename) in enumerate(figures, start=1):
        export_figs(export_dir, fig, index, filename)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = load_model()

    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
