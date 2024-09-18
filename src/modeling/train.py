import os

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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


def train_model(X_train, y_train):
    hyper_params = {
        "max_depth": 15,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "n_estimators": 200,
    }
    model = RandomForestRegressor(**hyper_params)
    model.fit(X_train, y_train)
    return model


def save_model(model):
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "../../models/bike-share.joblib")
    dump(model, model_path)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    save_model(model)
