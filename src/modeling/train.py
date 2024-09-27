import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV


def load_and_preprocess_data():
    """
    Load the bike-share dataset, preprocess it, and split into train and test sets.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "../../data/raw/daily-bike-share.csv")
    df = pd.read_csv(data_path)

    # Define categorical and feature columns
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

    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[target]

    # Convert categorical columns to category type
    X[categorical_cols] = X[categorical_cols].astype("category")
    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, dtype=int)

    # Split data into train and test sets
    return train_test_split(X, y, test_size=0.2, random_state=123)


def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model using GridSearchCV for hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target

    Returns:
        RandomForestRegressor: Best model after hyperparameter tuning
    """
    # Define the parameter grid for Grid Search
    param_grid = {
        "max_depth": [10, 15, 20, 25, 30],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [100, 200, 300, 400],
    }

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(random_state=123)

    # Set up Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    # Fit Grid Search to the training data
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    return grid_search.best_estimator_


def save_model(model):
    """
    Save the trained model to a file.

    Args:
        model: Trained model to be saved
    """
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "../../models/bike-share.joblib")
    dump(model, model_path)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model
    save_model(model)
