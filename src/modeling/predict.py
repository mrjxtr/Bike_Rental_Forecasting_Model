import os
import pandas as pd
from joblib import load


def load_model():
    """
    Load the trained model from a file.

    Returns:
        object: Loaded model
    """
    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, "../../models/bike-share.joblib")
    return load(model_path)


def preprocess_input(input_data):
    """
    Preprocess the input data for prediction.

    Args:
        input_data (pd.DataFrame): Input data to be preprocessed

    Returns:
        pd.DataFrame: Preprocessed input data
    """
    categorical_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]

    # Convert categorical columns to category type
    input_data[categorical_cols] = input_data[categorical_cols].astype("category")
    # One-hot encode categorical columns
    input_data = pd.get_dummies(input_data, columns=categorical_cols, dtype=int)

    return input_data


def predict(model, input_data):
    """
    Make predictions using the trained model.

    Args:
        model: Trained model
        input_data (pd.DataFrame): Preprocessed input data

    Returns:
        np.array: Predicted values
    """
    return model.predict(input_data)


if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Create a sample input (you can modify this based on your needs)
    sample_input = pd.DataFrame(
        {
            "season": [1],
            "yr": [0],
            "mnth": [1],
            "holiday": [0],
            "weekday": [1],
            "workingday": [1],
            "weathersit": [1],
            "temp": [0.3],
            "atemp": [0.3],
            "hum": [0.5],
            "windspeed": [0.2],
        }
    )

    # Preprocess the input data
    processed_input = preprocess_input(sample_input)

    # Make a prediction
    prediction = predict(model, processed_input)

    print(f"Predicted number of bike rentals: {prediction[0]:.2f}")
