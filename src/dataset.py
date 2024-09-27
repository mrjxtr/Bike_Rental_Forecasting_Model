import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load the dataset from the given file path."""
    return pd.read_csv(file_path)


def data_overview(df):
    """Provide an overview of the dataset including data types, missing values, and visualizations."""
    # Display the first few rows
    print("First few rows of the dataset:----------")
    print(df.head())

    # Data types for each column
    print("\nData types for each column:----------")
    print(df.info())

    # Display the shape of the dataset
    print("\nShape of the dataset:----------")
    print(df.shape)

    # Summary statistics for numerical columns
    print("\nSummary statistics for numerical columns:----------")
    print(df.describe())

    # Check for missing values
    print("\nMissing values in each column:----------")
    print(df.isnull().sum())

    # Check unique values per column
    print("\nUnique values per column:----------")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    # Correlation matrix (excluding non-numeric columns)
    print("\nCorrelation matrix:----------")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    print(df[numeric_cols].corr())

    # Plot histograms for numerical features
    print("\nDisplaying histograms for numerical features...")
    df[numeric_cols].hist(figsize=(12, 10), bins=30)
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

    # Convert categorical columns to 'category' dtype
    categorical_cols = [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    df[categorical_cols] = df[categorical_cols].astype("category")

    # Display value counts for categorical columns
    print("\nValue counts for categorical columns:----------")
    categorical_cols = df.select_dtypes(include=["category"]).columns
    for col in categorical_cols:
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print()


if __name__ == "__main__":
    # Define paths and load data
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "../data/raw/daily-bike-share.csv")

    df = load_data(data_path)
    print("############################")
    print("# Data loaded successfully #")
    print("############################")

    data_overview(df)
    print("##########################")
    print("# Data overview complete #")
    print("##########################")
