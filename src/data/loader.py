import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, na_values=['?'])
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def get_basic_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET BASIC INFORMATION")
    print("=" * 60)
    print(f"\n Shape: {df.shape}")
    print(f"\n Columns:\n{df.columns.tolist()}")
    print(f"\n Data Types:\n{df.dtypes}")
    print(f"\n Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n Target Variable Distribution:")
    print(df['readmitted'].value_counts())
    print(df['readmitted'].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    df = load_raw_data("data/raw/diabetic_data.csv")
    get_basic_info(df)