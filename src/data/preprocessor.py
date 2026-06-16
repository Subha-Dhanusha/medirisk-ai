import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    original_size = len(df)
    df = df.drop_duplicates(subset=['patient_nbr'], keep='first')
    invalid_discharge = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(invalid_discharge)]
    logger.info(f"Removed {original_size - len(df)} invalid rows")
    return df


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    logger.info(f"Dropping high-missing columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        'encounter_id', 'patient_nbr', 'examide', 'citoglipton',
        'weight', 'payer_code', 'medical_specialty',
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped useless columns: {cols_to_drop}")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df['readmitted'] = df['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    logger.info(f"Target distribution:\n{df['readmitted'].value_counts()}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    logger.info("Missing values handled")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    binary_map = {
        'gender': {'Male': 1, 'Female': 0, 'Unknown/Invalid': 0},
        'change': {'Ch': 1, 'No': 0},
        'diabetesMed': {'Yes': 1, 'No': 0},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    age_mapping = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95,
    }
    if 'age' in df.columns:
        df['age'] = df['age'].map(age_mapping)

    remaining_cats = df.select_dtypes(include=['object']).columns.tolist()
    remaining_cats = [c for c in remaining_cats if c != 'readmitted']
    if remaining_cats:
        df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)
        logger.info(f"One-hot encoded: {remaining_cats}")

    return df


def run_preprocessing_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    from src.data.loader import load_raw_data

    logger.info("Starting preprocessing pipeline...")
    df = load_raw_data(input_path)
    df = remove_invalid_rows(df)
    df = drop_high_missing_columns(df, threshold=0.4)
    df = drop_useless_columns(df)
    df = create_binary_target(df)
    df = handle_missing_values(df)
    df = encode_categorical_columns(df)

    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to: {output_path}")
    logger.info(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = run_preprocessing_pipeline(
        input_path="data/raw/diabetic_data.csv",
        output_path="data/processed/cleaned_data.csv"
    )
    print(df.head())
    print(df.shape)