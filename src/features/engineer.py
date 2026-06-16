import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature 1: Total Medications
    medication_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin'
    ]
    existing_med_cols = [c for c in medication_cols if c in df.columns]
    df['total_medications'] = (df[existing_med_cols] != 0).sum(axis=1)

    # Feature 2: Total Visits
    visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    existing_visit_cols = [c for c in visit_cols if c in df.columns]
    df['total_visits'] = df[existing_visit_cols].sum(axis=1)

    # Feature 3: High Utilizer
    df['high_utilizer'] = (df['total_visits'] >= 3).astype(int)

    # Feature 4: Number of Diagnoses
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    existing_diag_cols = [c for c in diag_cols if c in df.columns]
    df['num_diagnoses'] = df[existing_diag_cols].notna().sum(axis=1)

    # Feature 5: Diabetes Primary Diagnosis
    if 'diag_1' in df.columns:
        df['diabetes_primary_diag'] = df['diag_1'].astype(str).str.startswith('250').astype(int)

    # Feature 6: Medication Change + Diabetes Med
    if 'change' in df.columns and 'diabetesMed' in df.columns:
        df['med_change_and_diabetic'] = (
            (df['change'] == 1) & (df['diabetesMed'] == 1)
        ).astype(int)

    # Feature 7: Procedures Per Day
    if 'num_procedures' in df.columns and 'time_in_hospital' in df.columns:
        df['procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'] + 1)

    # Feature 8: Lab Tests Per Day
    if 'num_lab_procedures' in df.columns and 'time_in_hospital' in df.columns:
        df['lab_tests_per_day'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1)

    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df


def select_features(df: pd.DataFrame) -> tuple:
    target_col = 'readmitted'
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    return X, y


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_data.csv")
    df_engineered = create_features(df)
    X, y = select_features(df_engineered)
    print(f"Feature matrix shape: {X.shape}")
    print(f"New features: {[c for c in df_engineered.columns if c not in df.columns]}")