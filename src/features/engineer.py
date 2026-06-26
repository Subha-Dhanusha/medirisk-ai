# src/features/engineer.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Medication features ---
    medication_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'insulin',
        'glyburide-metformin', 'glipizide-metformin'
    ]
    existing_med_cols = [c for c in medication_cols if c in df.columns]

    # Count active medications (non-zero = prescribed)
    df['total_medications'] = (df[existing_med_cols] != 0).sum(axis=1)

    # Count medication changes (Up=2 or Down=3)
    df['num_med_changes'] = (df[existing_med_cols] >= 2).sum(axis=1)

    # Insulin specifically prescribed
    if 'insulin' in df.columns:
        df['on_insulin'] = (df['insulin'] != 0).astype(int)
        df['insulin_changed'] = (df['insulin'] >= 2).astype(int)

    # --- Visit history features ---
    visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    existing_visit_cols = [c for c in visit_cols if c in df.columns]
    df['total_visits'] = df[existing_visit_cols].sum(axis=1)
    df['high_utilizer'] = (df['total_visits'] >= 3).astype(int)

    # Prior inpatient visits are strongest readmission predictor
    if 'number_inpatient' in df.columns:
        df['has_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
        df['multiple_inpatient'] = (df['number_inpatient'] > 1).astype(int)

    if 'number_emergency' in df.columns:
        df['has_emergency'] = (df['number_emergency'] > 0).astype(int)

    # --- Diagnosis features ---
    # After preprocessing, diag values are category codes 0-18
    # 4 = Diabetes, 1 = Circulatory, 2 = Respiratory
    if 'diag_1' in df.columns:
        df['diabetes_primary_diag'] = (df['diag_1'] == 4).astype(int)
        df['circulatory_primary_diag'] = (df['diag_1'] == 1).astype(int)

    if 'diag_2' in df.columns:
        df['diabetes_secondary_diag'] = (df['diag_2'] == 4).astype(int)

    # Any diabetes diagnosis across all three
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    existing_diag_cols = [c for c in diag_cols if c in df.columns]
    df['any_diabetes_diag'] = (
        df[existing_diag_cols].eq(4).any(axis=1)
    ).astype(int)

    df['any_circulatory_diag'] = (
        df[existing_diag_cols].eq(1).any(axis=1)
    ).astype(int)

    # Number of unique diagnosis categories
    df['num_diag_categories'] = df[existing_diag_cols].nunique(axis=1)

    # --- Hospital stay features ---
    if 'time_in_hospital' in df.columns:
        df['long_stay'] = (df['time_in_hospital'] >= 7).astype(int)
        df['very_long_stay'] = (df['time_in_hospital'] >= 10).astype(int)

    if 'num_procedures' in df.columns and 'time_in_hospital' in df.columns:
        df['procedures_per_day'] = (
            df['num_procedures'] / (df['time_in_hospital'] + 1)
        )

    if 'num_lab_procedures' in df.columns and 'time_in_hospital' in df.columns:
        df['lab_tests_per_day'] = (
            df['num_lab_procedures'] / (df['time_in_hospital'] + 1)
        )

    # High lab usage = more complex patient
    if 'num_lab_procedures' in df.columns:
        df['high_lab_use'] = (df['num_lab_procedures'] >= 50).astype(int)

    # --- Interaction features ---
    if 'change' in df.columns and 'diabetesMed' in df.columns:
        df['med_change_and_diabetic'] = (
            (df['change'] == 1) & (df['diabetesMed'] == 1)
        ).astype(int)

    if 'number_inpatient' in df.columns and 'time_in_hospital' in df.columns:
        df['inpatient_x_stay'] = (
            df['number_inpatient'] * df['time_in_hospital']
        )

    if 'total_medications' in df.columns and 'number_inpatient' in df.columns:
        df['meds_x_inpatient'] = (
            df['total_medications'] * df['number_inpatient']
        )

    # --- Age-based features ---
    if 'age' in df.columns:
        df['is_elderly'] = (df['age'] >= 70).astype(int)
        df['is_very_elderly'] = (df['age'] >= 80).astype(int)

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
    new_feats = [c for c in df_engineered.columns if c not in df.columns]
    print(f"New features added: {new_feats}")