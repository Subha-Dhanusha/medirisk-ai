import pandas as pd
import numpy as np
import joblib
import logging
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cleaned_data(filepath: str):
    logger.info(f"Loading cleaned data from: {filepath}")
    df = pd.read_csv(filepath)

    bool_cols = df.select_dtypes(include=['object']).columns
    for col in bool_cols:
        if df[col].dropna().isin(['True', 'False']).all():
            df[col] = df[col].map({'True': 1, 'False': 0})

    logger.info(f"Loaded shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    from src.features.engineer import create_features, select_features

    df = create_features(df)
    X, y = select_features(df)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    return X, y


def apply_smote(X_train, y_train):
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    logger.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info("Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\n AUC-ROC Score: {auc:.4f}")
    print(f"\n Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['Not Readmitted', 'Readmitted']))
    return auc, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Readmitted', 'Readmitted'],
                yticklabels=['Not Readmitted', 'Readmitted'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.show()
    logger.info("Confusion matrix saved to models/confusion_matrix.png")


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Readmission Prediction', fontsize=14)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=150)
    plt.show()
    logger.info("ROC curve saved to models/roc_curve.png")


def plot_feature_importance(model, X, top_n=20):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.nlargest(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    plt.title(f'Top {top_n} Most Important Features', fontsize=14)
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png', dpi=150)
    plt.show()
    logger.info("Feature importance saved to models/feature_importance.png")

    print(f"\nTop 10 Features:")
    for feat, score in top_features.head(10).items():
        print(f"  {feat}: {score:.4f}")


def run_cross_validation(model, X, y, cv=5):
    logger.info(f"Running {cv}-fold cross validation...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_strategy,
                              scoring='roc_auc', n_jobs=-1)
    print(f"\n Cross-Validation AUC Scores: {scores.round(4)}")
    print(f" Mean AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to: {filepath}")


if __name__ == "__main__":
    print("TRAINER STARTED")

    df = load_cleaned_data("data/processed/cleaned_data.csv")
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    model = train_xgboost(X_train_bal, y_train_bal)
    auc, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # Save model — skip plots and cross validation
    save_model(model, "models/xgboost_readmission.joblib")

    print("\n✅ Phase 4 Complete! Model trained and saved.")