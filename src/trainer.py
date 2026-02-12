"""
GestuBot - Model Training Pipeline

Trains SVM classifier with GridSearchCV hyperparameter tuning.
Outputs trained model, confusion matrix, and classification report.

Usage:
    python trainer.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Import gesture class names for visualization
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import GESTURE_CLASSES


# --- Configuration ---

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATASET_PATH = os.path.join(DATA_DIR, 'gestures.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_svm.joblib')
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
HSV_CONFIG_PATH = os.path.join(MODEL_DIR, 'hsv_config.joblib')

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# RBF kernel works well for non-linear hand shape boundaries.
# C = regularization, gamma = kernel width.
PARAM_GRID = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
}


# --- Training Pipeline ---

def load_dataset() -> tuple:
    """
    Load and validate the gesture dataset.
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is label vector
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}\n"
            "Please run data_collector.py first to create training data."
        )
    
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values.astype(int)
    
    # Print class distribution
    print("\n=== Dataset Distribution ===")
    unique, counts = np.unique(y, return_counts=True)
    for cls_id, count in zip(unique, counts):
        cls_name = GESTURE_CLASSES.get(int(cls_id), "Unknown")
        pct = 100 * count / len(y)
        print(f"  Class {cls_id} ({cls_name}): {count} samples ({pct:.1f}%)")
    
    # Validate minimum samples per class
    min_samples = counts.min()
    if min_samples < 10:
        print(f"\n[WARNING] Some classes have very few samples (min: {min_samples})")
        print("Recommend at least 50 samples per class for reliable training.")
    
    return X, y


def create_model_pipeline() -> make_pipeline:
    """
    Create StandardScaler + SVC pipeline.

    StandardScaler is critical here because our features have wildly
    different scales: Hu Moments are ~10^-7, geometric ratios are 0-2,
    defect counts are 0-5. Without scaling, SVM ignores the small features.

    Pipeline ensures scaler fits only on training data (no leakage).
    """
    # RBF kernel SVM with StandardScaler preprocessing
    # Using make_pipeline for clean GridSearchCV integration
    model = make_pipeline(
        StandardScaler(),  # Critical: normalize feature scales
        SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    )
    
    return model


def train_with_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """
    Train model with hyperparameter optimization via GridSearchCV.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        
    Returns:
        Best estimator from grid search
    """
    print("\n=== Hyperparameter Tuning ===")
    print(f"Parameter grid: {PARAM_GRID}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    
    model = create_model_pipeline()
    
    # Stratified K-Fold ensures each fold has representative class distribution
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        model,
        PARAM_GRID,
        cv=cv,
        scoring='f1_macro',  # Macro-F1 balances all classes equally
        n_jobs=-1,           # Use all CPU cores
        verbose=1
    )
    
    print("\n[INFO] Starting grid search (this may take a minute)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n[RESULT] Best parameters: {grid_search.best_params_}")
    print(f"[RESULT] Best CV score (Macro-F1): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained sklearn model
        X_test: Test feature matrix
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n=== Model Evaluation ===")
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Test Macro-F1: {f1_macro:.4f}")
    
    # Target: >92% accuracy
    if accuracy >= 0.92:
        print("[SUCCESS] Target accuracy (>92%) achieved!")
    else:
        print(f"[INFO] Below target accuracy. Consider collecting more samples.")
    
    # Detailed classification report
    # Get actual classes present in the data
    all_classes = sorted(set(y_test) | set(y_pred))
    target_names = [GESTURE_CLASSES.get(int(c), f"Class {c}") for c in all_classes]
    
    print("\n=== Classification Report ===")
    report = classification_report(
        y_test, y_pred,
        labels=all_classes,
        target_names=target_names,
        digits=3
    )
    print(report)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'y_pred': y_pred,
        'y_test': y_test
    }


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Generate and save confusion matrix heatmap.
    Good for spotting which gestures get confused with each other.
    """
    # Get classes present in the data
    all_classes = sorted(set(y_test) | set(y_pred))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Get class names for labels
    class_names = [GESTURE_CLASSES.get(int(c), f"Class {c}") for c in all_classes]
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('GestuBot Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    
    print(f"\n[SAVED] Confusion matrix: {CONFUSION_MATRIX_PATH}")


def save_model(model) -> None:
    """Save trained model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[SAVED] Model: {MODEL_PATH}")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("GestuBot Model Training")
    print("="*60)
    
    # Load data
    X, y = load_dataset()
    
    # Train/test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n[INFO] Train set: {len(X_train)} samples")
    print(f"[INFO] Test set: {len(X_test)} samples")
    
    # Train with hyperparameter optimization
    best_model = train_with_grid_search(X_train, y_train)
    
    # Evaluate on test set
    results = evaluate_model(best_model, X_test, y_test)
    
    # Generate confusion matrix
    plot_confusion_matrix(results['y_test'], results['y_pred'])
    
    # Save model
    save_model(best_model)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {results['accuracy']*100:.1f}%")
    print(f"Final Test Macro-F1: {results['f1_macro']:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Run 'python inference.py' to start real-time gesture control.\n")


if __name__ == '__main__':
    main()
