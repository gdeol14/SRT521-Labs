# models/evaluator.py
# Model evaluation module
# Student: Gurmandeep Deol | ID: 104120233

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve, auc
)
import warnings

# ============================================================================
# PREDICTION GENERATION
# ============================================================================
def predict_all_models(models, X_test, X_test_scaled):
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'SVM']:
            predictions[name] = model.predict(X_test_scaled)
            
            # Handle probability prediction carefully
            if hasattr(model, 'predict_proba'):
                probabilities[name] = model.predict_proba(X_test_scaled)
            else:
                # SVM with probability=True should have this, but handle edge case
                probabilities[name] = None
        else:
            predictions[name] = model.predict(X_test)
            probabilities[name] = model.predict_proba(X_test)
    
    return predictions, probabilities

# ============================================================================
# PERFORMANCE METRICS CALCULATION
# ============================================================================
def calculate_metrics(y_test, predictions):
    """
    Calculate comprehensive performance metrics
    
    Handles both binary and multi-class classification
    """
    results = []
    
    # Determine if binary or multi-class
    n_classes = len(np.unique(y_test))
    is_binary = (n_classes == 2)
    
    for name, y_pred in predictions.items():
        # Choose averaging strategy based on problem type
        if is_binary:
            # Binary: use 'binary' for precision/recall/f1
            avg_strategy = 'binary'
        else:
            # Multi-class: use 'weighted' (accounts for class imbalance)
            avg_strategy = 'weighted'
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average=avg_strategy, zero_division=0),
            'Recall': recall_score(y_test, y_pred, average=avg_strategy, zero_division=0),
            'F1': f1_score(y_test, y_pred, average=avg_strategy, zero_division=0)
        }
        results.append(metrics)
    
    df = pd.DataFrame(results).round(4)
    return df

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
def analyze_errors(y_test, predictions):
    """
    Analyze prediction errors
    
    For binary: FP and FN
    For multi-class: Total misclassifications
    """
    error_analysis = {}
    n_classes = len(np.unique(y_test))
    is_binary = (n_classes == 2)
    
    for name, y_pred in predictions.items():
        if is_binary:
            # Binary classification: FP and FN
            # Assume positive class is 1 (after encoding)
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            error_analysis[name] = {
                'FP': fp,
                'FN': fn,
                'Total': fp + fn
            }
        else:
            # Multi-class: just total errors
            total_errors = np.sum(y_test != y_pred)
            
            error_analysis[name] = {
                'Total': total_errors,
                'Error_Rate': total_errors / len(y_test)
            }
    
    return error_analysis

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================
def evaluate_models(models, X_test, y_test, X_test_scaled):
    """
    Complete evaluation pipeline for all models
    
    Handles both binary and multi-class classification
    """
    print("="*70)
    print("MODEL EVALUATION".center(70))
    print("="*70)
    
    # Determine problem type
    n_classes = len(np.unique(y_test))
    is_binary = (n_classes == 2)
    print(f"\n📊 Classification type: {'Binary' if is_binary else f'Multi-class ({n_classes} classes)'}")
    
    # === STEP 1: Generate predictions ===
    print("\n📊 Generating predictions...")
    predictions, probabilities = predict_all_models(models, X_test, X_test_scaled)
    
    # === STEP 2: Calculate metrics ===
    print("\n📈 Calculating metrics...")
    metrics_df = calculate_metrics(y_test, predictions)
    print("\n" + metrics_df.to_string(index=False))
    
    # === STEP 3: Identify best model ===
    best_model = metrics_df.loc[metrics_df['F1'].idxmax(), 'Model']
    best_f1 = metrics_df.loc[metrics_df['F1'].idxmax(), 'F1']
    print(f"\n🏆 Best Model: {best_model} (F1={best_f1:.4f})")
    
    # === STEP 4: Error analysis ===
    print("\n🔍 Error Analysis:")
    errors = analyze_errors(y_test, predictions)
    
    # Show first 2 models' errors
    for name, err in list(errors.items())[:2]:
        if is_binary:
            print(f"   {name}: FP={err['FP']}, FN={err['FN']}, Total={err['Total']}")
        else:
            print(f"   {name}: Errors={err['Total']} ({err['Error_Rate']:.1%})")
    
    print("\n✅ Evaluation complete!")
    print("="*70)
    
    return predictions, probabilities, metrics_df

# ============================================================================
# CONFUSION MATRIX HELPER
# ============================================================================
def get_confusion_matrices(y_test, predictions):
    """Generate confusion matrices for all models"""
    cms = {}
    for name, y_pred in predictions.items():
        cms[name] = confusion_matrix(y_test, y_pred)
    return cms

# ============================================================================
# ROC CURVE DATA HELPER (BINARY ONLY)
# ============================================================================
def get_roc_data(y_test, probabilities):
    """
    Generate ROC curve data for binary classification
    
    Returns None for multi-class (ROC requires binary)
    """
    n_classes = len(np.unique(y_test))
    if n_classes != 2:
        warnings.warn("ROC curves only available for binary classification")
        return None
    
    roc_data = {}
    
    for name, y_prob in probabilities.items():
        if y_prob is None:
            continue
        
        # For binary classification, use probability of positive class
        if len(y_prob.shape) == 2 and y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]  # Probability of class 1
        else:
            y_prob_pos = y_prob  # Already 1D array
        
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
            roc_auc = auc(fpr, tpr)
            
            roc_data[name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        except Exception as e:
            warnings.warn(f"Could not compute ROC for {name}: {e}")
            continue
    
    return roc_data