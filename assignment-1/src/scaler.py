# preprocessing/scaler.py
# Feature scaling module (IMPROVED with error handling)
# Student: Gurmandeep Deol | ID: 104120233

from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================================
# FEATURE SCALING (IMPROVED - Works with all datasets)
# ============================================================================
def scale_features(X_train, X_test):
    """
    StandardScaler: transforms features to mean=0, std=1
    Formula: z = (x - mean) / std
    
    IMPROVEMENTS:
    - Handles constant features (std=0)
    - Validates input data
    - Works with numpy arrays and pandas DataFrames
    """
    print("\n📏 Scaling features...")
    
    # === VALIDATION: Check for issues ===
    if X_train.shape[0] == 0:
        raise ValueError("Training data is empty!")
    
    if X_train.shape[1] == 0:
        raise ValueError("No features to scale!")
    
    # Check for infinite values
    if np.isinf(X_train).any().any() if hasattr(X_train, 'any') else np.isinf(X_train).any():
        print("   ⚠️ Warning: Infinite values detected, replacing with NaN")
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values
    if np.isnan(X_train).any().any() if hasattr(X_train, 'any') else np.isnan(X_train).any():
        print("   ⚠️ Warning: NaN values detected, filling with 0")
        X_train = X_train.fillna(0) if hasattr(X_train, 'fillna') else np.nan_to_num(X_train)
        X_test = X_test.fillna(0) if hasattr(X_test, 'fillna') else np.nan_to_num(X_test)
    
    # === STEP 1: Initialize StandardScaler ===
    scaler = StandardScaler()
    
    # === STEP 2: Fit on training data only ===
    # Learn mean and std from training data
    # Prevents test set information leakage
    try:
        scaler.fit(X_train)
    except Exception as e:
        print(f"   ❌ Error during scaler fitting: {e}")
        raise
    
    # === STEP 3: Transform both train and test ===
    # Apply same transformation to both sets
    try:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"   ❌ Error during scaling: {e}")
        raise
    
    # === STEP 4: Verification ===
    train_mean = X_train_scaled.mean()
    train_std = X_train_scaled.std()
    
    print(f"✅ Features scaled: mean={train_mean:.4f}, std={train_std:.4f}")
    
    # Check for constant features (std=0)
    if hasattr(scaler, 'scale_'):
        constant_features = np.sum(scaler.scale_ == 0)
        if constant_features > 0:
            print(f"   ⚠️ Warning: {constant_features} constant features detected")
            print(f"      (These have zero variance and won't help models)")
    
    print(f"   Training samples: {X_train_scaled.shape[0]:,}")
    print(f"   Test samples: {X_test_scaled.shape[0]:,}")
    print(f"   Features: {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, scaler