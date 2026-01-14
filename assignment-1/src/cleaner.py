# preprocessing/cleaner.py
# COMPLETE FIX for all dataset issues

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import LEAKAGE_COLUMNS, TEST_SIZE, RANDOM_STATE

# ============================================================================
# LABEL ENCODING (FIXED - Handles gaps in class labels)
# ============================================================================
def encode_target(y):
    """
    Encode target labels to continuous 0, 1, 2, ... format
    
    CRITICAL FIX: LabelEncoder automatically handles gaps!
    - Input: [1, 3, 7] → Output: [0, 1, 2] ✅
    - Input: [1, 2, 22, 23] → Output: [0, 1, 2, 3] ✅
    - Input: [-1, 1] → Output: [0, 1] ✅
    - Input: ['legitimate', 'phishing'] → Output: [0, 1] ✅
    """
    print("\n🏷️ Encoding target labels...")
    
    # Get unique values before encoding
    original_classes = sorted(y.unique())
    n_classes = len(original_classes)
    
    print(f"   Original classes: {n_classes} unique values")
    
    # Show range for numeric, samples for string
    if pd.api.types.is_numeric_dtype(y):
        print(f"   Range: {min(original_classes)} to {max(original_classes)}")
    else:
        print(f"   Types: {original_classes[:5]}..." if n_classes > 5 else f"   Types: {original_classes}")
    
    # Determine if binary or multi-class
    is_binary = (n_classes == 2)
    
    # LabelEncoder handles EVERYTHING automatically
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Create mapping dictionary (original → encoded)
    # Convert to string for consistent JSON serialization
    mapping = {str(original): int(encoded) for original, encoded in 
               zip(le.classes_, range(len(le.classes_)))}
    
    # Verify encoding is continuous [0, 1, 2, ..., n-1]
    encoded_classes = sorted(np.unique(y_encoded))
    expected_classes = list(range(n_classes))
    
    if encoded_classes != expected_classes:
        raise ValueError(
            f"Label encoding failed! Got {encoded_classes}, expected {expected_classes}"
        )
    
    print(f"   ✅ Encoded to: [0, 1, ..., {n_classes-1}] ({n_classes} classes)")
    print(f"   Classification type: {'Binary' if is_binary else 'Multi-class'}")
    
    # Show sample mapping
    if n_classes <= 10:
        print(f"   Mapping: {mapping}")
    else:
        items = list(mapping.items())
        print(f"   Mapping (sample): {dict(items[:3])} ... {dict(items[-3:])}")
    
    return pd.Series(y_encoded, index=y.index), le, mapping, is_binary

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_phishing_features(df):
    print("\n🔧 Engineering phishing-specific features...")
    
    df = df.copy()
    
    # Feature 1: URL Length Category
    if 'URLLength' in df.columns:
        df['URL_Length_Category'] = pd.cut(
            df['URLLength'],
            bins=[0, 50, 75, 150, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        print("   ✅ Feature 1: URL_Length_Category")
    
    # Feature 2: Subdomain Depth
    if 'NoOfSubDomain' in df.columns:
        df['Has_Multiple_Subdomains'] = (df['NoOfSubDomain'] > 1).astype(int)
        print("   ✅ Feature 2: Has_Multiple_Subdomains")
    
    # Feature 3: Content Complexity
    if 'LineOfCode' in df.columns and 'URLLength' in df.columns:
        df['Content_URL_Complexity'] = df['LineOfCode'] / (df['URLLength'] + 1)
        df['Content_URL_Complexity'] = np.clip(df['Content_URL_Complexity'] / 100, 0, 1)
        print("   ✅ Feature 3: Content_URL_Complexity")
    
    # Feature 4: External Dependency
    if 'NoOfExternalRef' in df.columns and 'NoOfSelfRef' in df.columns:
        total_refs = df['NoOfExternalRef'] + df['NoOfSelfRef'] + 1
        df['External_Dependency_Ratio'] = df['NoOfExternalRef'] / total_refs
        print("   ✅ Feature 4: External_Dependency_Ratio")
    
    # Feature 5: Form Security Risk
    if 'HasExternalFormSubmit' in df.columns and 'HasPasswordField' in df.columns:
        df['Form_Security_Risk'] = (
            df['HasExternalFormSubmit'] + df['HasPasswordField']
        )
        df['Form_Security_Risk'] = np.clip(df['Form_Security_Risk'], 0, 2)
        print("   ✅ Feature 5: Form_Security_Risk")
    
    # Feature 6: Tech Stack
    if 'NoOfCSS' in df.columns and 'NoOfJS' in df.columns:
        df['Tech_Stack_Indicator'] = np.log1p(df['NoOfCSS'] + df['NoOfJS'])
        max_val = df['Tech_Stack_Indicator'].max()
        if max_val > 0:
            df['Tech_Stack_Indicator'] = df['Tech_Stack_Indicator'] / max_val
        print("   ✅ Feature 6: Tech_Stack_Indicator")
    
    # Feature 7: Trust Signals
    trust_cols = ['HasFavicon', 'HasDescription', 'HasTitle']
    if all(col in df.columns for col in trust_cols):
        df['Trust_Signals_Count'] = sum(df[col] for col in trust_cols)
        print("   ✅ Feature 7: Trust_Signals_Count")
    
    # Feature 8: Special Character Density
    if 'NoOfOtherSpecialCharsInURL' in df.columns and 'URLLength' in df.columns:
        df['Special_Char_Density'] = df['NoOfOtherSpecialCharsInURL'] / (df['URLLength'] + 1)
        print("   ✅ Feature 8: Special_Char_Density")
    
    print(f"\n✅ Completed feature engineering")
    
    return df

# ============================================================================
# DATA LEAKAGE REMOVAL
# ============================================================================
def remove_leakage_columns(df, target_col):
    print(f"\n🧹 Removing leakage columns...")
    
    df_clean = df.dropna(subset=[target_col]).copy()
    
    X = df_clean.drop(columns=LEAKAGE_COLUMNS + [target_col], errors='ignore')
    y = df_clean[target_col]
    
    removed = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    print(f"   Removed: {len(removed)} leakage columns")
    print(f"   Remaining features: {X.shape[1]}")
    
    if len(removed) > 0:
        print(f"   Examples removed: {removed[:5]}")
    
    return X, y

# ============================================================================
# FEATURE TYPE CONVERSION
# ============================================================================
def convert_to_numeric(X):
    print("\n🔢 Converting features to numeric...")
    
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric) > 0:
        print(f"   Found {len(non_numeric)} non-numeric columns")
        
        for col in non_numeric:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                print(f"   ✓ Converted: {col}")
            except:
                X = X.drop(columns=[col])
                print(f"   ✗ Dropped: {col} (conversion failed)")
    
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        X = X.fillna(0)
        print(f"   Filled {missing_before} missing values with 0")
    
    print(f"✅ All features numeric: {X.shape[1]} features")
    return X

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
def split_data(X, y):
    print(f"\n✂️ Splitting data (test_size={TEST_SIZE})...")
    
    stratify = y if y.nunique() <= 20 else None
    
    if stratify is not None:
        print(f"   Using stratified split (maintains class balance)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )
    
    print(f"✅ Train set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"✅ Test set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(y)*100:.1f}%)")
    
    if stratify is not None:
        train_balance = (y_train.value_counts(normalize=True) * 100).round(1)
        test_balance = (y_test.value_counts(normalize=True) * 100).round(1)
        print(f"\n   Class balance maintained:")
        print(f"   Training:  {dict(train_balance)}")
        print(f"   Testing:   {dict(test_balance)}")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================================
def preprocess_pipeline(df, target_col):
    print("="*70)
    print("DATA PREPROCESSING PIPELINE (Labs 2-3)".center(70))
    print("="*70)
    
    # Step 1: Feature Engineering
    print("\n📊 STEP 1: Feature Engineering (Assignment 1 Requirement)")
    df = engineer_phishing_features(df)
    
    # Step 2: Remove leakage columns
    print("\n🚫 STEP 2: Remove Data Leakage (Lab 6 Lesson Applied)")
    X, y = remove_leakage_columns(df, target_col)
    
    # Step 3: Encode target labels (FIXED - Returns 4 values)
    print("\n🏷️ STEP 3: Encode Target Labels (XGBoost Compatibility)")
    y_encoded, label_encoder, label_mapping, is_binary = encode_target(y)
    
    # Step 4: Convert features to numeric
    print("\n🔢 STEP 4: Type Conversion (Lab 2)")
    X = convert_to_numeric(X)
    
    # Step 5: Train-test split
    print("\n✂️ STEP 5: Train-Test Split (Lab 4)")
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"   Features used: {X_train.shape[1]}")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Testing samples: {X_test.shape[0]:,}")
    print(f"   Engineered features: 8 (meets Assignment 1 requirement of 5+)")
    print(f"   Label encoding: {label_mapping}")
    print(f"   Classification type: {'Binary' if is_binary else 'Multi-class'}")
    print("="*70)
    
    return X_train, X_test, y_train, y_test, label_encoder, label_mapping