# data/loader.py
# Data loading with robust encoding detection
# Student: Gurmandeep Deol | ID: 104120233

import pandas as pd
import os

# ============================================================================
# TARGET COLUMN AUTO-DETECTION
# ============================================================================
def detect_target_column(df):
    candidates = [
        'label', 'target', 'class', 'y', 'output', 'result',
        'is_fraud', 'is_phishing', 'attack_type', 'malware_family',
        'class_label', 'status', 'phishing', 'label_phishing'
    ]
    
    # Exact match (case-insensitive)
    for col in df.columns:
        if col.lower() in candidates:
            return col
    
    # Partial match
    for col in df.columns:
        for cand in candidates:
            if cand in col.lower():
                return col
    
    # Binary column heuristic
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    
    # Small number of classes
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 30:
            return col
    
    return None

# ============================================================================
# DATASET LOADING (ROBUST ENCODING - EXPANDED)
# ============================================================================
def load_dataset(path):
    """
    Load CSV with automatic encoding detection
    
    EXPANDED: Now tries 8 different encodings including exotic ones
    """
    original_path = path
    path = os.path.normpath(path.strip().strip('"').strip("'"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    print(f"📥 Loading dataset from: {path}")
    
    # Extended list of encodings to try
    encodings = [
        'utf-8',           # Standard
        'utf-8-sig',       # UTF-8 with BOM (Excel exports)
        'latin-1',         # Windows Western Europe
        'iso-8859-1',      # Similar to latin-1
        'cp1252',          # Windows Western
        'cp1250',          # Windows Central Europe
        'cp850',           # DOS Western Europe
        'ascii'            # Basic ASCII
    ]
    
    df = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, on_bad_lines='skip')
            used_encoding = encoding
            
            # Verify data loaded correctly
            if df.shape[0] == 0:
                print(f"   ⚠️ {encoding}: File loaded but empty, trying next...")
                df = None
                continue
            
            if df.shape[1] < 2:
                print(f"   ⚠️ {encoding}: Too few columns, trying next...")
                df = None
                continue
            
            # Success!
            if encoding != 'utf-8':
                print(f"   ⚠️ Used {encoding} encoding (file not UTF-8)")
            break
            
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as e:
            print(f"   ⚠️ {encoding}: Parser error, trying next...")
            continue
        except Exception as e:
            if encoding == encodings[-1]:
                raise e
            continue
    
    if df is None:
        error_msg = f"""
❌ Could not load file with any encoding!

Tried: {', '.join(encodings)}

Possible solutions:
1. Open file in Excel/LibreOffice and Save As → UTF-8 CSV
2. Use command line: iconv -f WINDOWS-1252 -t UTF-8 input.csv > output.csv
3. Check if file is corrupted or not a CSV
"""
        raise ValueError(error_msg)
    
    # Display loading success
    print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    
    return df

# ============================================================================
# DATASET VALIDATION
# ============================================================================
def validate_dataset(df, target_col):
    # Check 1: Target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Check 2: Display target info
    print(f"\n🎯 Target column: {target_col}")
    
    n_classes = df[target_col].nunique()
    print(f"📊 Number of classes: {n_classes}")
    
    if n_classes == 2:
        print(f"   Classification type: Binary")
    elif n_classes <= 10:
        print(f"   Classification type: Multi-class ({n_classes} classes)")
    else:
        print(f"   Classification type: Multi-class ({n_classes} classes)")
        print(f"   ⚠️ Large number of classes may affect performance")
    
    # Check 3: Class distribution
    print(f"\n📊 Class distribution:")
    value_counts = df[target_col].value_counts().sort_index()
    
    if n_classes <= 20:
        for val, count in value_counts.items():
            percentage = count/len(df)*100
            print(f"   Class {val}: {count:,} ({percentage:.1f}%)")
    else:
        print("   Top 10 classes:")
        for val, count in value_counts.head(10).items():
            percentage = count/len(df)*100
            print(f"   Class {val}: {count:,} ({percentage:.1f}%)")
        print(f"   ... ({n_classes - 20} more classes) ...")
        print("   Bottom 10 classes:")
        for val, count in value_counts.tail(10).items():
            percentage = count/len(df)*100
            print(f"   Class {val}: {count:,} ({percentage:.1f}%)")
    
    # Check 4: Class imbalance
    if n_classes == 2:
        majority_pct = value_counts.max() / len(df) * 100
        if majority_pct > 90:
            print(f"\n⚠️ SEVERE CLASS IMBALANCE: {majority_pct:.1f}% in majority class")
            print("   Consider using SMOTE, class weights, or resampling")
        elif majority_pct > 70:
            print(f"\n⚠️ Moderate class imbalance: {majority_pct:.1f}% in majority class")
    
    # Check 5: Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"⚠️ Missing values: {missing:,}")
    
    # Check 6: Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"⚠️ Duplicate rows: {dupes:,}")
    
    return True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_available_csvs():
    """List all CSV files in current directory"""
    return [f for f in os.listdir('.') if f.endswith('.csv')]