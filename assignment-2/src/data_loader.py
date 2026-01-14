"""
Data Loading and Preprocessing Module
Handles CSV loading, feature extraction, and train/val/test splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class DataLoader:
    """Data loader for phishing dataset"""
    
    def __init__(self, data_path):
        cleaned = str(data_path).strip().strip('"').strip("'")
        self.data_path = Path(cleaned)
        self.scaler = StandardScaler()
        
        # Feature categories
        self.text_features = ['URL', 'Domain', 'Title']
        self.identifier_features = ['FILENAME']
        raw_targets = "label, Result"
        self.possible_targets = [t.strip() for t in raw_targets.split(",") if t.strip()]
        self.target = None

    def detect_target_column(self, df):
        for col in self.possible_targets:
            if col in df.columns:
                self.target = col
                return
        raise ValueError(
            f"No valid target column found! Expected one of: {self.possible_targets}\n"
            f"Dataset columns: {list(df.columns)}"
        )

    def load_data(self):
        print(f"\n📂 Loading data from: {self.data_path}")
        clean_path = str(self.data_path).strip().strip('"').strip("'")
        self.data_path = Path(clean_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"   ✓ Loaded {len(df):,} samples")
        print(f"   ✓ Columns: {len(df.columns)}")
        self.detect_target_column(df)
        print(f"   ✓ Target column detected: {self.target}")
        return df

    def prepare_features(self, df):
        print(f"\n🔧 Preparing features...")

        for col in self.text_features:
            if col in df.columns:
                df[col] = df[col].fillna('')

        text_cols = [col for col in self.text_features if col in df.columns]
        if text_cols:
            df['combined_text'] = df[text_cols[0]].astype(str)
            for col in text_cols[1:]:
                df['combined_text'] += " [SEP] " + df[col].astype(str)
            X_text = df['combined_text'].values
            print(f"   ✓ Created combined text from {len(text_cols)} columns")
        else:
            X_text = np.array([])
            print(f"   ⚠️  No text columns found")

        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        to_exclude = [self.target] + self.identifier_features
        numerical_features = [c for c in numerical_features if c not in to_exclude]
        X_numerical = df[numerical_features].values
        X_numerical = np.nan_to_num(X_numerical, nan=0.0, posinf=999999, neginf=-999999)
        print(f"   ✓ Numerical features: {len(numerical_features)}")

        y = df[self.target].values
        print(f"   ✓ Target distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = "Legitimate" if label == 0 else "Phishing"
            print(f"      {label_name}: {count:,} ({count/len(y)*100:.1f}%)")

        self.numerical_feature_names = numerical_features
        return X_text, X_numerical, y

    def create_splits(self, X_text, X_numerical, y, test_size=0.15, val_size=0.15, random_state=42):
        print(f"\n✂️  Creating train/val/test splits...")

        if len(X_text) > 0:
            X_text_trainval, X_text_test, X_num_trainval, X_num_test, y_trainval, y_test = train_test_split(
                X_text, X_numerical, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            X_text_trainval, X_text_test = [], []
            X_num_trainval, X_num_test, y_trainval, y_test = train_test_split(
                X_numerical, y, test_size=test_size, random_state=random_state, stratify=y)

        val_adj = val_size / (1 - test_size)
        if len(X_text) > 0:
            X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
                X_text_trainval, X_num_trainval, y_trainval, test_size=val_adj, random_state=random_state, stratify=y_trainval)
        else:
            X_text_train, X_text_val = [], []
            X_num_train, X_num_val, y_train, y_val = train_test_split(
                X_num_trainval, y_trainval, test_size=val_adj, random_state=random_state, stratify=y_trainval)

        # Scale numeric
        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        X_num_val_scaled = self.scaler.transform(X_num_val)
        X_num_test_scaled = self.scaler.transform(X_num_test)

        # Convert to DataFrames to preserve column names
        X_train_num_df = pd.DataFrame(X_num_train_scaled, columns=self.numerical_feature_names)
        X_val_num_df = pd.DataFrame(X_num_val_scaled, columns=self.numerical_feature_names)
        X_test_num_df = pd.DataFrame(X_num_test_scaled, columns=self.numerical_feature_names)

        print(f"   ✓ Training:   {len(y_train):,} samples ({len(y_train)/len(y)*100:.1f}%)")
        print(f"   ✓ Validation: {len(y_val):,} samples ({len(y_val)/len(y)*100:.1f}%)")
        print(f"   ✓ Test:       {len(y_test):,} samples ({len(y_test)/len(y)*100:.1f}%)")

        return {
            'X_train_text': X_text_train,
            'X_val_text': X_text_val,
            'X_test_text': X_text_test,
            'X_train_num': X_num_train_scaled,
            'X_val_num': X_num_val_scaled,
            'X_test_num': X_num_test_scaled,
            'X_train_num_df': X_train_num_df,
            'X_val_num_df': X_val_num_df,
            'X_test_num_df': X_test_num_df,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': self.numerical_feature_names
        }
