"""
Baseline Models Module
Traditional ML models for comparison
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

class BaselineModels:
    """Baseline ML models wrapper"""
    
    def __init__(self):
        """Initialize baseline models"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        
        self.results = {}
        
    def train_all(self, X_train, X_test, y_train, y_test):
        """
        Train all baseline models
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary of results
        """
        print(f"\n🔧 Training baseline models...")
        
        for name, model in self.models.items():
            print(f"\n   Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'predictions': y_pred,
                'true_labels': y_test,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            self.results[name] = metrics
            
            print(f"      ✓ Accuracy: {metrics['accuracy']:.4f}")
            print(f"      ✓ F1 Score: {metrics['f1']:.4f}")
        
        return self.results
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance from Random Forest
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features
            
        Returns:
            DataFrame with feature importance
        """
        if 'Random Forest' not in self.models:
            return None
        
        rf_model = self.models['Random Forest']
        
        if not hasattr(rf_model, 'feature_importances_'):
            return None
        
        import pandas as pd
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save(self, output_dir):
        """
        Save all models
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for name, model in self.models.items():
            model_file = output_path / f"{name.replace(' ', '_').lower()}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"   ✓ Baseline models saved to: {output_path}")