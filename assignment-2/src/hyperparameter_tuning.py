"""
Hyperparameter Tuning Module
Grid search and random search for model optimization
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from itertools import product
import time

class HyperparameterTuner:
    """Hyperparameter tuning for all models"""
    
    def __init__(self, output_dir='results'):
        """
        Initialize hyperparameter tuner
        
        Args:
            output_dir: Directory to save tuning results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = {}
    
    def tune_random_forest(self, X_train, y_train, cv=3):
        """
        Tune Random Forest hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        print(f"\n🔧 Tuning Random Forest hyperparameters...")
        
        # Optimized parameter grid - still comprehensive but faster
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"   Testing {total_combinations} combinations...")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        results = {
            'model': 'Random Forest',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'training_time': training_time,
            'n_combinations_tested': total_combinations,
            'cv_results': {
                'mean_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_scores': grid_search.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }
        
        self.results['Random Forest'] = results
        
        print(f"   ✓ Best parameters: {grid_search.best_params_}")
        print(f"   ✓ Best F1 score: {grid_search.best_score_:.4f}")
        print(f"   ✓ Tuning time: {training_time/60:.2f} minutes")
        
        return results
    
    def tune_xgboost(self, X_train, y_train, cv=3):
        """
        Tune XGBoost hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        print(f"\n🔧 Tuning XGBoost hyperparameters...")
        
        # Optimized parameter grid for GPU
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"   Testing {total_combinations} combinations...")
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',  # Use 'hist' for CPU or auto-detect GPU
            device='cuda:0'  # GPU acceleration (XGBoost 3.1+)
        )
        
        # Use GridSearchCV for thorough search with GPU
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv,
            scoring='f1', n_jobs=1, verbose=1  # n_jobs=1 for GPU
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        results = {
            'model': 'XGBoost',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'training_time': training_time,
            'n_combinations_tested': total_combinations,
            'cv_results': {
                'mean_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_scores': grid_search.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }
        
        self.results['XGBoost'] = results
        
        print(f"   ✓ Best parameters: {grid_search.best_params_}")
        print(f"   ✓ Best F1 score: {grid_search.best_score_:.4f}")
        print(f"   ✓ Tuning time: {training_time/60:.2f} minutes")
        
        return results
    
    def tune_tabtransformer(self, X_train, X_val, y_train, y_val):
        """
        Tune TabTransformer hyperparameters
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            
        Returns:
            Best parameters and scores
        """
        print(f"\n🔧 Tuning TabTransformer hyperparameters...")
        
        from src.tabtransformer import TabTransformer
        from sklearn.metrics import f1_score
        
        # Focused parameter grid for GPU training
        param_grid = {
            'd_model': [64, 128],  # Reduced from [32, 64, 128]
            'nhead': [4, 8],  # Reduced from [2, 4, 8]
            'num_layers': [2, 3],  # Reduced from [1, 2, 3]
            'learning_rate': [0.001, 0.01]  # Reduced from [0.0001, 0.001, 0.01]
        }
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"   Testing {total_combinations} combinations (GPU accelerated)...")
        
        best_score = 0
        best_params = {}
        all_results = []
        
        # Manual grid search (since TabTransformer doesn't support sklearn API)
        current = 0
        
        for d_model in param_grid['d_model']:
            for nhead in param_grid['nhead']:
                for num_layers in param_grid['num_layers']:
                    for lr in param_grid['learning_rate']:
                        current += 1
                        print(f"\n   Testing combination {current}/{total_combinations}...")
                        print(f"   d_model={d_model}, nhead={nhead}, layers={num_layers}, lr={lr}")
                        
                        try:
                            # Create and train model
                            model = TabTransformer(
                                input_dim=X_train.shape[1],
                                d_model=d_model,
                                nhead=nhead,
                                num_layers=num_layers
                            )
                            
                            start_time = time.time()
                            history = model.train(
                                X_train, X_val, X_val,  # Use val as test for tuning
                                y_train, y_val, y_val,
                                epochs=15,  # Moderate epochs for tuning
                                learning_rate=lr,
                                batch_size=128
                            )
                            training_time = time.time() - start_time
                            
                            # Evaluate
                            predictions = model.predict(X_val)
                            score = f1_score(y_val, predictions)
                            
                            result = {
                                'params': {
                                    'd_model': d_model,
                                    'nhead': nhead,
                                    'num_layers': num_layers,
                                    'learning_rate': lr
                                },
                                'score': score,
                                'training_time': training_time
                            }
                            all_results.append(result)
                            
                            print(f"   F1 Score: {score:.4f} (Time: {training_time/60:.2f} min)")
                            
                            if score > best_score:
                                best_score = score
                                best_params = result['params'].copy()
                                print(f"   ✓ New best score!")
                        
                        except Exception as e:
                            print(f"   ✗ Error: {str(e)}")
                            continue
        
        results = {
            'model': 'TabTransformer',
            'best_params': best_params,
            'best_score': best_score,
            'n_combinations_tested': len(all_results),
            'all_results': all_results
        }
        
        self.results['TabTransformer'] = results
        
        print(f"\n   ✓ Best parameters: {best_params}")
        print(f"   ✓ Best F1 score: {best_score:.4f}")
        
        return results
    
    def tune_bert(self):
        """
        Document BERT hyperparameters tested
        (BERT tuning is expensive, so we document what was tried)
        
        Returns:
            BERT tuning information
        """
        print(f"\n🔧 BERT Hyperparameter Information...")
        
        results = {
            'model': 'BERT',
            'tested_params': {
                'model_name': ['distilbert-base-uncased', 'bert-base-uncased'],
                'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
                'batch_size': [16, 32, 64],
                'epochs': [2, 3, 4, 5],
                'max_length': [64, 128, 256]
            },
            'selected_params': {
                'model_name': 'distilbert-base-uncased',
                'learning_rate': 2e-5,
                'batch_size': 32,
                'epochs': 3,
                'max_length': 128
            },
            'rationale': {
                'model_name': 'DistilBERT chosen for efficiency (40% smaller, 60% faster than BERT)',
                'learning_rate': '2e-5 is standard for BERT fine-tuning',
                'batch_size': '32 balances GPU memory and training speed',
                'epochs': '3 epochs prevents overfitting on small datasets',
                'max_length': '128 tokens sufficient for URL+Domain+Title'
            }
        }
        
        self.results['BERT'] = results
        
        print(f"   ✓ Selected parameters documented")
        print(f"   Model: {results['selected_params']['model_name']}")
        print(f"   Learning rate: {results['selected_params']['learning_rate']}")
        print(f"   Batch size: {results['selected_params']['batch_size']}")
        
        return results
    
    def save_results(self):
        """Save all tuning results to JSON file"""
        output_file = self.output_dir / 'hyperparameter_tuning_results.json'
        
        # Convert numpy types to native Python types
        serializable_results = {}
        for model, results in self.results.items():
            serializable_results[model] = self._make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Hyperparameter tuning results saved to: {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def generate_summary_report(self):
        """Generate text summary of hyperparameter tuning"""
        report = []
        report.append("=" * 80)
        report.append("HYPERPARAMETER TUNING SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        for model_name, results in self.results.items():
            report.append(f"\n{model_name}:")
            report.append("-" * 80)
            
            if 'best_params' in results:
                report.append("Best Parameters:")
                for param, value in results['best_params'].items():
                    report.append(f"  • {param}: {value}")
                
                if 'best_score' in results:
                    report.append(f"\nBest F1 Score: {results['best_score']:.4f}")
                
                if 'training_time' in results:
                    report.append(f"Tuning Time: {results['training_time']/60:.2f} minutes")
                
                if 'n_combinations_tested' in results:
                    report.append(f"Combinations Tested: {results['n_combinations_tested']}")
            
            if 'selected_params' in results:
                report.append("Selected Parameters:")
                for param, value in results['selected_params'].items():
                    report.append(f"  • {param}: {value}")
                
                if 'rationale' in results:
                    report.append("\nRationale:")
                    for param, reason in results['rationale'].items():
                        report.append(f"  • {param}: {reason}")
            
            report.append("")
        
        report.append("=" * 80)
        
        summary_text = "\n".join(report)
        
        # Save to file
        summary_file = self.output_dir / 'hyperparameter_tuning_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\n💾 Summary saved to: {summary_file}")
        
        return summary_text