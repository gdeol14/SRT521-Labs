# models/trainer.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
import xgboost as xgb
from config import MODELS_CONFIG, XGBOOST_PARAM_GRID, RANDOM_FOREST_PARAM_GRID, CV_FOLDS, RANDOM_STATE

# ============================================================================
# MODEL INITIALIZATION (Lab 4)
# ============================================================================
def initialize_models():
    models = {
        # Tree-based models (don't need scaling)
        'Random Forest': RandomForestClassifier(**MODELS_CONFIG['Random Forest']),
        'XGBoost': xgb.XGBClassifier(**MODELS_CONFIG['XGBoost']),
        
        # Linear models (require scaling)
        'Logistic Regression': LogisticRegression(**MODELS_CONFIG['Logistic Regression']),
        'SVM': SVC(**MODELS_CONFIG['SVM'])
    }
    return models

# ============================================================================
# SINGLE MODEL TRAINING (Lab 4)
# ============================================================================
def train_model(name, model, X_train, y_train, X_train_scaled=None):
    """
    Train a single model with appropriate data (scaled/unscaled)
    
    Lab 4: Proper handling of tree-based vs linear models
    - Tree models: Use original features
    - Linear models: Use scaled features
    """
    print(f"\n🔧 Training {name}...")
    
    if name in ['Logistic Regression', 'SVM']:
        if X_train_scaled is None:
            raise ValueError(f"Scaled data required for {name}")
        
        # SVM subset for speed (Lab 4 approach)
        if name == 'SVM' and X_train_scaled.shape[0] > 10000:
            print(f"   ⚠️  Using 5000-sample subset for SVM (O(n²) complexity)")
            idx = np.random.choice(X_train_scaled.shape[0], 5000, replace=False)
            model.fit(X_train_scaled[idx], y_train.iloc[idx])
        else:
            model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    print(f"✅ {name} trained")
    return model

# ============================================================================
# TRAIN ALL MODELS (Lab 4)
# ============================================================================
def train_all_models(X_train, y_train, X_train_scaled):
    """
    Train all 4 baseline models
    
    Lab 4 requirement: Train multiple algorithms and compare
    """
    print("="*70)
    print("MODEL TRAINING (Lab 4)".center(70))
    print("="*70)
    
    models = initialize_models()
    trained_models = {}
    
    for name, model in models.items():
        trained_models[name] = train_model(name, model, X_train, y_train, X_train_scaled)
    
    print("\n✅ All 4 models trained!")
    print("="*70)
    
    return trained_models


def cross_validate_models(models, X_train, y_train, X_train_scaled):
    print("\n" + "="*70)
    print("CROSS-VALIDATION - ALL 4 MODELS (Lab 6)".center(70))
    print("="*70)
    print(f"\nPerforming {CV_FOLDS}-fold stratified cross-validation...")
    print(f"✅ Running on ALL 4 models")
    print(f"⚠️  This may take 2-3 minutes (SVM is slow)...")
    
    # Stratified K-Fold (maintains class proportions)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    cv_results = {}
    
    # IMPORTANT: Cross-validate ALL models
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] Cross-validating {name}...")
        
        # Select appropriate data
        if name in ['Logistic Regression', 'SVM']:
            X_cv = X_train_scaled
            print(f"   Using scaled features")
        else:
            X_cv = X_train
            print(f"   Using original features")
        
        # For SVM, use subset for CV to save time but still get estimate
        if name == 'SVM' and X_cv.shape[0] > 10000:
            print(f"   ⚠️  Using 10,000-sample subset for SVM CV (O(n²) limitation)")
            # Stratified subset
            from sklearn.model_selection import train_test_split
            X_subset, _, y_subset, _ = train_test_split(
                X_cv, y_train, train_size=10000, random_state=RANDOM_STATE, stratify=y_train
            )
            scores = cross_val_score(
                model, X_subset, y_subset,
                cv=skf, scoring='f1_weighted', n_jobs=-1
            )
        else:
            # Full CV for other models
            scores = cross_val_score(
                model, X_cv, y_train,
                cv=skf, scoring='f1_weighted', n_jobs=-1
            )
        
        cv_results[name] = scores
        
        # Display results
        print(f"   Scores: {scores.round(4)}")
        print(f"   Mean: {scores.mean():.4f} ± {scores.std()*2:.4f}")
    
    # Summary
    print("\n" + "-"*70)
    print("CROSS-VALIDATION SUMMARY (All 4 Models)".center(70))
    print("-"*70)
    for name, scores in cv_results.items():
        print(f"{name:20} | {scores.mean():.4f} ± {scores.std():.4f}")
    print("-"*70)
    
    best_model = max(cv_results.items(), key=lambda x: x[1].mean())
    print(f"\n🏆 Best CV Performance: {best_model[0]} ({best_model[1].mean():.4f})")
    print("="*70)
    
    return cv_results

# ============================================================================
# HYPERPARAMETER TUNING - XGBOOST (Lab 6)
# ============================================================================
def tune_xgboost(X_train, y_train):
    """
    Hyperparameter tuning for XGBoost (Lab 6 - Advanced Analysis)
    
    Uses RandomizedSearchCV (faster than GridSearch)
    Tests 10 random combinations with 3-fold CV
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - XGBoost (Lab 6)".center(70))
    print("="*70)
    
    print(f"\n🎯 Tuning XGBoost with RandomizedSearchCV...")
    print(f"   Parameter space: {XGBOOST_PARAM_GRID}")
    
    base_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    random_search = RandomizedSearchCV(
        base_model,
        XGBOOST_PARAM_GRID,
        n_iter=10,              # Test 10 random combinations
        cv=3,                   # 3-fold CV (faster than 5)
        scoring='f1_weighted',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"\n⏳ Training 30 models (10 combinations × 3 folds)...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Tuning complete!")
    print(f"\n📋 Best Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"   • {param}: {value}")
    print(f"\n📊 Best CV F1: {random_search.best_score_:.4f}")
    
    print("="*70)
    
    return random_search.best_estimator_

# ============================================================================
# GRID SEARCH - RANDOM FOREST (Lab 6 Bonus)
# ============================================================================
def tune_random_forest(X_train, y_train):
    """
    Grid search for Random Forest (Lab 6 comprehensive tuning)
    
    Optional: More thorough than XGBoost random search
    Tests ALL combinations (slower but exhaustive)
    """
    print("\n" + "="*70)
    print("GRID SEARCH - Random Forest (Lab 6 Bonus)".center(70))
    print("="*70)
    
    print(f"\n🔍 Grid search with {len(RANDOM_FOREST_PARAM_GRID['n_estimators']) * len(RANDOM_FOREST_PARAM_GRID['max_depth']) * len(RANDOM_FOREST_PARAM_GRID['min_samples_split']) * len(RANDOM_FOREST_PARAM_GRID['min_samples_leaf'])} combinations...")
    
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    grid_search = GridSearchCV(
        base_model,
        RANDOM_FOREST_PARAM_GRID,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"⏳ This may take 3-5 minutes...")
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Grid search complete!")
    print(f"\n📋 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   • {param}: {value}")
    print(f"\n📊 Best CV F1: {grid_search.best_score_:.4f}")
    
    print("="*70)
    
    return grid_search.best_estimator_


def run_advanced_tuning(X_train, y_train):
    tuned_models = {}
    
    # Tune XGBoost (primary - fastest performer)
    print("\n🚀 Tuning XGBoost (primary model)...")
    tuned_models['XGBoost_Tuned'] = tune_xgboost(X_train, y_train)
    
    # Optionally tune Random Forest (slower but thorough)
    tune_rf = input("\n❓ Also tune Random Forest? (takes 3-5 min) (y/n): ").strip().lower()
    if tune_rf == 'y':
        print("\n🚀 Tuning Random Forest...")
        tuned_models['RandomForest_Tuned'] = tune_random_forest(X_train, y_train)
    
    return tuned_models