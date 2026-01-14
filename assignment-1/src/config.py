# config.py

# ============================================================================
# BASIC SETTINGS (Lab 1-2)
# ============================================================================
RANDOM_STATE = 42  # Seed for reproducibility across all experiments
TEST_SIZE = 0.2    # 20% of data reserved for testing (80-20 split)

# ============================================================================
# DATA LEAKAGE REMOVAL (Lab 2 + Lab 6 Lesson Learned)
# ============================================================================
# IMPORTANT: These columns are REMOVED because they cause data leakage
# Data leakage = using information that wouldn't be available at prediction time
LEAKAGE_COLUMNS = [
    # Identifier columns (not predictive features)
    'FILENAME',   # Internal file reference - not available in production
    'URL',        # Raw URL string - should extract features from it instead
    'Domain',     # Raw domain string - should extract features instead
    'Title',      # Raw title text - should extract features instead
    'TLD',        # Top-level domain as text - converted to numeric features
    
    # Pre-calculated aggregate scores (from Lab 3 - these were engineered but too perfect)
    # Removing these based on Lab 6 findings - they created unrealistic 100% accuracy
    'Security_Risk_Score',        # Composite score - causes leakage
    'Suspicious_Elements_Count',  # Aggregate count - causes leakage
    'Financial_Keywords',         # Binary aggregate - causes leakage
    'URL_Complexity_Score',       # Composite score - causes leakage
    'Domain_Trust_Score',         # Composite score - causes leakage
    'Content_Quality_Score',      # Composite score - causes leakage
    'URLSimilarityIndex',         # Pre-calculated similarity - causes leakage
    'HasSocialNet',               # May indicate manual verification
    'HasCopyrightInfo',           # May indicate manual verification
    'IsHTTPS'                     # Should be part of URL features, not standalone
]

# ============================================================================
# FEATURE ENGINEERING SETTINGS (Lab 3 - Assignment Requirement)
# ============================================================================
# Assignment 1 requires "Create at least 5 domain-relevant features"
# These are LEGITIMATE phishing-detection features (not data leakage)

# Features to create from URL analysis
URL_FEATURES = [
    'url_length',              # Length of URL (phishing often uses long URLs)
    'domain_length',           # Length of domain (short domains often legitimate)
    'num_dots',                # Number of dots (subdomains = suspicious)
    'num_special_chars',       # Count of @, -, _, = (obfuscation tactics)
    'has_ip_address'           # Boolean: IP instead of domain (highly suspicious)
]

# Features to create from content analysis
CONTENT_FEATURES = [
    'page_complexity',         # LineOfCode / URLLength ratio
    'external_resources_ratio',# External refs / Total refs
    'form_submission_risk',    # Combination of form + password fields
    'technology_stack_score',  # CSS + JS + Images complexity
    'trust_indicators'         # Favicon + Description + Title presence
]

# ============================================================================
# MODEL CONFIGURATIONS (Lab 4 - Baseline Models)
# ============================================================================
MODELS_CONFIG = {
    'Random Forest': {
        'n_estimators': 100,      # Number of trees in forest
        'random_state': RANDOM_STATE,
        'n_jobs': -1              # Use all CPU cores
    },
    'XGBoost': {
        'n_estimators': 100,      # Number of boosting rounds
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss', # Loss function for binary classification
        'use_label_encoder': False # Suppress warning
    },
    'Logistic Regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000          # Maximum iterations for convergence
    },
    'SVM': {
        'kernel': 'rbf',          # Radial basis function kernel (non-linear)
        'random_state': RANDOM_STATE,
        'probability': True       # Enable probability estimates for ROC curves
    }
}

# ============================================================================
# HYPERPARAMETER TUNING GRID (Lab 6 - Advanced Analysis)
# ============================================================================
# For XGBoost Random Search (most promising model from Lab 5)
XGBOOST_PARAM_GRID = {
    'n_estimators': [100, 200, 300],         # More trees = better but slower
    'max_depth': [3, 4, 5, 6],               # Tree depth (deeper = more complex)
    'learning_rate': [0.01, 0.1, 0.2, 0.3]   # Step size (smaller = more careful)
}

# For Random Forest Grid Search
RANDOM_FOREST_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ============================================================================
# CROSS-VALIDATION SETTINGS (Lab 6)
# ============================================================================
CV_FOLDS = 5  # 5-fold cross-validation (standard practice)
              # Provides reliable performance estimates
              # Each fold uses 80% train, 20% validation

# ============================================================================
# OUTPUT PATHS (Lab 1-6 Integration)
# ============================================================================
OUTPUT_DIR = 'outputs'        # Visualizations (EDA, confusion matrices, ROC curves)
MODELS_DIR = 'saved_models'   # Trained models as .pkl files
REPORTS_DIR = 'reports'       # Generated markdown reports

# ============================================================================
# EVALUATION METRICS (Lab 5)
# ============================================================================
# Primary metric for phishing detection
PRIMARY_METRIC = 'f1_weighted'  # F1 handles class imbalance well
                                 # Better than accuracy for security tasks

# Metrics to track
METRICS_TO_TRACK = [
    'accuracy',    # Overall correctness
    'precision',   # Of predicted phishing, how many correct?
    'recall',      # Of actual phishing, how many caught?
    'f1_score'     # Harmonic mean of precision and recall
]