import warnings
warnings.filterwarnings('ignore')

# Import data loading and validation modules
from loader import load_dataset, detect_target_column, validate_dataset

# Import preprocessing modules
from cleaner import preprocess_pipeline
from scaler import scale_features

# Import model training and evaluation modules
from trainer import train_all_models, tune_xgboost, cross_validate_models
from evaluator import evaluate_models

# Import visualization utilities
from visualizer import (
    plot_target_distribution, plot_correlations,
    plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance, plot_performance_comparison
)

# Import helper utilities
from helpers import (
    create_directories, print_header, print_section,
    save_model, generate_report
)

from sklearn.metrics import f1_score

# ============================================================================
# GLOBAL STATE STORAGE
# ============================================================================
state = {
    'dataset_path': None,
    'target_col': None,
    'data': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'X_train_scaled': None,
    'X_test_scaled': None,
    'scaler': None,
    'models': {},
    'predictions': {},
    'probabilities': {},
    'metrics_df': None,
    'label_encoder': None,      
    'label_mapping': None,      
    'is_binary': None           
}

# ============================================================================
# STEP 1: DATASET CONFIGURATION
# ============================================================================
def configure_dataset():
    print_header('DATASET CONFIGURATION')
    
    print('📂 Please enter your dataset path:')
    print('   Examples: engineered_dataset4.csv OR /path/to/data.csv\n')
    
    while True:
        path = input('Dataset path: ').strip()
        
        if not path:
            print('❌ Path cannot be empty.')
            continue
        
        try:
            state['data'] = load_dataset(path)
            state['dataset_path'] = path
            
            print_section('Target Column Detection')
            detected = detect_target_column(state['data'])
            
            if detected:
                print(f"🎯 Auto-detected target: '{detected}'")
                print(f"\n📊 Distribution:")
                for val, count in state['data'][detected].value_counts().items():
                    print(f"   Class {val}: {count:,} ({count/len(state['data'])*100:.1f}%)")
                
                if input(f"\nUse '{detected}' as target? (y/n): ").strip().lower() == 'y':
                    state['target_col'] = detected
                else:
                    detected = None
            
            if not state['target_col']:
                print('\n📋 Available columns:')
                for i, col in enumerate(state['data'].columns, 1):
                    nunique = state['data'][col].nunique()
                    print(f"   {i:2d}. {col} ({state['data'][col].dtype}, {nunique} unique)")
                
                while True:
                    target = input('\nEnter target column name: ').strip()
                    if target in state['data'].columns:
                        state['target_col'] = target
                        break
                    print(f"❌ Column '{target}' not found.")
            
            validate_dataset(state['data'], state['target_col'])
            
            print_section('Configuration Complete')
            print(f"✅ Dataset: {state['dataset_path']}")
            print(f"✅ Target: {state['target_col']}")
            print(f"✅ Shape: {state['data'].shape[0]:,} rows × {state['data'].shape[1]} columns")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if input('\nTry again? (y/n): ').strip().lower() != 'y':
                return False

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
def run_eda():
    if state['data'] is None:
        print("⚠️  Dataset not configured.")
        if input("Configure dataset now? (y/n): ").strip().lower() == 'y':
            if not configure_dataset():
                return
        else:
            return
    
    print_header('EXPLORATORY DATA ANALYSIS')
    
    print_section('Dataset Overview')
    
    print(f"📊 Shape: {state['data'].shape[0]:,} rows × {state['data'].shape[1]} columns")
    print(f"💾 Memory: {state['data'].memory_usage(deep=True).sum()/1024**2:.2f} MB")
    print(f"🔍 Missing values: {state['data'].isnull().sum().sum():,}")
    print(f"🔍 Duplicates: {state['data'].duplicated().sum():,}")
    
    print("\n📋 First 5 rows:")
    print(state['data'].head())
    
    print_section('Target Analysis')
    target_counts = state['data'][state['target_col']].value_counts().sort_index()
    
    for label, count in target_counts.items():
        print(f"Class {label}: {count:,} ({count/len(state['data'])*100:.1f}%)")
    
    print_section('Generating Visualizations')
    
    plot_target_distribution(state['data'], state['target_col'])
    plot_correlations(state['data'], state['target_col'])
    
    print("\n✅ EDA complete! Check outputs/ folder for visualizations.")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
def run_preprocessing():
    if state['data'] is None:
        print("⚠️  Dataset not configured.")
        if input("Configure dataset now? (y/n): ").strip().lower() == 'y':
            if not configure_dataset():
                return
        else:
            return
    
    print_header('DATA PREPROCESSING')
    
    # NEW - Handle label encoder return
    X_train, X_test, y_train, y_test, label_encoder, label_mapping = preprocess_pipeline(
        state['data'],
        state['target_col']
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    state.update({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'label_encoder': label_encoder,      # NEW
        'label_mapping': label_mapping       # NEW
    })
    
    print(f"\n✅ Preprocessing complete! {X_train.shape[1]} features ready for training.")
    print(f"   Train set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================
def run_training():
    if state['X_train'] is None:
        print("⚠️  Data not preprocessed.")
        if input("Run preprocessing now? (y/n): ").strip().lower() == 'y':
            run_preprocessing()
            if state['X_train'] is None:
                return
        else:
            return
    
    print_header('MODEL TRAINING')
    
    models = train_all_models(
        state['X_train'],
        state['y_train'],
        state['X_train_scaled']
    )
    
    state['models'] = models
    
    print(f"\n✅ All models trained successfully!")
    print(f"   Models: {', '.join(models.keys())}")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
def run_evaluation():
    if not state['models']:
        print("⚠️  Models not trained.")
        if input("Train models now? (y/n): ").strip().lower() == 'y':
            run_training()
            if not state['models']:
                return
        else:
            return
    
    print_header('MODEL EVALUATION')
    
    predictions, probabilities, metrics_df = evaluate_models(
        state['models'],
        state['X_test'],
        state['y_test'],
        state['X_test_scaled']
    )
    
    state.update({
        'predictions': predictions,
        'probabilities': probabilities,
        'metrics_df': metrics_df
    })
    
    print_section('Generating Visualizations')
    
    plot_confusion_matrices(state['y_test'], predictions)
    plot_roc_curves(state['y_test'], probabilities)
    plot_performance_comparison(metrics_df)
    
    if 'Random Forest' in state['models']:
        plot_feature_importance(
            state['models']['Random Forest'],
            state['X_train'].columns
        )
    
    print("\n✅ Evaluation complete! Check outputs/ folder for visualizations.")

# ============================================================================
# STEP 6: ADVANCED ANALYSIS
# ============================================================================
def run_advanced():
    if not state['models']:
        print("⚠️  Models not trained.")
        if input("Train models now? (y/n): ").strip().lower() == 'y':
            run_training()
            if not state['models']:
                return
        else:
            return
    
    print_header('ADVANCED ANALYSIS')
    
    print_section('Cross-Validation')
    cv_results = cross_validate_models(
        state['models'],
        state['X_train'],
        state['y_train'],
        state['X_train_scaled']
    )
    
    print_section('Hyperparameter Tuning')
    tuned_xgb = tune_xgboost(state['X_train'], state['y_train'])
    
    print_section('Tuned Model Performance')
    tuned_pred = tuned_xgb.predict(state['X_test'])
    tuned_f1 = f1_score(state['y_test'], tuned_pred, average='weighted')
    print(f"✅ Tuned XGBoost Test F1: {tuned_f1:.4f}")
    
    state['models']['XGBoost_Tuned'] = tuned_xgb
    
    print("\n✅ Advanced analysis complete!")

# ============================================================================
# STEP 7: CLUSTERING
# ============================================================================
def run_clustering():
    if state['X_train'] is None:
        print("❌ Please run preprocessing first (Option 3)")
        return
    
    print_header('UNSUPERVISED LEARNING - CLUSTERING')
    
    from clustering import run_clustering_analysis
    from visualizer import plot_clustering_results
    
    clustering_results = run_clustering_analysis(state['X_train'], sample_size=5000)
    
    print_section('Generating Clustering Visualizations')
    plot_clustering_results(clustering_results)
    
    state['clustering_results'] = clustering_results
    
    print("\n✅ Clustering analysis complete!")

# ============================================================================
# STEP 8: SAVE MODELS & GENERATE REPORT
# ============================================================================
def run_save():
    if not state['models']:
        print("⚠️  Models not trained.")
        if input("Train models now? (y/n): ").strip().lower() == 'y':
            run_training()
            if not state['models']:
                return
        else:
            return
    
    print_header('SAVING MODELS AND REPORT')
    
    saved_models = {}
    
    print_section('Saving Models')
    for name, model in state['models'].items():
        if state['metrics_df'] is not None and name in state['metrics_df']['Model'].values:
            metadata = {
                'accuracy': float(state['metrics_df'][state['metrics_df']['Model']==name]['Accuracy'].values[0]),
                'f1': float(state['metrics_df'][state['metrics_df']['Model']==name]['F1'].values[0]),
                'label_mapping': state['label_mapping']  # NEW - Include label mapping
            }
        else:
            metadata = {
                'accuracy': 0, 
                'f1': 0,
                'label_mapping': state['label_mapping']  # NEW
            }
        
        scaler = state['scaler'] if name in ['Logistic Regression', 'SVM'] else None
        
        filepath = save_model(
            model=model,
            name=name,
            metadata=metadata,
            scaler=scaler,
            feature_names=state['X_train'].columns.tolist(),
            target_col=state['target_col'],
            label_encoder=state['label_encoder']  # NEW - Save label encoder
        )
        
        saved_models[name] = filepath
    
    if state['metrics_df'] is not None:
        print_section('Generating Report')
        report_path = generate_report(
            state['dataset_path'],
            state['target_col'],
            state['metrics_df'],
            saved_models,
            state['label_mapping']  # NEW - Pass label mapping to report
        )
    
    print(f"\n✅ All files saved!")
    print(f"   Models: {len(saved_models)}")
    print(f"   Figures: outputs/")
    if state['metrics_df'] is not None:
        print(f"   Report: {report_path}")

# ============================================================================
# STEP 9: RUN COMPLETE PIPELINE
# ============================================================================
def run_all():
    print_header('RUN COMPLETE PIPELINE')
    
    if input('This will run all steps. Continue? (y/n): ').strip().lower() != 'y':
        return
    
    steps = [
        ('Configuration', configure_dataset),
        ('EDA', run_eda),
        ('Preprocessing', run_preprocessing),
        ('Training', run_training),
        ('Evaluation', run_evaluation),
        ('Advanced Analysis', run_advanced),
        ('Clustering Analysis', run_clustering),
        ('Save & Report', run_save)
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}...")
        if i == 1:
            if not func():
                print("❌ Pipeline stopped.")
                return
        else:
            func()
        
        if i < len(steps):
            input('\nPress Enter to continue...')
    
    print_header('🎉 PIPELINE COMPLETE!')
    print("All results saved in outputs/, saved_models/, and reports/")

# ============================================================================
# STATUS DISPLAY
# ============================================================================
def print_status():
    print_section('Pipeline Status')
    
    print(f"  Dataset:      {'✅ ' + state['dataset_path'] if state['dataset_path'] else '⚠️  Not configured'}")
    print(f"  Target:       {'✅ ' + state['target_col'] if state['target_col'] else '⚠️  Not set'}")
    print(f"  Data Loaded:  {'✅ ' + str(state['data'].shape[0]) + ' rows' if state['data'] is not None else '⚠️  No'}")
    print(f"  Preprocessed: {'✅ ' + str(state['X_train'].shape[0]) + ' train samples' if state['X_train'] is not None else '⚠️  No'}")
    print(f"  Models:       {'✅ ' + str(len(state['models'])) + ' trained' if state['models'] else '⚠️  None'}")

# ============================================================================
# MAIN MENU
# ============================================================================
def main_menu():
    create_directories()
    
    print_header('ASSIGNMENT 1: PHISHING DETECTION ML PIPELINE')
    print('Student: Gurmandeep Deol | ID: 104120233 | Course: SRT521')
    
    while True:
        print('\n' + '='*70)
        print_status()
        print('='*70)
        print('\n📋 MENU:')
        print('  1. Configure Dataset')
        print('  2. Exploratory Data Analysis (EDA)')
        print('  3. Data Preprocessing')
        print('  4. Train Models')
        print('  5. Evaluate Models')
        print('  6. Advanced Analysis (CV + Tuning)')
        print('  7. Clustering Analysis (Unsupervised)')
        print('  8. Save Models & Generate Report')
        print('  9. Run Complete Pipeline')
        print('  0. Exit')
        print('='*70)
        
        choice = input('\nSelect option (0-9): ').strip()
        
        if choice == '1':
            configure_dataset()
        elif choice == '2':
            run_eda()
        elif choice == '3':
            run_preprocessing()
        elif choice == '4':
            run_training()
        elif choice == '5':
            run_evaluation()
        elif choice == '6':
            run_advanced()
        elif choice == '7':
            run_clustering()
        elif choice == '8':
            run_save()
        elif choice == '9':
            run_all()
        elif choice == '0':
            print('\n✅ Goodbye! 🎉\n')
            break
        else:
            print('\n❌ Invalid option. Please try again.')
        
        if choice != '0' and choice != '9':
            input('\nPress Enter to continue...')

# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    main_menu()