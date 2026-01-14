"""
Assignment 2: Transformer-Based ML Pipeline
Main execution script with interactive CLI
Author: Gurmandeep Deol
Course: SRT521 - Advanced Data Analysis for Security
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
# Import all pipeline components
from src.data_loader import DataLoader
from src.bert_model import BERTModel
from src.tabtransformer import TabTransformer
from src.baseline_models import BaselineModels
from src.hybrid_model import HybridModel
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer
from src.utils import setup_logging, save_results
from src.hyperparameter_tuning import HyperparameterTuner
from src.computational_efficiency import ComputationalEfficiencyAnalyzer

# --------------------------------------------------------------------------------------
# UI HELPERS
# --------------------------------------------------------------------------------------

def print_banner():
    print("=" * 80)
    print("  ASSIGNMENT 2: TRANSFORMER-BASED ML PIPELINE")
    print("  Phishing Website Detection using BERT & TabTransformer")
    print("  Author: Gurmandeep Deol")
    print("=" * 80)
    print()

def print_menu():
    print("\n" + "=" * 80)
    print("  PIPELINE OPTIONS")
    print("=" * 80)
    print("  1. Load and Prepare Data")
    print("  2. Train BERT Model (Text-based)")
    print("  3. Train TabTransformer (Numerical features)")
    print("  4. Train Baseline Models (Random Forest, XGBoost, etc.)")
    print("  5. Train Hybrid Model (BERT + TabTransformer fusion)")
    print("  6. Evaluate All Models")
    print("  7. Generate Visualizations")
    print("  8. Hyperparameter Tuning")
    print("  9. Computational Efficiency Analysis")
    print(" 10. Save Results")
    print(" 11. Run Complete Pipeline (All steps)")
    print(" 12. Exit")
    print("=" * 80)

# --------------------------------------------------------------------------------------
# PIPELINE STEPS
# --------------------------------------------------------------------------------------

def run_step_1(config):
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 80)

    if not config.get("data_path"):
        config["data_path"] = input("Enter path to CSV file: ").strip()

    loader = DataLoader(config['data_path'])
    data = loader.load_data()
    X_text, X_num, y = loader.prepare_features(data)
    splits = loader.create_splits(X_text, X_num, y)

    print(f"\n✅ Data loaded successfully!")
    print(f"   Total samples: {len(data):,}")
    print(f"   Text features: {len(X_text)}")
    print(f"   Numerical features: {X_num.shape[1]}")
    print(f"   Training samples: {len(splits['y_train']):,}")
    print(f"   Validation samples: {len(splits['y_val']):,}")
    print(f"   Test samples: {len(splits['y_test']):,}")

    return splits

def run_step_2(splits, config, efficiency_analyzer=None):
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING BERT MODEL")
    print("=" * 80)

    bert = BERTModel(
        model_name=config.get("bert_model", "distilbert-base-uncased"),
        max_length=config.get("max_length", 128),
    )

    ds = bert.prepare_datasets(
        splits["X_train_text"],
        splits["X_val_text"],
        splits["X_test_text"],
        splits["y_train"],
        splits["y_val"],
        splits["y_test"],
    )

    # Measure training time if analyzer provided
    if efficiency_analyzer:
        def train_func():
            return bert.train(
                ds["train"],
                ds["val"],
                epochs=config.get("bert_epochs", 3),
                batch_size=config.get("bert_batch_size", 32),
            )
        _, results = efficiency_analyzer.measure_training_time("BERT", train_func)
    else:
        results = bert.train(
            ds["train"],
            ds["val"],
            epochs=config.get("bert_epochs", 3),
            batch_size=config.get("bert_batch_size", 32),
        )

    metrics = bert.evaluate(ds["test"])

    print(f"\n✅ BERT training complete!")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test F1 Score: {metrics['f1']:.4f}")

    return {"model": bert, "metrics": metrics, "results": results, "datasets": ds}

def run_step_3(splits, config, efficiency_analyzer=None):
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING TABTRANSFORMER")
    print("=" * 80)

    tab = TabTransformer(
        input_dim=splits["X_train_num"].shape[1],
        d_model=config.get("tabtrans_d_model", 64),
        nhead=config.get("tabtrans_nhead", 4),
        num_layers=config.get("tabtrans_layers", 2),
    )

    # Measure training time if analyzer provided
    if efficiency_analyzer:
        def train_func():
            return tab.train(
                splits["X_train_num"],
                splits["X_val_num"],
                splits["X_test_num"],
                splits["y_train"],
                splits["y_val"],
                splits["y_test"],
                epochs=config.get("tabtrans_epochs", 20),
                batch_size=config.get("tabtrans_batch_size", 128),
            )
        _, results = efficiency_analyzer.measure_training_time("TabTransformer", train_func)
    else:
        results = tab.train(
            splits["X_train_num"],
            splits["X_val_num"],
            splits["X_test_num"],
            splits["y_train"],
            splits["y_val"],
            splits["y_test"],
            epochs=config.get("tabtrans_epochs", 20),
            batch_size=config.get("tabtrans_batch_size", 128),
        )

    metrics = tab.evaluate(splits["X_test_num"], splits["y_test"])

    print(f"\n✅ TabTransformer training complete!")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test F1 Score: {metrics['f1']:.4f}")

    return {"model": tab, "metrics": metrics, "results": results}

def run_step_4(splits, config, efficiency_analyzer=None):
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING BASELINE MODELS")
    print("=" * 80)

    base = BaselineModels()
    
    # Measure training time if analyzer provided
    if efficiency_analyzer:
        def train_func():
            return base.train_all(
                splits["X_train_num"],
                splits["X_test_num"],
                splits["y_train"],
                splits["y_test"],
            )
        _, results = efficiency_analyzer.measure_training_time("Random Forest + XGBoost + LR", train_func)
    else:
        results = base.train_all(
            splits["X_train_num"],
            splits["X_test_num"],
            splits["y_train"],
            splits["y_test"],
        )

    print(f"\n✅ Baseline models trained!")
    for name, m in results.items():
        print(f"   {name}: F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}")

    return {"models": base, "results": results}

def run_step_5(bert, splits, config, efficiency_analyzer=None):
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING HYBRID MODEL")
    print("=" * 80)

    # Extract BERT embeddings
    hybrid = HybridModel(
        text_dim=768,  # DistilBERT hidden size
        num_dim=splits["X_train_num"].shape[1],
        hidden_dim=config.get("hybrid_hidden_dim", 256),
        dropout=config.get("hybrid_dropout", 0.3)
    )

    print(f"\n📝 Extracting BERT embeddings for all splits...")
    text_emb_train = hybrid.extract_bert_embeddings(bert["model"], splits["X_train_text"])
    text_emb_val = hybrid.extract_bert_embeddings(bert["model"], splits["X_val_text"])
    text_emb_test = hybrid.extract_bert_embeddings(bert["model"], splits["X_test_text"])

    # Measure training time if analyzer provided
    if efficiency_analyzer:
        def train_func():
            return hybrid.train(
                text_emb_train, text_emb_val, text_emb_test,
                splits["X_train_num"], splits["X_val_num"], splits["X_test_num"],
                splits["y_train"], splits["y_val"], splits["y_test"],
                epochs=config.get("hybrid_epochs", 20),
                batch_size=config.get("hybrid_batch_size", 64),
            )
        _, results = efficiency_analyzer.measure_training_time("Hybrid Model", train_func)
    else:
        results = hybrid.train(
            text_emb_train, text_emb_val, text_emb_test,
            splits["X_train_num"], splits["X_val_num"], splits["X_test_num"],
            splits["y_train"], splits["y_val"], splits["y_test"],
            epochs=config.get("hybrid_epochs", 20),
            batch_size=config.get("hybrid_batch_size", 64),
        )

    metrics = hybrid.evaluate(text_emb_test, splits["X_test_num"], splits["y_test"])

    print(f"\n✅ Hybrid Model training complete!")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test F1 Score: {metrics['f1']:.4f}")

    return {
        "model": hybrid, 
        "metrics": metrics, 
        "results": results,
        "embeddings": {
            "train": text_emb_train,
            "val": text_emb_val,
            "test": text_emb_test
        }
    }

def run_step_6(bert, tab, base, hybrid, splits):
    print("\n" + "=" * 80)
    print("STEP 6: COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)

    evaluator = ModelEvaluator()
    all_results = {
        "BERT": bert["metrics"],
        "TabTransformer": tab["metrics"],
        "Hybrid Model": hybrid["metrics"],
        **base["results"],
    }
    comparison = evaluator.compare_models(all_results)

    print("\n📊 MODEL COMPARISON:")
    print("=" * 80)
    print(comparison.to_string(index=False))

    best = comparison.loc[comparison["F1 Score"].idxmax(), "Model"]
    print(f"\n🏆 Best Model: {best}")

    return comparison

def run_step_7(bert, tab, base, hybrid, comparison, splits, config):
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 80)

    viz = Visualizer(output_dir=config["output_dir"])

    viz.plot_training_curves(bert["results"], tab["results"])
    viz.plot_model_comparison(comparison)
    viz.plot_confusion_matrices(
        bert["model"], tab["model"], base["models"], splits
    )
    viz.plot_roc_curves(
        bert["model"], tab["model"], base["models"], splits
    )
    viz.plot_feature_importance(base["models"], splits["X_train_num_df"])
    viz.plot_attention_analysis(bert["model"], splits["X_test_text"][:10])

    print(f"\n✅ Visualizations saved.")

def run_step_8(splits, config):
    print("\n" + "=" * 80)
    print("STEP 8: HYPERPARAMETER TUNING (GPU OPTIMIZED)")
    print("=" * 80)

    tuner = HyperparameterTuner(output_dir=config["output_dir"])

    # Use subset for speed
    subset_size = min(10000, len(splits["y_train"]))
    X_subset = splits["X_train_num"][:subset_size]
    y_subset = splits["y_train"][:subset_size]

    print(f"Using subset of {subset_size} samples for hyperparameter tuning...")

    # Tune Random Forest - 48 combinations
    print("\n" + "-" * 80)
    rf_results = tuner.tune_random_forest(X_subset, y_subset, cv=3)

    # Tune XGBoost - 36 combinations (GPU accelerated)
    print("\n" + "-" * 80)
    xgb_results = tuner.tune_xgboost(X_subset, y_subset, cv=3)

    # Tune TabTransformer - 16 combinations (GPU accelerated)
    print("\n" + "-" * 80)
    tab_subset_size = min(5000, len(splits["y_train"]))
    tabtrans_results = tuner.tune_tabtransformer(
        splits["X_train_num"][:tab_subset_size],
        splits["X_val_num"],
        splits["y_train"][:tab_subset_size],
        splits["y_val"]
    )

    # Document BERT parameters
    print("\n" + "-" * 80)
    bert_results = tuner.tune_bert()

    # Save results
    tuner.save_results()
    tuner.generate_summary_report()

    print(f"\n✅ Hyperparameter tuning complete!")
    print(f"   Total combinations tested: ~100 (optimized for GPU)")

    return tuner

def run_step_9(bert, tab, base, hybrid, splits, config):
    print("\n" + "=" * 80)
    print("STEP 9: COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("=" * 80)

    analyzer = ComputationalEfficiencyAnalyzer(output_dir=config["output_dir"])

    # Measure inference times
    print("\n⚡ Measuring inference speeds...")
    
    # BERT inference (on test set)
    if bert and "model" in bert:
        try:
            analyzer.measure_inference_time(
                "BERT", 
                bert["model"], 
                splits["X_test_text"][:1000],  # Sample for speed
                batch_sizes=[1, 16, 32]
            )
        except Exception as e:
            print(f"   Warning: Could not measure BERT inference: {str(e)}")

    # TabTransformer inference
    if tab and "model" in tab:
        analyzer.measure_inference_time(
            "TabTransformer",
            tab["model"],
            splits["X_test_num"],
            batch_sizes=[1, 16, 32, 64, 128]
        )

    # Baseline models inference
    if base and "models" in base:
        for model_name in ["Random Forest", "XGBoost"]:
            analyzer.measure_inference_time(
                model_name,
                base["models"].models[model_name],
                splits["X_test_num"],
                batch_sizes=[1, 16, 32, 64, 128]
            )

    # Hybrid model inference
    if hybrid and "model" in hybrid and "embeddings" in hybrid:
        try:
            # Create simple wrapper for hybrid inference
            class HybridWrapper:
                def __init__(self, model, text_emb, num_feat):
                    self.model = model
                    self.text_emb = text_emb
                    self.num_feat = num_feat
                
                def predict(self, X):
                    # X is just indices in this case
                    n = len(X) if hasattr(X, '__len__') else X.shape[0]
                    return self.model.predict(
                        self.text_emb[:n],
                        self.num_feat[:n]
                    )
            
            wrapper = HybridWrapper(
                hybrid["model"],
                hybrid["embeddings"]["test"],
                splits["X_test_num"]
            )
            
            analyzer.measure_inference_time(
                "Hybrid Model",
                wrapper,
                splits["X_test_num"],
                batch_sizes=[1, 16, 32, 64]
            )
        except Exception as e:
            print(f"   Warning: Could not measure Hybrid inference: {str(e)}")

    # Measure model sizes
    print("\n💾 Measuring model sizes...")
    out_dir = Path(config["output_dir"])
    
    if bert and "model" in bert:
        analyzer.measure_model_size("BERT", bert["model"], out_dir / "bert_model")
    
    if tab and "model" in tab:
        analyzer.measure_model_size("TabTransformer", tab["model"], out_dir / "tabtransformer_model")
    
    if hybrid and "model" in hybrid:
        analyzer.measure_model_size("Hybrid Model", hybrid["model"], out_dir / "hybrid_model")

    # Generate comparison and plots
    comparison_df = analyzer.compare_all_models()
    analyzer.plot_training_time_comparison()
    analyzer.plot_inference_speed_comparison()
    analyzer.plot_resource_usage()
    
    # Save results
    analyzer.save_results()
    analyzer.generate_summary_report()

    print(f"\n✅ Computational efficiency analysis complete!")

    return analyzer

def run_step_10(bert, tab, base, hybrid, comparison, config):
    print("\n" + "=" * 80)
    print("STEP 10: SAVING RESULTS")
    print("=" * 80)

    out = Path(config["output_dir"])
    out.mkdir(exist_ok=True)

    bert["model"].save(out / "bert_model")
    tab["model"].save(out / "tabtransformer_model")
    base["models"].save(out / "baseline_models")
    hybrid["model"].save(out / "hybrid_model")

    # Update save_results to include hybrid
    import json
    
    # Save all metrics
    all_metrics = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'bert': {k: float(v) if not isinstance(v, (list, dict)) else v 
                 for k, v in bert["metrics"].items() 
                 if k not in ['predictions', 'true_labels', 'confusion_matrix']},
        'tabtransformer': {k: float(v) if not isinstance(v, (list, dict)) else v 
                          for k, v in tab["metrics"].items() 
                          if k not in ['predictions', 'true_labels', 'confusion_matrix']},
        'hybrid': {k: float(v) if not isinstance(v, (list, dict)) else v 
                   for k, v in hybrid["metrics"].items() 
                   if k not in ['predictions', 'true_labels', 'confusion_matrix']},
        'baseline': {}
    }
    
    for model_name, metrics in base["results"].items():
        all_metrics['baseline'][model_name] = {
            k: float(v) if not isinstance(v, (list, dict)) else v 
            for k, v in metrics.items() 
            if k not in ['predictions', 'true_labels', 'confusion_matrix']
        }
    
    with open(out / 'all_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Save comparison
    comparison.to_csv(out / 'model_comparison.csv', index=False)

    print(f"\n✅ Results saved to {out}/")

def run_complete_pipeline(config):
    print_banner()
    
    # Initialize efficiency analyzer
    efficiency_analyzer = ComputationalEfficiencyAnalyzer(output_dir=config["output_dir"])
    
    splits = run_step_1(config)
    bert = run_step_2(splits, config, efficiency_analyzer)
    tab = run_step_3(splits, config, efficiency_analyzer)
    base = run_step_4(splits, config, efficiency_analyzer)
    hybrid = run_step_5(bert, splits, config, efficiency_analyzer)
    comp = run_step_6(bert, tab, base, hybrid, splits)
    run_step_7(bert, tab, base, hybrid, comp, splits, config)
    
    # Optional: Run hyperparameter tuning
    response = input("\nRun hyperparameter tuning? (y/n): ").strip().lower()
    if response == 'y':
        run_step_8(splits, config)
    
    run_step_9(bert, tab, base, hybrid, splits, config)
    run_step_10(bert, tab, base, hybrid, comp, config)
    
    print("\n🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")

# --------------------------------------------------------------------------------------
# INTERACTIVE MODE
# --------------------------------------------------------------------------------------

def interactive_mode(config):
    print_banner()
    splits = None
    bert = None
    tab = None
    base = None
    hybrid = None
    comp = None
    efficiency_analyzer = ComputationalEfficiencyAnalyzer(output_dir=config["output_dir"])

    while True:
        print_menu()
        choice = input("\nEnter your choice (1-12): ").strip()
        
        if choice == "1":
            splits = run_step_1(config)
        
        elif choice == "2":
            if not splits: print("\n⚠️ Load data first (Option 1)"); continue
            bert = run_step_2(splits, config, efficiency_analyzer)
        
        elif choice == "3":
            if not splits: print("\n⚠️ Load data first (Option 1)"); continue
            tab = run_step_3(splits, config, efficiency_analyzer)
        
        elif choice == "4":
            if not splits: print("\n⚠️ Load data first (Option 1)"); continue
            base = run_step_4(splits, config, efficiency_analyzer)
        
        elif choice == "5":
            if not all([splits, bert]): print("\n⚠️ Load data and train BERT first"); continue
            hybrid = run_step_5(bert, splits, config, efficiency_analyzer)
        
        elif choice == "6":
            if not all([bert, tab, base, hybrid]): 
                print("\n⚠️ Train all models first (BERT, TabTransformer, Baselines, Hybrid)"); 
                continue
            comp = run_step_6(bert, tab, base, hybrid, splits)
        
        elif choice == "7":
            if not all([bert, tab, base, hybrid, comp]): 
                print("\n⚠️ Run evaluation first (Option 6)"); 
                continue
            run_step_7(bert, tab, base, hybrid, comp, splits, config)
        
        elif choice == "8":
            if not splits: print("\n⚠️ Load data first (Option 1)"); continue
            run_step_8(splits, config)
        
        elif choice == "9":
            if not all([bert, tab, base, hybrid]): 
                print("\n⚠️ Train all models first"); 
                continue
            run_step_9(bert, tab, base, hybrid, splits, config)
        
        elif choice == "10":
            if not all([bert, tab, base, hybrid, comp]): 
                print("\n⚠️ Complete previous steps first"); 
                continue
            run_step_10(bert, tab, base, hybrid, comp, config)
        
        elif choice == "11":
            run_complete_pipeline(config)
            break
        
        elif choice == "12":
            print("\n👋 Exiting...")
            break
        
        else:
            print("\n❌ Invalid choice.")

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=False)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    config = {
        "data_path": args.data,
        "output_dir": args.output,
        "bert_model": "distilbert-base-uncased",
        "bert_epochs": 3,
        "bert_batch_size": 32,
        "max_length": 128,
        "tabtrans_epochs": 20,
        "tabtrans_batch_size": 128,
        "tabtrans_d_model": 64,
        "tabtrans_nhead": 4,
        "tabtrans_layers": 2,
        "hybrid_epochs": 20,
        "hybrid_batch_size": 64,
        "hybrid_hidden_dim": 256,
        "hybrid_dropout": 0.3,
    }

    setup_logging(config["output_dir"])

    if args.run_all:
        run_complete_pipeline(config)
    else:
        interactive_mode(config)

if __name__ == "__main__":
    main()