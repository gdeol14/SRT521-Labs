"""
Evaluation Module
Model comparison and metrics
"""

import pandas as pd
import numpy as np

class ModelEvaluator:
    """Model evaluation and comparison"""
    
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def compare_models(self, results_dict):
        """
        Compare all models
        
        Args:
            results_dict: Dictionary with model names as keys and metrics as values
            
        Returns:
            DataFrame with comparison
        """
        print(f"\n📊 Comparing {len(results_dict)} models...")
        
        comparison_data = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': []
        }
        
        for model_name, metrics in results_dict.items():
            comparison_data['Model'].append(model_name)
            comparison_data['Accuracy'].append(metrics['accuracy'])
            comparison_data['Precision'].append(metrics['precision'])
            comparison_data['Recall'].append(metrics['recall'])
            comparison_data['F1 Score'].append(metrics['f1'])
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def calculate_improvements(self, transformer_metrics, baseline_metrics):
        """
        Calculate improvements over baseline
        
        Args:
            transformer_metrics: Transformer model metrics
            baseline_metrics: Baseline model metrics
            
        Returns:
            Dictionary of improvements
        """
        improvements = {}
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            transformer_val = transformer_metrics[metric]
            baseline_val = baseline_metrics[metric]
            
            absolute_improvement = transformer_val - baseline_val
            relative_improvement = (absolute_improvement / baseline_val) * 100 if baseline_val > 0 else 0
            
            improvements[metric] = {
                'absolute': absolute_improvement,
                'relative': relative_improvement,
                'transformer': transformer_val,
                'baseline': baseline_val
            }
        
        return improvements
    
    def generate_summary_report(self, comparison_df, bert_metrics, tabtrans_metrics, baseline_results):
        """
        Generate text summary report
        
        Args:
            comparison_df: Model comparison dataframe
            bert_metrics: BERT metrics
            tabtrans_metrics: TabTransformer metrics
            baseline_results: Baseline results dict
            
        Returns:
            String report
        """
        report = []
        report.append("=" * 80)
        report.append("ASSIGNMENT 2: MODEL PERFORMANCE SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Best model
        best_model = comparison_df.iloc[0]
        report.append(f"🏆 BEST MODEL: {best_model['Model']}")
        report.append(f"   F1 Score: {best_model['F1 Score']:.4f}")
        report.append(f"   Accuracy: {best_model['Accuracy']:.4f}")
        report.append("")
        
        # All models
        report.append("📊 ALL MODELS:")
        report.append("-" * 80)
        for idx, row in comparison_df.iterrows():
            report.append(f"{idx+1}. {row['Model']:<25} | "
                         f"F1: {row['F1 Score']:.4f} | "
                         f"Acc: {row['Accuracy']:.4f}")
        report.append("")
        
        # Transformer vs Baseline comparison
        rf_metrics = baseline_results.get('Random Forest', baseline_results.get('random_forest'))
        if rf_metrics:
            report.append("🔬 TRANSFORMER IMPROVEMENTS (vs Random Forest):")
            report.append("-" * 80)
            
            bert_improv = self.calculate_improvements(bert_metrics, rf_metrics)
            report.append(f"BERT:")
            report.append(f"   F1 Score: {bert_improv['f1']['absolute']:+.4f} "
                         f"({bert_improv['f1']['relative']:+.2f}%)")
            
            tabtrans_improv = self.calculate_improvements(tabtrans_metrics, rf_metrics)
            report.append(f"TabTransformer:")
            report.append(f"   F1 Score: {tabtrans_improv['f1']['absolute']:+.4f} "
                         f"({tabtrans_improv['f1']['relative']:+.2f}%)")
            report.append("")
        
        # Error analysis
        report.append("🔍 ERROR ANALYSIS:")
        report.append("-" * 80)
        
        for model_name in ['BERT', 'TabTransformer', 'Random Forest']:
            if model_name == 'BERT':
                cm = bert_metrics['confusion_matrix']
            elif model_name == 'TabTransformer':
                cm = tabtrans_metrics['confusion_matrix']
            elif model_name in baseline_results:
                cm = baseline_results[model_name]['confusion_matrix']
            else:
                continue
            
            tn, fp, fn, tp = cm.ravel()
            report.append(f"{model_name}:")
            report.append(f"   True Positives:  {tp:>6,}")
            report.append(f"   False Positives: {fp:>6,}")
            report.append(f"   False Negatives: {fn:>6,}")
            report.append(f"   True Negatives:  {tn:>6,}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)