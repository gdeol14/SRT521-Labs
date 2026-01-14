"""
Visualization Module
Generate all plots and charts for Assignment 2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import torch

class Visualizer:
    """Visualization generator"""
    
    def __init__(self, output_dir='results'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_training_curves(self, bert_history, tabtrans_history):
        """Plot training curves for both models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Curves: BERT vs TabTransformer', fontsize=16, fontweight='bold')
        
        # BERT Loss
        ax1 = axes[0, 0]
        if len(bert_history['train_loss']) > 0:
            ax1.plot(bert_history['train_loss'], label='BERT Train Loss', marker='o', linewidth=2)
        if len(bert_history['val_loss']) > 0:
            ax1.plot(bert_history['val_loss'], label='BERT Val Loss', marker='s', linewidth=2)
        ax1.set_title('BERT Training Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # TabTransformer Loss
        ax2 = axes[0, 1]
        ax2.plot(tabtrans_history['train_loss'], label='TabTransformer Train Loss', marker='o', linewidth=2)
        ax2.plot(tabtrans_history['val_loss'], label='TabTransformer Val Loss', marker='s', linewidth=2)
        ax2.set_title('TabTransformer Training Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # BERT Accuracy
        ax3 = axes[1, 0]
        if len(bert_history['val_accuracy']) > 0:
            ax3.plot(bert_history['val_accuracy'], label='BERT Val Accuracy', marker='o', linewidth=2)
        ax3.set_title('BERT Validation Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # TabTransformer Accuracy
        ax4 = axes[1, 1]
        ax4.plot(tabtrans_history['val_accuracy'], label='TabTransformer Val Accuracy', marker='o', linewidth=2)
        ax4.set_title('TabTransformer Validation Accuracy', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved training_curves.png")
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison chart"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=color, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_title(metric, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim([0.98, 1.001])
            ax.grid(axis='y', alpha=0.3)
            
            for bar, value in zip(bars, comparison_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001, f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved model_comparison.png")
    
    def plot_confusion_matrices(self, bert_model, tabtrans_model, baseline_models, splits):
        """Plot confusion matrices for all models"""
        from sklearn.metrics import confusion_matrix
        
        bert_pred = bert_model.trainer.predict(bert_model.tokenize_data(splits['X_test_text'], splits['y_test'])).predictions.argmax(axis=-1)
        tabtrans_pred = tabtrans_model.predict(splits['X_test_num'])
        rf_pred = baseline_models.models['Random Forest'].predict(splits['X_test_num'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        models_data = [('BERT', bert_pred), ('TabTransformer', tabtrans_pred), ('Random Forest', rf_pred)]
        
        for ax, (name, pred) in zip(axes, models_data):
            cm = confusion_matrix(splits['y_test'], pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Legitimate', 'Phishing'],
                        yticklabels=['Legitimate', 'Phishing'],
                        cbar_kws={'label': 'Count'})
            ax.set_title(f'{name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved confusion_matrices.png")
    
    def plot_roc_curves(self, bert_model, tabtrans_model, baseline_models, splits):
        """Plot ROC curves for all models"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            bert_test_dataset = bert_model.tokenize_data(splits['X_test_text'], splits['y_test'])
            bert_probs = torch.nn.functional.softmax(torch.tensor(bert_model.trainer.predict(bert_test_dataset).predictions), dim=1)[:, 1].numpy()
            fpr, tpr, _ = roc_curve(splits['y_test'], bert_probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'BERT (AUC = {roc_auc:.4f})', linewidth=2)
        except:
            pass
        
        for model_name in ['Random Forest', 'XGBoost']:
            probs = baseline_models.models[model_name].predict_proba(splits['X_test_num'])[:, 1]
            fpr, tpr, _ = roc_curve(splits['y_test'], probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved roc_curves.png")
    
    def plot_feature_importance(self, baseline_models, X_train, feature_names=None, top_n=20):
        """Plot feature importance from Random Forest"""
        rf_model = baseline_models.models['Random Forest']

        if feature_names is None:
            if hasattr(X_train, 'columns'):
                feature_names = X_train.columns
            else:
                feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'])):
            ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved feature_importance.png")
    
    def plot_attention_analysis(self, bert_model, sample_texts):
        """Plot attention analysis for BERT"""
        fig, ax = plt.subplots(figsize=(12, 6))
        results = bert_model.predict(sample_texts[:5])
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        
        colors = ['red' if 'LABEL_1' in label or 'POSITIVE' in label else 'green' for label in labels]
        bars = ax.barh(range(len(scores)), scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(scores)))
        ax.set_yticklabels([f'Sample {i+1}' for i in range(len(scores))])
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_title('BERT Prediction Confidence', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, score, label) in enumerate(zip(bars, scores, labels)):
            pred = 'Phishing' if 'LABEL_1' in label or 'POSITIVE' in label else 'Legitimate'
            ax.text(score + 0.02, i, f'{pred} ({score:.3f})', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Saved attention_analysis.png")
