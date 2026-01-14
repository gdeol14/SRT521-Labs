# utils/visualizer.py
# Visualization module (FIXED for all datasets)
# Student: Gurmandeep Deol | ID: 104120233

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import OUTPUT_DIR
import numpy as np

plt.style.use('default')
sns.set_palette('husl')

# ============================================================================
# UTILITY: SAVE FIGURE
# ============================================================================
def save_figure(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.close(fig)

# ============================================================================
# PLOT 1: TARGET DISTRIBUTION
# ============================================================================
def plot_target_distribution(data, target_col):
    counts = data[target_col].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar Chart
    counts.plot(kind='bar', ax=ax1, color=['green', 'red'], alpha=0.7)
    ax1.set_title('Target Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(rotation=0)
    
    # Pie Chart
    ax2.pie(counts.values,
            labels=[f'Class {i}' for i in counts.index],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Class Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'target_distribution.png')

# ============================================================================
# PLOT 2: FEATURE CORRELATIONS (FIXED - Handle string targets)
# ============================================================================
def plot_correlations(data, target_col, top_n=15):
    """
    Visualize top feature correlations with target
    
    FIXED: Converts string targets to numeric before correlation
    """
    print(f"\n📊 Calculating feature correlations...")
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    numeric_cols = [c for c in numeric_cols if c != target_col]
    
    if len(numeric_cols) == 0:
        print("⚠️  No numeric features for correlation analysis")
        return
    
    # FIX: Convert target to numeric if it's string
    target_data = data[target_col].copy()
    if target_data.dtype == 'object':
        print(f"   ⚠️ Target is string type, converting to numeric...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        target_data = pd.Series(le.fit_transform(target_data), index=target_data.index)
        print(f"   ✅ Converted: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Calculate correlations
    try:
        correlations = data[numeric_cols].corrwith(target_data).abs()
    except Exception as e:
        print(f"⚠️  Could not calculate correlations: {e}")
        return
    
    # Remove NaN correlations
    correlations = correlations.dropna()
    
    if len(correlations) == 0:
        print("⚠️  No valid correlations found")
        return
    
    # Get top N
    top_corr = correlations.sort_values(ascending=False).head(top_n)
    
    if len(top_corr) == 0:
        print("⚠️  No features to plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(top_corr)), top_corr.values, color='steelblue')
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index)
    ax.invert_yaxis()
    
    ax.set_xlabel('Absolute Correlation')
    ax.set_title(f'Top {len(top_corr)} Feature Correlations with Target', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'feature_correlations.png')

# ============================================================================
# PLOT 3: CONFUSION MATRICES
# ============================================================================
def plot_confusion_matrices(y_test, predictions):
    from sklearn.metrics import confusion_matrix
    
    models_list = list(predictions.items())
    n_models = len(models_list)
    
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (name, y_pred) in enumerate(models_list):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm,
                   annot=True,
                   fmt='d',
                   cmap='Blues',
                   ax=axes[i],
                   cbar=False)
        
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    for i in range(n_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'confusion_matrices.png')

# ============================================================================
# PLOT 4: ROC CURVES (FIXED - Extract positive class probability)
# ============================================================================
def plot_roc_curves(y_test, probabilities):
    """
    Plot ROC curves for all models
    
    CRITICAL FIX: Extract probability of positive class (column 1)
    sklearn.roc_curve needs 1D array, not 2D probabilities
    """
    from sklearn.metrics import roc_curve, auc
    
    print(f"\n📊 Generating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, y_prob in probabilities.items():
        if y_prob is None:
            print(f"   ⚠️ Skipping {name}: No probabilities available")
            continue
        
        try:
            # CRITICAL FIX: Extract positive class probability
            if len(y_prob.shape) == 2:
                # Binary classification: use column 1 (positive class)
                if y_prob.shape[1] == 2:
                    y_prob_pos = y_prob[:, 1]
                    print(f"   ✅ {name}: Extracted positive class probabilities")
                else:
                    # Multi-class: use max probability
                    y_prob_pos = y_prob.max(axis=1)
                    print(f"   ⚠️ {name}: Multi-class, using max probability")
            else:
                # Already 1D
                y_prob_pos = y_prob
                print(f"   ✅ {name}: Probabilities already 1D")
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr,
                   label=f'{name} (AUC={roc_auc:.3f})',
                   linewidth=2)
            
        except Exception as e:
            print(f"   ❌ {name}: ROC curve failed - {e}")
            continue
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'roc_curves.png')

# ============================================================================
# PLOT 5: FEATURE IMPORTANCE
# ============================================================================
def plot_feature_importance(model, feature_names, top_n=20):
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  Model doesn't have feature_importances_ attribute")
        return
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.barh(range(len(importance_df)),
           importance_df['Importance'],
           color='forestgreen')
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'feature_importance.png')

# ============================================================================
# PLOT 6: PERFORMANCE COMPARISON
# ============================================================================
def plot_performance_comparison(metrics_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, metric in enumerate(metrics):
        axes[i].barh(metrics_df['Model'],
                    metrics_df[metric],
                    color=colors[i])
        
        axes[i].set_xlabel(metric, fontsize=11)
        axes[i].set_title(f'{metric} Comparison', 
                         fontsize=12, fontweight='bold')
        axes[i].grid(axis='x', alpha=0.3)
        axes[i].set_xlim(0, 1)
    
    plt.tight_layout()
    save_figure(fig, 'performance_comparison.png')

# ============================================================================
# CLUSTERING VISUALIZATIONS
# ============================================================================
def plot_clustering_results(clustering_results):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    k_range = clustering_results['k_range']
    inertias = clustering_results['inertias']
    silhouette_scores = clustering_results['silhouette_scores']
    optimal_k = clustering_results['optimal_k']
    X_pca = clustering_results['X_pca']
    kmeans_labels = clustering_results['kmeans_labels']
    dbscan_labels = clustering_results['dbscan_labels']
    pca = clustering_results['pca']
    
    # 1. Elbow Method
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Silhouette Scores
    axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, 
                    label=f'Optimal K={optimal_k}')
    axes[1].set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. K-means PCA
    scatter1 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                               cmap='viridis', alpha=0.6, s=20)
    axes[2].set_title(f'K-means Clustering (K={optimal_k})', 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter1, ax=axes[2])
    
    # 4. DBSCAN PCA
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    scatter2 = axes[3].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, 
                               cmap='viridis', alpha=0.6, s=20)
    axes[3].set_title(f'DBSCAN Clustering ({n_clusters_dbscan} clusters)', 
                      fontsize=14, fontweight='bold')
    axes[3].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[3].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter2, ax=axes[3])
    
    # 5. Cluster Size Comparison
    kmeans_sizes = np.bincount(kmeans_labels)
    dbscan_sizes = np.bincount(dbscan_labels[dbscan_labels != -1])
    
    x_pos = np.arange(max(len(kmeans_sizes), len(dbscan_sizes)))
    width = 0.35
    
    axes[4].bar(x_pos - width/2, 
                np.pad(kmeans_sizes, (0, len(x_pos)-len(kmeans_sizes))), 
                width, label='K-means', alpha=0.7)
    axes[4].bar(x_pos + width/2, 
                np.pad(dbscan_sizes, (0, len(x_pos)-len(dbscan_sizes))), 
                width, label='DBSCAN', alpha=0.7)
    axes[4].set_title('Cluster Size Comparison', fontsize=14, fontweight='bold')
    axes[4].set_xlabel('Cluster ID')
    axes[4].set_ylabel('Number of Samples')
    axes[4].legend()
    axes[4].grid(axis='y', alpha=0.3)
    
    # 6. Algorithm Comparison
    kmeans_sil = silhouette_scores[optimal_k - min(k_range)]
    dbscan_sil = clustering_results['dbscan_silhouette']
    
    algorithms = ['K-means', 'DBSCAN']
    scores = [kmeans_sil, dbscan_sil]
    colors_bar = ['steelblue', 'coral']
    
    bars = axes[5].bar(algorithms, scores, color=colors_bar, alpha=0.7)
    axes[5].set_title('Clustering Algorithm Comparison', 
                      fontsize=14, fontweight='bold')
    axes[5].set_ylabel('Silhouette Score')
    axes[5].set_ylim(0, max(scores) * 1.2)
    axes[5].grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[5].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'clustering_analysis.png')
    
    print("✅ Clustering visualizations saved!")