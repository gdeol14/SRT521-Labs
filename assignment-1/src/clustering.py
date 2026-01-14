# models/clustering.py
"""Unsupervised learning - Clustering analysis"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from config import RANDOM_STATE

# ============================================================================
# DATA PREPARATION FOR CLUSTERING
# ============================================================================
def prepare_clustering_data(X_train, sample_size=5000):
    """
    Prepare data for clustering analysis
    
    - Remove labels (unsupervised learning)
    - Scale features for distance-based algorithms
    - Use sample for large datasets
    """
    print("="*70)
    print("PREPARING DATA FOR CLUSTERING".center(70))
    print("="*70)
    
    print(f"\n📊 Original data shape: {X_train.shape}")
    
    # Scale features (important for distance-based clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    print(f"✅ Features scaled for clustering")
    
    # Use sample for large datasets
    if X_scaled.shape[0] > 10000:
        print(f"\n⚠️  Large dataset detected. Using sample for clustering...")
        sample_size = min(sample_size, X_scaled.shape[0])
        sample_idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
        print(f"✅ Sample size: {X_sample.shape[0]} samples")
    else:
        X_sample = X_scaled
        print(f"✅ Using full dataset: {X_sample.shape[0]} samples")
    
    print("="*70)
    return X_sample, scaler

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================
def find_optimal_k(X_sample, k_range=range(2, 11)):
    """
    Find optimal number of clusters using elbow method and silhouette score
    """
    print("\n" + "="*70)
    print("K-MEANS: FINDING OPTIMAL K".center(70))
    print("="*70)
    
    inertias = []
    silhouette_scores = []
    
    print(f"\nTesting K from {min(k_range)} to {max(k_range)}...\n")
    
    for k in k_range:
        # Train K-means
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_sample)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_sample, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        print(f"K={k:2d} | Inertia: {inertia:10.2f} | Silhouette: {silhouette:.4f}")
    
    # Find optimal K based on silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    
    print(f"\n🏆 Optimal K: {optimal_k} (Silhouette: {best_silhouette:.4f})")
    print("="*70)
    
    return optimal_k, inertias, silhouette_scores, k_range

def train_kmeans(X_sample, n_clusters):
    """
    Train final K-means model with optimal K
    """
    print("\n" + "="*70)
    print(f"TRAINING K-MEANS (K={n_clusters})".center(70))
    print("="*70)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_sample)
    
    # Cluster distribution
    print(f"\n📊 Cluster Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"   Cluster {cluster}: {count:5d} samples ({pct:5.1f}%)")
    
    # Final silhouette score
    silhouette = silhouette_score(X_sample, labels)
    print(f"\n✅ K-means Silhouette Score: {silhouette:.4f}")
    print("="*70)
    
    return kmeans, labels

# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================
def test_dbscan_parameters(X_sample, eps_values=[1.0, 2.0, 3.0, 5.0, 7.0]):
    """
    Test different DBSCAN parameters to find optimal settings
    """
    print("\n" + "="*70)
    print("DBSCAN: PARAMETER TESTING".center(70))
    print("="*70)
    
    print(f"\nTesting eps values: {eps_values}\n")
    
    results = []
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels = dbscan.fit_predict(X_sample)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_pct = (n_noise / len(labels)) * 100
        
        results.append({
            'eps': eps,
            'clusters': n_clusters,
            'noise_pct': noise_pct
        })
        
        print(f"eps={eps:.1f} | Clusters: {n_clusters:2d} | Noise: {noise_pct:5.1f}%")
    
    print("\n💡 Choose eps with 10-30% noise and 2-5 clusters")
    print("="*70)
    
    return results

def train_dbscan(X_sample, eps=5.0, min_samples=10):
    """
    Train DBSCAN with specified parameters
    """
    print("\n" + "="*70)
    print(f"TRAINING DBSCAN (eps={eps}, min_samples={min_samples})".center(70))
    print("="*70)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_sample)
    
    # Count clusters and noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\n📊 DBSCAN Results:")
    print(f"   Number of clusters: {n_clusters}")
    print(f"   Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    # Cluster distribution
    print(f"\n📊 Cluster Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        pct = count / len(labels) * 100
        if cluster == -1:
            print(f"   Noise:      {count:5d} samples ({pct:5.1f}%)")
        else:
            print(f"   Cluster {cluster}: {count:5d} samples ({pct:5.1f}%)")
    
    # Calculate silhouette (excluding noise)
    if n_clusters > 1 and n_noise < len(labels):
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            silhouette = silhouette_score(X_sample[non_noise_mask], labels[non_noise_mask])
            print(f"\n✅ DBSCAN Silhouette Score: {silhouette:.4f} (excluding noise)")
        else:
            silhouette = 0
            print("\n⚠️  All points classified as noise")
    else:
        silhouette = 0
        print("\n⚠️  Cannot calculate silhouette (need at least 2 clusters)")
    
    print("="*70)
    
    return dbscan, labels, silhouette

# ============================================================================
# PCA FOR VISUALIZATION
# ============================================================================
def reduce_dimensions_pca(X_sample, n_components=2):
    """
    Reduce dimensionality using PCA for visualization
    """
    print("\n" + "="*70)
    print("PCA DIMENSIONALITY REDUCTION".center(70))
    print("="*70)
    
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_sample)
    
    print(f"\n📊 PCA Results:")
    print(f"   Components: {n_components}")
    print(f"   Explained variance:")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"      PC{i}: {var:.4f} ({var*100:.2f}%)")
    print(f"   Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
    
    print("="*70)
    
    return X_pca, pca

# ============================================================================
# COMPLETE CLUSTERING PIPELINE
# ============================================================================
def run_clustering_analysis(X_train, sample_size=5000):
    """
    Complete unsupervised learning pipeline
    """
    print("\n" + "="*70)
    print("UNSUPERVISED LEARNING - CLUSTERING ANALYSIS".center(70))
    print("="*70)
    
    # Step 1: Prepare data
    X_sample, scaler = prepare_clustering_data(X_train, sample_size)
    
    # Step 2: K-Means clustering
    optimal_k, inertias, silhouette_scores, k_range = find_optimal_k(X_sample)
    kmeans, kmeans_labels = train_kmeans(X_sample, optimal_k)
    
    # Step 3: DBSCAN clustering
    dbscan_results = test_dbscan_parameters(X_sample)
    dbscan, dbscan_labels, dbscan_silhouette = train_dbscan(X_sample, eps=5.0)
    
    # Step 4: PCA for visualization
    X_pca, pca = reduce_dimensions_pca(X_sample)
    
    # Package results
    results = {
        'X_sample': X_sample,
        'X_pca': X_pca,
        'pca': pca,
        'kmeans': kmeans,
        'kmeans_labels': kmeans_labels,
        'optimal_k': optimal_k,
        'k_range': k_range,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'dbscan': dbscan,
        'dbscan_labels': dbscan_labels,
        'dbscan_silhouette': dbscan_silhouette,
        'scaler': scaler
    }
    
    print("\n" + "="*70)
    print("CLUSTERING ANALYSIS COMPLETE!".center(70))
    print("="*70)
    
    return results