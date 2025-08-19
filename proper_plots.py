#!/usr/bin/env python3
"""
Generate proper PNG/PDF plots for the reciprocals analysis
No more ASCII nonsense!
"""

import csv
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def load_data_for_plotting(filepath):
    """Load data with proper grouping for visualization"""
    
    items = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[2:]
        
        for row in reader:
            lemma = row[0]
            category = row[1]
            features = [float(x) for x in row[2:]]
            
            # Group items for plotting
            if lemma in ['each_other', 'one_another']:
                group = 'Reciprocal'
                color = 'red'
                marker = 'X'
                size = 100
            elif lemma in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']:
                group = 'Fused Determinative'
                color = 'blue'
                marker = 'o'
                size = 80
            elif category == 'pronoun':
                group = 'Pronoun'
                color = 'green'
                marker = 's'
                size = 40
            elif category == 'determinative':
                group = 'Other Determinative'
                color = 'orange'
                marker = '^'
                size = 40
            else:
                group = 'Other'
                color = 'gray'
                marker = '.'
                size = 20
            
            items.append({
                'lemma': lemma,
                'category': category,
                'group': group,
                'features': features,
                'color': color,
                'marker': marker,
                'size': size
            })
    
    return items, feature_names

def create_pca_plot(items, save_path='pca_projection.png'):
    """Create proper PCA plot with matplotlib"""
    
    # Prepare data
    X = np.array([item['features'] for item in items])
    groups = [item['group'] for item in items]
    colors = [item['color'] for item in items]
    markers = [item['marker'] for item in items]
    sizes = [item['size'] for item in items]
    lemmas = [item['lemma'] for item in items]
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each group separately for proper legend
    unique_groups = list(set(groups))
    group_colors = {'Reciprocal': 'red', 'Fused Determinative': 'blue', 
                   'Pronoun': 'green', 'Other Determinative': 'orange', 'Other': 'gray'}
    group_markers = {'Reciprocal': 'X', 'Fused Determinative': 'o', 
                    'Pronoun': 's', 'Other Determinative': '^', 'Other': '.'}
    group_sizes = {'Reciprocal': 100, 'Fused Determinative': 80, 
                  'Pronoun': 40, 'Other Determinative': 40, 'Other': 20}
    
    for group in unique_groups:
        mask = [g == group for g in groups]
        X_group = X_pca[mask]
        plt.scatter(X_group[:, 0], X_group[:, 1], 
                   c=group_colors[group], marker=group_markers[group], 
                   s=group_sizes[group], label=group, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add labels for reciprocals and fused determinatives
    for i, (lemma, group) in enumerate(zip(lemmas, groups)):
        if group in ['Reciprocal', 'Fused Determinative']:
            plt.annotate(lemma, (X_pca[i, 0], X_pca[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, ha='left')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA Projection of Linguistic Feature Space\nReciprocals Hypothesis Test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ PCA plot saved: {save_path} and PDF version")
    
    plt.show()
    
    return pca, X_pca

def create_tsne_plot(items, save_path='tsne_projection.png'):
    """Create t-SNE plot for alternative dimensionality reduction"""
    
    # Prepare data
    X = np.array([item['features'] for item in items])
    groups = [item['group'] for item in items]
    lemmas = [item['lemma'] for item in items]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each group separately
    group_colors = {'Reciprocal': 'red', 'Fused Determinative': 'blue', 
                   'Pronoun': 'green', 'Other Determinative': 'orange', 'Other': 'gray'}
    group_markers = {'Reciprocal': 'X', 'Fused Determinative': 'o', 
                    'Pronoun': 's', 'Other Determinative': '^', 'Other': '.'}
    group_sizes = {'Reciprocal': 100, 'Fused Determinative': 80, 
                  'Pronoun': 40, 'Other Determinative': 40, 'Other': 20}
    
    unique_groups = list(set(groups))
    for group in unique_groups:
        mask = [g == group for g in groups]
        X_group = X_tsne[mask]
        plt.scatter(X_group[:, 0], X_group[:, 1], 
                   c=group_colors[group], marker=group_markers[group], 
                   s=group_sizes[group], label=group, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add labels for key items
    for i, (lemma, group) in enumerate(zip(lemmas, groups)):
        if group in ['Reciprocal', 'Fused Determinative']:
            plt.annotate(lemma, (X_tsne[i, 0], X_tsne[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, ha='left')
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Projection of Linguistic Feature Space\nReciprocals Hypothesis Test')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ t-SNE plot saved: {save_path} and PDF version")
    
    plt.show()

def create_distance_heatmap(items, save_path='distance_heatmap.png'):
    """Create heatmap of distances between key items"""
    
    # Extract key items
    key_items = []
    key_names = []
    
    for item in items:
        if (item['group'] in ['Reciprocal', 'Fused Determinative'] or 
            item['lemma'] in ['he', 'him', 'himself', 'she', 'her', 'they', 'them']):
            key_items.append(item['features'])
            key_names.append(item['lemma'])
    
    # Calculate distance matrix
    n = len(key_items)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = math.sqrt(sum((a - b)**2 for a, b in zip(key_items[i], key_items[j])))
                distance_matrix[i, j] = dist
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, 
                xticklabels=key_names, 
                yticklabels=key_names,
                annot=True, fmt='.2f', 
                cmap='viridis', 
                square=True)
    
    plt.title('Euclidean Distances Between Key Linguistic Items\n(Lower = More Similar)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Distance heatmap saved: {save_path} and PDF version")
    
    plt.show()

def create_feature_importance_plot(items, feature_names, save_path='feature_importance.png'):
    """Plot most discriminating features"""
    
    # Group items
    groups = {}
    for item in items:
        group = item['group']
        if group not in groups:
            groups[group] = []
        groups[group].append(item['features'])
    
    # Calculate between-group variance for each feature
    feature_variances = []
    
    for feat_idx in range(len(feature_names)):
        group_means = []
        for group_name, group_items in groups.items():
            if len(group_items) > 0:
                group_mean = sum(item[feat_idx] for item in group_items) / len(group_items)
                group_means.append(group_mean)
        
        if len(group_means) > 1:
            overall_mean = sum(group_means) / len(group_means)
            variance = sum((mean - overall_mean)**2 for mean in group_means) / len(group_means)
            feature_variances.append((feat_idx, variance, feature_names[feat_idx]))
    
    # Sort and get top features
    feature_variances.sort(key=lambda x: x[1], reverse=True)
    top_features = feature_variances[:20]  # Top 20
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_names_short = [name[:30] + '...' if len(name) > 30 else name 
                          for _, _, name in top_features]
    variances = [var for _, var, _ in top_features]
    
    bars = ax.barh(range(len(top_features)), variances)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(feature_names_short)
    ax.set_xlabel('Between-Group Variance')
    ax.set_title('Top 20 Most Discriminating Linguistic Features')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Color bars
    for i, bar in enumerate(bars):
        if 'each' in top_features[i][2].lower() or 'another' in top_features[i][2].lower():
            bar.set_color('red')
        elif any(word in top_features[i][2].lower() for word in ['any', 'some', 'thing', 'body']):
            bar.set_color('blue')
        else:
            bar.set_color('gray')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Feature importance plot saved: {save_path} and PDF version")
    
    plt.show()

def generate_all_plots():
    """Generate all proper plots for the analysis"""
    
    print("🎨 GENERATING PROPER PLOTS (PNG/PDF)")
    print("No more ASCII nonsense!")
    print("="*50)
    
    # Load data
    items, feature_names = load_data_for_plotting('matrix_clean.csv')
    print(f"Loaded {len(items)} items with {len(feature_names)} features")
    
    # Generate plots
    print("\n1. Creating PCA projection...")
    pca, X_pca = create_pca_plot(items, 'reciprocals_pca_projection.png')
    
    print("\n2. Creating t-SNE projection...")
    create_tsne_plot(items, 'reciprocals_tsne_projection.png')
    
    print("\n3. Creating distance heatmap...")
    create_distance_heatmap(items, 'reciprocals_distance_heatmap.png')
    
    print("\n4. Creating feature importance plot...")
    create_feature_importance_plot(items, feature_names, 'reciprocals_feature_importance.png')
    
    print(f"\n✅ All plots generated!")
    print(f"📁 Files created:")
    print(f"   - reciprocals_pca_projection.png/.pdf")
    print(f"   - reciprocals_tsne_projection.png/.pdf") 
    print(f"   - reciprocals_distance_heatmap.png/.pdf")
    print(f"   - reciprocals_feature_importance.png/.pdf")
    
    return pca, X_pca

if __name__ == "__main__":
    generate_all_plots()