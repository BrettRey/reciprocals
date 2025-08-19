#!/usr/bin/env python3
"""
Gelman-style data visualization for linguistic feature matrix
Focus: Understanding the data structure, not just confirming/rejecting hypotheses
"""

import csv
import math
import random

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
                group = 'reciprocal'
            elif lemma in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']:
                group = 'fused_determinative'  
            elif category == 'pronoun':
                group = 'pronoun'
            elif category == 'determinative':
                group = 'other_determinative'
            else:
                group = 'other'
            
            items.append({
                'lemma': lemma,
                'category': category,
                'group': group,
                'features': features
            })
    
    return items, feature_names

def pca_simple(X, n_components=2):
    """Simple PCA implementation for dimensionality reduction"""
    
    # Center the data
    n_samples = len(X)
    n_features = len(X[0])
    
    # Calculate means
    means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    
    # Center data
    X_centered = []
    for i in range(n_samples):
        centered_row = [X[i][j] - means[j] for j in range(n_features)]
        X_centered.append(centered_row)
    
    # Calculate covariance matrix (simplified)
    # For large feature matrices, we'll use a random projection approximation
    if n_features > 50:
        # Random projection to reduce computational complexity
        random.seed(42)
        projection_dim = min(20, n_features)
        projection_matrix = []
        for i in range(projection_dim):
            row = [random.gauss(0, 1) for _ in range(n_features)]
            norm = math.sqrt(sum(x*x for x in row))
            projection_matrix.append([x/norm for x in row])
        
        # Project data
        X_proj = []
        for i in range(n_samples):
            proj_row = []
            for j in range(projection_dim):
                val = sum(X_centered[i][k] * projection_matrix[j][k] for k in range(n_features))
                proj_row.append(val)
            X_proj.append(proj_row)
        
        X_centered = X_proj
        n_features = projection_dim
    
    # Simple 2D projection (using first two principal directions approximation)
    # This is a very simplified PCA - in real analysis you'd use proper SVD
    
    # For visualization, let's use the first two "synthetic" dimensions
    # that capture maximum variance in different directions
    
    # Direction 1: maximize variance
    var1 = [0] * n_features
    for j in range(n_features):
        var_j = sum(X_centered[i][j] ** 2 for i in range(n_samples)) / n_samples
        var1[j] = var_j
    
    # Normalize direction 1
    norm1 = math.sqrt(sum(x*x for x in var1))
    if norm1 > 0:
        var1 = [x/norm1 for x in var1]
    
    # Project onto direction 1
    proj1 = [sum(X_centered[i][j] * var1[j] for j in range(n_features)) for i in range(n_samples)]
    
    # Direction 2: orthogonal to direction 1, maximize remaining variance
    var2 = [0] * n_features
    for j in range(n_features):
        # Remove component along direction 1
        component1 = sum(X_centered[i][j] * var1[j] for i in range(n_samples)) / n_samples
        residual_var = sum((X_centered[i][j] - component1 * var1[j]) ** 2 for i in range(n_samples)) / n_samples
        var2[j] = residual_var
    
    # Normalize direction 2
    norm2 = math.sqrt(sum(x*x for x in var2))
    if norm2 > 0:
        var2 = [x/norm2 for x in var2]
    
    # Project onto direction 2
    proj2 = [sum(X_centered[i][j] * var2[j] for j in range(n_features)) for i in range(n_samples)]
    
    return list(zip(proj1, proj2))

def create_ascii_scatterplot(data_points, groups, title, width=80, height=20):
    """Create ASCII scatterplot - Gelman loves simple, clear plots"""
    
    # Find data range
    x_vals = [p[0] for p in data_points]
    y_vals = [p[1] for p in data_points]
    
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1 
    y_max += y_range * 0.1
    
    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    symbols = {
        'reciprocal': 'R',
        'fused_determinative': 'F', 
        'pronoun': 'P',
        'other_determinative': 'D',
        'other': '.'
    }
    
    for i, (x, y) in enumerate(data_points):
        # Convert to grid coordinates
        grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
        
        # Flip y-axis for display
        grid_y = height - 1 - grid_y
        
        if 0 <= grid_x < width and 0 <= grid_y < height:
            symbol = symbols.get(groups[i], '?')
            if grid[grid_y][grid_x] == ' ':
                grid[grid_y][grid_x] = symbol
            else:
                grid[grid_y][grid_x] = '*'  # Overlap
    
    # Print plot
    print(f"\n{title}")
    print("=" * len(title))
    print("Legend: R=Reciprocal, F=Fused_Det, P=Pronoun, D=Other_Det, .=Other")
    print()
    
    for row in grid:
        print(''.join(row))
    
    print(f"\nX-axis: {x_min:.2f} to {x_max:.2f}")
    print(f"Y-axis: {y_min:.2f} to {y_max:.2f}")

def analyze_feature_distributions(items, feature_names):
    """Gelman would look at individual feature distributions"""
    
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    print("Gelman: 'Let's understand what these features actually measure'")
    
    # Group items
    groups = {}
    for item in items:
        group = item['group']
        if group not in groups:
            groups[group] = []
        groups[group].append(item['features'])
    
    # Look at a few key features
    interesting_features = []
    
    # Find features with high variance between groups
    for feat_idx in range(min(10, len(feature_names))):  # Check first 10 features
        group_means = {}
        for group_name, group_items in groups.items():
            if len(group_items) > 0:
                group_means[group_name] = sum(item[feat_idx] for item in group_items) / len(group_items)
        
        if len(group_means) > 1:
            variance = sum((mean - sum(group_means.values())/len(group_means))**2 
                          for mean in group_means.values()) / len(group_means)
            interesting_features.append((feat_idx, variance, feature_names[feat_idx]))
    
    # Sort by variance and show top features
    interesting_features.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop discriminating features (by between-group variance):")
    for i, (feat_idx, variance, feat_name) in enumerate(interesting_features[:5]):
        print(f"\n{i+1}. {feat_name} (variance: {variance:.3f})")
        
        # Show means by group
        for group_name, group_items in groups.items():
            if len(group_items) > 0:
                mean_val = sum(item[feat_idx] for item in group_items) / len(group_items)
                print(f"   {group_name}: {mean_val:.3f} (n={len(group_items)})")

def reciprocal_detailed_comparison(items):
    """Detailed look at reciprocals vs comparison groups"""
    
    print("\n" + "="*60)
    print("RECIPROCAL DETAILED COMPARISON")
    print("="*60)
    print("Gelman: 'What exactly makes reciprocals different (or not)?'")
    
    # Extract groups
    reciprocals = [item for item in items if item['group'] == 'reciprocal']
    fused_dets = [item for item in items if item['group'] == 'fused_determinative']
    pronouns = [item for item in items if item['group'] == 'pronoun']
    
    print(f"\nReciprocals found: {[item['lemma'] for item in reciprocals]}")
    print(f"Fused determinatives: {[item['lemma'] for item in fused_dets]}")
    print(f"Pronouns: {len(pronouns)} items")
    
    # For each reciprocal, find nearest neighbors
    for reciprocal in reciprocals:
        print(f"\n--- {reciprocal['lemma']} ---")
        
        # Calculate distances to all other items
        distances = []
        for other_item in items:
            if other_item['lemma'] != reciprocal['lemma']:
                # Euclidean distance
                dist = math.sqrt(sum((a - b)**2 for a, b in 
                                   zip(reciprocal['features'], other_item['features'])))
                distances.append((dist, other_item))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        print("Nearest neighbors:")
        for i, (dist, neighbor) in enumerate(distances[:5]):
            print(f"  {i+1}. {neighbor['lemma']} ({neighbor['group']}) - distance: {dist:.3f}")
        
        # Count neighbors by group in top 10
        top_10 = distances[:10]
        group_counts = {}
        for _, neighbor in top_10:
            group = neighbor['group']
            group_counts[group] = group_counts.get(group, 0) + 1
        
        print(f"Top 10 neighbors by group: {group_counts}")

def save_plot_to_log(plot_content, plot_name):
    """Save plot output to log file"""
    
    log_filename = f"plot_log_{plot_name.lower().replace(' ', '_')}.txt"
    
    with open(log_filename, 'w') as f:
        f.write(f"# {plot_name}\n")
        f.write(f"Generated: 2025-08-18\n")
        f.write(f"Analysis: Reciprocals hypothesis testing\n\n")
        f.write(plot_content)
    
    print(f"📊 Plot saved to: {log_filename}")
    return log_filename

def gelman_exploratory_plots():
    """Main function - Gelman's exploratory data analysis approach"""
    
    print("=== GELMAN-STYLE EXPLORATORY DATA ANALYSIS ===")
    print("Focus: Understanding the data structure, not confirming hypotheses")
    print("Motto: 'Make plots. Make more plots. Make different plots.'")
    print("="*70)
    
    # Load data
    items, feature_names = load_data_for_plotting('matrix_clean.csv')
    print(f"\nLoaded {len(items)} items with {len(feature_names)} features")
    
    # Group summary
    group_counts = {}
    for item in items:
        group = item['group']
        group_counts[group] = group_counts.get(group, 0) + 1
    
    print("\nGroup sizes:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")
    
    # 1. PCA visualization
    print(f"\n{'='*60}")
    print("1. DIMENSIONALITY REDUCTION PLOT")
    print("='*60")
    print("Gelman: 'Let's see the overall structure in 2D'")
    
    X = [item['features'] for item in items]
    groups = [item['group'] for item in items]
    
    # Simple PCA
    pca_coords = pca_simple(X, n_components=2)
    
    # Capture plot output for logging
    import io
    import sys
    from contextlib import redirect_stdout
    
    plot_output = io.StringIO()
    with redirect_stdout(plot_output):
        create_ascii_scatterplot(pca_coords, groups, 
                               "PCA Projection of Linguistic Feature Space")
    
    plot_content = plot_output.getvalue()
    print(plot_content)  # Display to console
    
    # Save to log
    log_file = save_plot_to_log(plot_content, "PCA_Projection")
    
    # 2. Feature analysis
    analyze_feature_distributions(items, feature_names)
    
    # 3. Detailed reciprocal analysis
    reciprocal_detailed_comparison(items)
    
    print(f"\n{'='*60}")
    print("GELMAN'S SUMMARY")
    print("="*60)
    print("Key insights from exploratory analysis:")
    print("1. Look at the PCA plot - do you see natural clusters?")
    print("2. Check the feature distributions - what drives the differences?")
    print("3. Examine reciprocal neighbors - are they really intermediate?")
    print("\nGelman: 'The data should tell a story. What story does this tell?'")

if __name__ == "__main__":
    gelman_exploratory_plots()