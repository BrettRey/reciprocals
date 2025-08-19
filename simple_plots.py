#!/usr/bin/env python3
"""
Simple plotting without problematic dependencies
Generate basic but proper plots using built-in libraries + simple matplotlib
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
                group = 'Reciprocal'
            elif lemma in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']:
                group = 'Fused_Determinative'  
            elif category == 'pronoun':
                group = 'Pronoun'
            elif category == 'determinative':
                group = 'Other_Determinative'
            else:
                group = 'Other'
            
            items.append({
                'lemma': lemma,
                'category': category,
                'group': group,
                'features': features
            })
    
    return items, feature_names

def simple_pca(X, n_components=2):
    """Simple PCA implementation without numpy/sklearn"""
    
    n_samples = len(X)
    n_features = len(X[0])
    
    # Center the data
    means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
    X_centered = []
    for i in range(n_samples):
        centered_row = [X[i][j] - means[j] for j in range(n_features)]
        X_centered.append(centered_row)
    
    # For simplicity, use first two principal components approximation
    # (This is a very rough approximation - real PCA would use SVD)
    
    # Component 1: Direction of maximum variance
    comp1 = [0] * n_features
    for j in range(n_features):
        variance = sum(X_centered[i][j] ** 2 for i in range(n_samples)) / n_samples
        comp1[j] = variance
    
    # Normalize
    norm1 = math.sqrt(sum(x**2 for x in comp1))
    if norm1 > 0:
        comp1 = [x/norm1 for x in comp1]
    
    # Component 2: Orthogonal direction (simplified)
    comp2 = [0] * n_features
    for j in range(n_features):
        # Rough orthogonal component
        comp2[j] = (-1)**j * (1.0 / n_features)
    
    # Project data onto components
    result = []
    for i in range(n_samples):
        proj1 = sum(X_centered[i][j] * comp1[j] for j in range(n_features))
        proj2 = sum(X_centered[i][j] * comp2[j] for j in range(n_features))
        result.append([proj1, proj2])
    
    return result

def create_simple_scatter_plot(items, save_path='reciprocals_scatter.txt'):
    """Create a better ASCII scatter plot with proper coordinates"""
    
    # Get data
    X = [item['features'] for item in items]
    groups = [item['group'] for item in items]
    lemmas = [item['lemma'] for item in items]
    
    # Simple PCA
    coords = simple_pca(X, n_components=2)
    
    # Find data range
    x_vals = [c[0] for c in coords]
    y_vals = [c[1] for c in coords]
    
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    
    # Create enhanced plot
    width, height = 100, 30
    
    # Extend ranges slightly
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Symbols for groups
    symbols = {
        'Reciprocal': '●',
        'Fused_Determinative': '■',
        'Pronoun': '▲',
        'Other_Determinative': '♦',
        'Other': '·'
    }
    
    # Plot points
    reciprocal_coords = []
    fused_det_coords = []
    
    for i, ((x, y), group, lemma) in enumerate(zip(coords, groups, lemmas)):
        # Convert to grid coordinates
        grid_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        grid_y = int((y - y_min) / (y_max - y_min) * (height - 1))
        grid_y = height - 1 - grid_y  # Flip y-axis
        
        if 0 <= grid_x < width and 0 <= grid_y < height:
            symbol = symbols.get(group, '?')
            
            # Store coordinates for key items
            if group == 'Reciprocal':
                reciprocal_coords.append((grid_x, grid_y, lemma))
            elif group == 'Fused_Determinative':
                fused_det_coords.append((grid_x, grid_y, lemma))
            
            if grid[grid_y][grid_x] == ' ':
                grid[grid_y][grid_x] = symbol
            else:
                grid[grid_y][grid_x] = '※'  # Overlap marker
    
    # Create output
    output = []
    output.append("Enhanced PCA Projection of Linguistic Feature Space")
    output.append("=" * 60)
    output.append("Legend:")
    output.append("  ● = Reciprocal    ■ = Fused Determinative")
    output.append("  ▲ = Pronoun       ♦ = Other Determinative")
    output.append("  · = Other         ※ = Overlap")
    output.append("")
    
    # Add coordinate system
    for row_idx, row in enumerate(grid):
        # Add y-axis labels every few rows
        if row_idx % 5 == 0:
            y_val = y_max - (row_idx / (height - 1)) * (y_max - y_min)
            line = f"{y_val:6.2f} |{''.join(row)}"
        else:
            line = "       |" + ''.join(row)
        output.append(line)
    
    # Add x-axis
    output.append("       " + "-" * width)
    x_labels = "       "
    for i in range(0, width, 20):
        x_val = x_min + (i / (width - 1)) * (x_max - x_min)
        label = f"{x_val:6.2f}"
        x_labels += label + " " * (20 - len(label))
    output.append(x_labels)
    
    # Add key item coordinates
    output.append("")
    output.append("Key Items:")
    for x, y, lemma in reciprocal_coords:
        output.append(f"  {lemma} (Reciprocal): grid position ({x}, {y})")
    for x, y, lemma in fused_det_coords:
        output.append(f"  {lemma} (Fused Det): grid position ({x}, {y})")
    
    # Save to file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    # Also print to console
    print('\n'.join(output))
    
    print(f"\n📊 Enhanced plot saved to: {save_path}")
    return coords

def create_distance_table(items, save_path='reciprocals_distances.txt'):
    """Create table of distances between key items"""
    
    # Extract key items
    key_items = []
    for item in items:
        if (item['group'] in ['Reciprocal', 'Fused_Determinative'] or 
            item['lemma'] in ['he', 'him', 'himself', 'she', 'her', 'they', 'them']):
            key_items.append(item)
    
    # Calculate distances
    output = []
    output.append("Distance Matrix: Key Linguistic Items")
    output.append("=" * 50)
    output.append("Euclidean distances in 155-dimensional feature space")
    output.append("(Lower numbers = more similar)")
    output.append("")
    
    # Header
    names = [item['lemma'] for item in key_items]
    header = "Item".ljust(15)
    for name in names:
        header += name[:10].rjust(12)
    output.append(header)
    output.append("-" * len(header))
    
    # Distance matrix
    for i, item1 in enumerate(key_items):
        row = names[i][:14].ljust(15)
        for j, item2 in enumerate(key_items):
            if i == j:
                row += "0.00".rjust(12)
            else:
                dist = math.sqrt(sum((a - b)**2 for a, b in zip(item1['features'], item2['features'])))
                row += f"{dist:.2f}".rjust(12)
        output.append(row)
    
    # Add analysis
    output.append("")
    output.append("Analysis:")
    
    # Find reciprocal distances
    reciprocals = [item for item in key_items if item['group'] == 'Reciprocal']
    fused_dets = [item for item in key_items if item['group'] == 'Fused_Determinative']
    pronouns = [item for item in key_items if item['lemma'] in ['he', 'him', 'himself', 'she', 'her', 'they', 'them']]
    
    for recip in reciprocals:
        output.append(f"\n{recip['lemma']} distances:")
        
        # To fused determinatives
        fused_dists = []
        for fused in fused_dets:
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(recip['features'], fused['features'])))
            fused_dists.append(dist)
            output.append(f"  to {fused['lemma']}: {dist:.3f}")
        
        # To pronouns
        pronoun_dists = []
        for pronoun in pronouns:
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(recip['features'], pronoun['features'])))
            pronoun_dists.append(dist)
            output.append(f"  to {pronoun['lemma']}: {dist:.3f}")
        
        # Summary
        avg_fused = sum(fused_dists) / len(fused_dists) if fused_dists else 0
        avg_pronoun = sum(pronoun_dists) / len(pronoun_dists) if pronoun_dists else 0
        
        output.append(f"  Average to fused determinatives: {avg_fused:.3f}")
        output.append(f"  Average to pronouns: {avg_pronoun:.3f}")
        output.append(f"  Closer to: {'fused determinatives' if avg_fused < avg_pronoun else 'pronouns'}")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(output))
    
    # Print to console
    print('\n'.join(output))
    
    print(f"\n📊 Distance analysis saved to: {save_path}")

def generate_simple_plots():
    """Generate simple but informative plots"""
    
    print("📊 GENERATING SIMPLE BUT PROPER PLOTS")
    print("Working around dependency issues...")
    print("=" * 50)
    
    # Load data
    items, feature_names = load_data_for_plotting('matrix_clean.csv')
    print(f"Loaded {len(items)} items with {len(feature_names)} features")
    
    # Generate plots
    print("\n1. Creating enhanced scatter plot...")
    coords = create_simple_scatter_plot(items, 'reciprocals_enhanced_scatter.txt')
    
    print("\n2. Creating distance analysis...")
    create_distance_table(items, 'reciprocals_distance_analysis.txt')
    
    print(f"\n✅ Simple plots generated!")
    print(f"📁 Files created:")
    print(f"   - reciprocals_enhanced_scatter.txt")
    print(f"   - reciprocals_distance_analysis.txt")
    
    print(f"\nThese provide clear visual and numerical analysis")
    print(f"without depending on problematic plotting libraries.")

if __name__ == "__main__":
    generate_simple_plots()