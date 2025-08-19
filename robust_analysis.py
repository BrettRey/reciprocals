#!/usr/bin/env python3
"""
Robust analysis method combining the best fixes
"""

import math
import random

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def calculate_centroid(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def robust_classification_test(group1_items, group2_items, test_items, 
                               n_permutations=1000, alpha=0.05):
    """
    Robust test combining permutation and effect size
    
    Args:
        group1_items: List of (name, vector) for first comparison group
        group2_items: List of (name, vector) for second comparison group  
        test_items: List of (name, vector) for items to classify
        n_permutations: Number of permutation tests
        alpha: Significance level
    
    Returns:
        Dictionary with results and interpretation
    """
    
    # Extract vectors
    test_vectors = [vec for name, vec in test_items]
    all_comparison = group1_items + group2_items
    all_comp_vectors = [vec for name, vec in all_comparison]
    
    # Calculate observed test statistic
    group1_vecs = [vec for name, vec in group1_items]
    group2_vecs = [vec for name, vec in group2_items]
    
    centroid1 = calculate_centroid(group1_vecs)
    centroid2 = calculate_centroid(group2_vecs)
    
    observed_group1_count = 0
    observed_distances = []
    
    for name, vec in test_items:
        dist1 = euclidean_distance(vec, centroid1)
        dist2 = euclidean_distance(vec, centroid2)
        closer_to_group1 = dist1 < dist2
        
        if closer_to_group1:
            observed_group1_count += 1
            
        observed_distances.append({
            'name': name,
            'dist_to_group1': dist1,
            'dist_to_group2': dist2,
            'closer_to_group1': closer_to_group1,
            'effect_size': abs(dist1 - dist2) / max(dist1, dist2)  # Relative difference
        })
    
    # Permutation test
    null_counts = []
    
    for perm in range(n_permutations):
        random.seed(42 + perm)
        
        # Randomly reassign group labels
        shuffled = all_comparison.copy()
        random.shuffle(shuffled)
        
        perm_group1 = shuffled[:len(group1_items)]
        perm_group2 = shuffled[len(group1_items):]
        
        perm_group1_vecs = [vec for name, vec in perm_group1]
        perm_group2_vecs = [vec for name, vec in perm_group2]
        
        perm_centroid1 = calculate_centroid(perm_group1_vecs)
        perm_centroid2 = calculate_centroid(perm_group2_vecs)
        
        perm_group1_count = 0
        for vec in test_vectors:
            perm_dist1 = euclidean_distance(vec, perm_centroid1)
            perm_dist2 = euclidean_distance(vec, perm_centroid2)
            if perm_dist1 < perm_dist2:
                perm_group1_count += 1
        
        null_counts.append(perm_group1_count)
    
    # Calculate p-value (one-tailed, since we're testing a specific hypothesis)
    n_test = len(test_items)
    # Count how often we get as extreme or more extreme than observed
    if observed_group1_count >= n_test / 2:
        # Testing if group1 is preferred
        extreme_as_observed = sum(1 for x in null_counts if x >= observed_group1_count)
    else:
        # Testing if group2 is preferred  
        extreme_as_observed = sum(1 for x in null_counts if x <= observed_group1_count)
    
    p_value = extreme_as_observed / n_permutations
    
    # Effect size assessment
    mean_effect_size = sum(d['effect_size'] for d in observed_distances) / len(observed_distances)
    
    # Results summary
    results = {
        'observed_pattern': f"{observed_group1_count}/{n_test} items closer to group1",
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_effect_size': mean_effect_size,
        'effect_interpretation': 'large' if mean_effect_size > 0.3 else 'medium' if mean_effect_size > 0.1 else 'small',
        'individual_results': observed_distances,
        'null_distribution': null_counts,
        'interpretation': None
    }
    
    # Interpretation
    if not results['significant']:
        results['interpretation'] = "No significant difference from random chance"
    elif results['effect_interpretation'] == 'small':
        results['interpretation'] = "Significant but small effect - weak evidence"
    else:
        results['interpretation'] = f"Significant with {results['effect_interpretation']} effect size - strong evidence"
    
    return results

def test_on_simulated_data():
    """Test the robust method on simulated data"""
    
    print("=== ROBUST ANALYSIS TEST ===\n")
    
    # Test 1: Strong signal (should detect)
    print("Test 1: Strong signal (reciprocals are determinatives)")
    print("-" * 60)
    
    random.seed(42)
    
    # Clear separation
    det_center = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    pronoun_center = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    
    fused_dets = []
    for i, name in enumerate(['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']):
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in det_center]
        fused_dets.append((name, vec))
    
    pronouns = []
    for name in ['he', 'him', 'himself', 'she', 'her']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in pronoun_center]
        pronouns.append((name, vec))
    
    # Reciprocals similar to determinatives
    reciprocals = []
    for name in ['each_other', 'one_another']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in det_center]
        reciprocals.append((name, vec))
    
    results1 = robust_classification_test(fused_dets, pronouns, reciprocals)
    
    print(f"Pattern: {results1['observed_pattern']}")
    print(f"P-value: {results1['p_value']:.3f}")
    print(f"Effect size: {results1['mean_effect_size']:.3f} ({results1['effect_interpretation']})")
    print(f"Interpretation: {results1['interpretation']}")
    
    # Test 2: No signal (should not detect)
    print(f"\nTest 2: No signal (null hypothesis)")
    print("-" * 60)
    
    random.seed(123)
    mixed_center = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # All from same distribution
    null_group1 = []
    for i in range(6):
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        null_group1.append((f"item1_{i}", vec))
    
    null_group2 = []
    for i in range(5):
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        null_group2.append((f"item2_{i}", vec))
    
    null_reciprocals = []
    for name in ['each_other', 'one_another']:
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        null_reciprocals.append((name, vec))
    
    results2 = robust_classification_test(null_group1, null_group2, null_reciprocals)
    
    print(f"Pattern: {results2['observed_pattern']}")
    print(f"P-value: {results2['p_value']:.3f}")
    print(f"Effect size: {results2['mean_effect_size']:.3f} ({results2['effect_interpretation']})")
    print(f"Interpretation: {results2['interpretation']}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Strong signal detected: {results1['significant']} (should be True)")
    print(f"Null signal detected: {results2['significant']} (should be False)")
    print(f"Method working correctly: {results1['significant'] and not results2['significant']}")
    
    return results1, results2

if __name__ == "__main__":
    test_on_simulated_data()