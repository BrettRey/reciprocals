#!/usr/bin/env python3
"""
Test with adequate sample size (10 test items) to assess statistical power
"""

import math
import random

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def calculate_centroid(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def permutation_test_with_power(group1_items, group2_items, test_items, n_permutations=1000):
    """
    Permutation test with adequate sample size
    """
    
    test_vectors = [vec for name, vec in test_items]
    all_comparison = group1_items + group2_items
    
    # Calculate observed test statistic
    group1_vecs = [vec for name, vec in group1_items]
    group2_vecs = [vec for name, vec in group2_items]
    
    centroid1 = calculate_centroid(group1_vecs)
    centroid2 = calculate_centroid(group2_vecs)
    
    observed_group1_count = 0
    effect_sizes = []
    
    for name, vec in test_items:
        dist1 = euclidean_distance(vec, centroid1)
        dist2 = euclidean_distance(vec, centroid2)
        closer_to_group1 = dist1 < dist2
        
        if closer_to_group1:
            observed_group1_count += 1
            
        # Effect size: relative difference in distances
        effect_size = abs(dist1 - dist2) / max(dist1, dist2)
        effect_sizes.append(effect_size)
    
    mean_effect_size = sum(effect_sizes) / len(effect_sizes)
    
    # Permutation test
    null_counts = []
    
    for perm in range(n_permutations):
        random.seed(42 + perm)
        
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
    
    # One-tailed p-value
    n_test = len(test_items)
    if observed_group1_count >= n_test / 2:
        extreme_count = sum(1 for x in null_counts if x >= observed_group1_count)
    else:
        extreme_count = sum(1 for x in null_counts if x <= observed_group1_count)
    
    p_value = extreme_count / n_permutations
    
    return {
        'observed_count': observed_group1_count,
        'total_items': n_test,
        'proportion': observed_group1_count / n_test,
        'p_value': p_value,
        'mean_effect_size': mean_effect_size,
        'null_distribution': null_counts
    }

def simulate_with_power():
    """Test with 10 items instead of 2"""
    
    print("=== POWER TEST WITH 10 TEST ITEMS ===\n")
    
    # Test 1: Strong signal - should detect
    print("Test 1: Strong signal (10 items clearly prefer group1)")
    print("-" * 60)
    
    random.seed(42)
    
    # Create distinct groups
    group1_center = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # "fused determinatives"
    group2_center = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  # "pronouns"
    
    # Comparison groups
    group1_items = []
    for i in range(6):
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in group1_center]
        group1_items.append((f"det_{i}", vec))
    
    group2_items = []
    for i in range(6):
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in group2_center]
        group2_items.append((f"pronoun_{i}", vec))
    
    # 10 test items that should clearly prefer group1
    test_items_strong = []
    for i in range(10):
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in group1_center]
        test_items_strong.append((f"test_{i}", vec))
    
    results_strong = permutation_test_with_power(group1_items, group2_items, test_items_strong)
    
    print(f"Observed: {results_strong['observed_count']}/{results_strong['total_items']} prefer group1 ({results_strong['proportion']:.1%})")
    print(f"P-value: {results_strong['p_value']:.3f}")
    print(f"Mean effect size: {results_strong['mean_effect_size']:.3f}")
    print(f"Significant (p < 0.05): {results_strong['p_value'] < 0.05}")
    
    # Test 2: No signal - should not detect
    print(f"\nTest 2: No signal (10 items from mixed distribution)")
    print("-" * 60)
    
    random.seed(123)
    mixed_center = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # All from same distribution
    null_group1 = []
    for i in range(6):
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        null_group1.append((f"null1_{i}", vec))
    
    null_group2 = []
    for i in range(6):
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        null_group2.append((f"null2_{i}", vec))
    
    test_items_null = []
    for i in range(10):
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        test_items_null.append((f"test_null_{i}", vec))
    
    results_null = permutation_test_with_power(null_group1, null_group2, test_items_null)
    
    print(f"Observed: {results_null['observed_count']}/{results_null['total_items']} prefer group1 ({results_null['proportion']:.1%})")
    print(f"P-value: {results_null['p_value']:.3f}")
    print(f"Mean effect size: {results_null['mean_effect_size']:.3f}")
    print(f"Significant (p < 0.05): {results_null['p_value'] < 0.05}")
    
    # Test 3: Medium signal - borderline
    print(f"\nTest 3: Medium signal (10 items with moderate preference)")
    print("-" * 60)
    
    random.seed(456)
    
    # Test items with moderate preference for group1
    test_items_medium = []
    for i in range(7):  # 7 prefer group1
        vec = [max(0, min(1, x + random.gauss(0, 0.15))) for x in group1_center]
        test_items_medium.append((f"test_med_{i}", vec))
    
    for i in range(3):  # 3 prefer group2
        vec = [max(0, min(1, x + random.gauss(0, 0.15))) for x in group2_center]
        test_items_medium.append((f"test_med_{i+7}", vec))
    
    results_medium = permutation_test_with_power(group1_items, group2_items, test_items_medium)
    
    print(f"Observed: {results_medium['observed_count']}/{results_medium['total_items']} prefer group1 ({results_medium['proportion']:.1%})")
    print(f"P-value: {results_medium['p_value']:.3f}")
    print(f"Mean effect size: {results_medium['mean_effect_size']:.3f}")
    print(f"Significant (p < 0.05): {results_medium['p_value'] < 0.05}")
    
    print(f"\n=== POWER ANALYSIS SUMMARY ===")
    print(f"Strong signal detected: {results_strong['p_value'] < 0.05} (should be True)")
    print(f"Null signal detected: {results_null['p_value'] < 0.05} (should be False)")  
    print(f"Medium signal detected: {results_medium['p_value'] < 0.05} (may vary)")
    print(f"\nMethod has adequate power: {results_strong['p_value'] < 0.05 and not results_null['p_value'] < 0.05}")
    
    # Show null distributions for comparison
    print(f"\nNull distribution stats:")
    print(f"Strong signal null mean: {sum(results_strong['null_distribution'])/len(results_strong['null_distribution']):.1f}")
    print(f"No signal null mean: {sum(results_null['null_distribution'])/len(results_null['null_distribution']):.1f}")
    print(f"Medium signal null mean: {sum(results_medium['null_distribution'])/len(results_medium['null_distribution']):.1f}")
    
    return results_strong, results_null, results_medium

if __name__ == "__main__":
    simulate_with_power()