#!/usr/bin/env python3
"""
Realistic simulation with 50 words and 50 features
"""

import math
import random

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def calculate_centroid(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def generate_realistic_groups(n_features=50):
    """
    Generate groups with realistic feature patterns
    """
    random.seed(42)
    
    # Create more realistic feature patterns
    # Group 1: "Fused determinatives" - pattern in first half of features
    group1_pattern = []
    for i in range(n_features):
        if i < n_features // 2:
            # First half: mostly 1s with some noise
            group1_pattern.append(0.8 + random.gauss(0, 0.1))
        else:
            # Second half: mostly 0s with some noise  
            group1_pattern.append(0.2 + random.gauss(0, 0.1))
    
    # Group 2: "Pronouns" - opposite pattern
    group2_pattern = []
    for i in range(n_features):
        if i < n_features // 2:
            # First half: mostly 0s
            group2_pattern.append(0.2 + random.gauss(0, 0.1))
        else:
            # Second half: mostly 1s
            group2_pattern.append(0.8 + random.gauss(0, 0.1))
    
    # Clip to [0,1] range
    group1_pattern = [max(0, min(1, x)) for x in group1_pattern]
    group2_pattern = [max(0, min(1, x)) for x in group2_pattern]
    
    return group1_pattern, group2_pattern

def create_word_vectors(base_pattern, n_words, noise_level=0.15, name_prefix="word"):
    """
    Create word vectors based on a base pattern with noise
    """
    words = []
    for i in range(n_words):
        vec = [max(0, min(1, x + random.gauss(0, noise_level))) for x in base_pattern]
        words.append((f"{name_prefix}_{i}", vec))
    return words

def permutation_test_realistic(group1_items, group2_items, test_items, n_permutations=1000):
    """
    Permutation test for realistic simulation
    """
    
    test_vectors = [vec for name, vec in test_items]
    all_comparison = group1_items + group2_items
    
    # Calculate observed test statistic
    group1_vecs = [vec for name, vec in group1_items]
    group2_vecs = [vec for name, vec in group2_items]
    
    centroid1 = calculate_centroid(group1_vecs)
    centroid2 = calculate_centroid(group2_vecs)
    
    observed_group1_count = 0
    individual_results = []
    
    for name, vec in test_items:
        dist1 = euclidean_distance(vec, centroid1)
        dist2 = euclidean_distance(vec, centroid2)
        closer_to_group1 = dist1 < dist2
        
        if closer_to_group1:
            observed_group1_count += 1
            
        effect_size = abs(dist1 - dist2) / max(dist1, dist2) if max(dist1, dist2) > 0 else 0
        individual_results.append({
            'name': name,
            'dist1': dist1,
            'dist2': dist2,
            'closer_to_group1': closer_to_group1,
            'effect_size': effect_size
        })
    
    mean_effect_size = sum(r['effect_size'] for r in individual_results) / len(individual_results)
    
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
    
    # Calculate p-value
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
        'null_distribution': null_counts,
        'individual_results': individual_results
    }

def run_realistic_simulation():
    """Run simulation with 50 words and 50 features"""
    
    print("=== REALISTIC SIMULATION: 50 words × 50 features ===\n")
    
    # Generate base patterns
    group1_pattern, group2_pattern = generate_realistic_groups(n_features=50)
    
    # Test 1: Strong signal - test items strongly resemble group1
    print("Test 1: Strong signal (test items clearly resemble group1)")
    print("-" * 70)
    
    random.seed(42)
    
    # Create comparison groups (20 words each)
    group1_words = create_word_vectors(group1_pattern, 20, noise_level=0.1, name_prefix="fused_det")
    group2_words = create_word_vectors(group2_pattern, 20, noise_level=0.1, name_prefix="pronoun")
    
    # Create 10 test items that strongly resemble group1
    test_items_strong = create_word_vectors(group1_pattern, 10, noise_level=0.1, name_prefix="reciprocal_strong")
    
    results_strong = permutation_test_realistic(group1_words, group2_words, test_items_strong)
    
    print(f"Observed: {results_strong['observed_count']}/{results_strong['total_items']} prefer group1 ({results_strong['proportion']:.1%})")
    print(f"P-value: {results_strong['p_value']:.3f}")
    print(f"Mean effect size: {results_strong['mean_effect_size']:.3f}")
    print(f"Significant (p < 0.05): {results_strong['p_value'] < 0.05}")
    
    # Test 2: No signal - test items from mixed distribution
    print(f"\nTest 2: No signal (test items from mixed distribution)")
    print("-" * 70)
    
    random.seed(123)
    
    # Mixed pattern (average of group1 and group2)
    mixed_pattern = [(g1 + g2) / 2 for g1, g2 in zip(group1_pattern, group2_pattern)]
    
    # Create comparison groups from same mixed distribution
    mixed_group1 = create_word_vectors(mixed_pattern, 20, noise_level=0.2, name_prefix="mixed1")
    mixed_group2 = create_word_vectors(mixed_pattern, 20, noise_level=0.2, name_prefix="mixed2")
    
    # Test items also from mixed distribution
    test_items_null = create_word_vectors(mixed_pattern, 10, noise_level=0.2, name_prefix="reciprocal_null")
    
    results_null = permutation_test_realistic(mixed_group1, mixed_group2, test_items_null)
    
    print(f"Observed: {results_null['observed_count']}/{results_null['total_items']} prefer group1 ({results_null['proportion']:.1%})")
    print(f"P-value: {results_null['p_value']:.3f}")
    print(f"Mean effect size: {results_null['mean_effect_size']:.3f}")
    print(f"Significant (p < 0.05): {results_null['p_value'] < 0.05}")
    
    # Test 3: Medium signal - test items somewhat resemble group1
    print(f"\nTest 3: Medium signal (test items moderately resemble group1)")
    print("-" * 70)
    
    random.seed(456)
    
    # Create pattern that's 70% group1, 30% group2
    medium_pattern = [0.7 * g1 + 0.3 * g2 for g1, g2 in zip(group1_pattern, group2_pattern)]
    
    test_items_medium = create_word_vectors(medium_pattern, 10, noise_level=0.15, name_prefix="reciprocal_medium")
    
    results_medium = permutation_test_realistic(group1_words, group2_words, test_items_medium)
    
    print(f"Observed: {results_medium['observed_count']}/{results_medium['total_items']} prefer group1 ({results_medium['proportion']:.1%})")
    print(f"P-value: {results_medium['p_value']:.3f}")
    print(f"Mean effect size: {results_medium['mean_effect_size']:.3f}")
    print(f"Significant (p < 0.05): {results_medium['p_value'] < 0.05}")
    
    print(f"\n=== REALISTIC SIMULATION SUMMARY ===")
    print(f"Strong signal detected: {results_strong['p_value'] < 0.05} (should be True)")
    print(f"Null signal detected: {results_null['p_value'] < 0.05} (should be False)")  
    print(f"Medium signal detected: {results_medium['p_value'] < 0.05} (may vary)")
    print(f"\nMethod performance: {results_strong['p_value'] < 0.05 and not results_null['p_value'] < 0.05}")
    
    # Show separation between groups
    print(f"\nGroup separation metrics:")
    group1_centroid = calculate_centroid([vec for name, vec in group1_words])
    group2_centroid = calculate_centroid([vec for name, vec in group2_words])
    inter_group_distance = euclidean_distance(group1_centroid, group2_centroid)
    print(f"Distance between group centroids: {inter_group_distance:.3f}")
    
    # Show null distribution stats
    print(f"\nNull distribution stats:")
    print(f"Strong signal - null mean: {sum(results_strong['null_distribution'])/len(results_strong['null_distribution']):.1f}/10")
    print(f"No signal - null mean: {sum(results_null['null_distribution'])/len(results_null['null_distribution']):.1f}/10")
    print(f"Medium signal - null mean: {sum(results_medium['null_distribution'])/len(results_medium['null_distribution']):.1f}/10")
    
    return results_strong, results_null, results_medium

if __name__ == "__main__":
    run_realistic_simulation()