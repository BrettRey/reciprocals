#!/usr/bin/env python3
"""
Fixed version addressing over-sensitivity issues
Multiple approaches to test for robustness
"""

import math
import random

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def calculate_centroid(vectors):
    """Calculate centroid of a list of vectors"""
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def permutation_test(reciprocal_vecs, all_comparison_items, n_permutations=1000):
    """
    Fix 1: Permutation test to establish null distribution
    Randomly assign comparison items to "fused" vs "pronoun" groups many times
    """
    
    reciprocal_names = [name for name, vec in reciprocal_vecs]
    reciprocal_vectors = [vec for name, vec in reciprocal_vecs]
    
    # All comparison vectors
    all_vectors = [vec for name, vec in all_comparison_items]
    
    observed_scores = []
    null_scores = []
    
    for perm in range(n_permutations):
        random.seed(42 + perm)  # Reproducible but different each time
        
        # Randomly split comparison items into two groups
        shuffled = all_comparison_items.copy()
        random.shuffle(shuffled)
        
        group1 = shuffled[:6]  # "fused determinatives"
        group2 = shuffled[6:]  # "pronouns"
        
        group1_vecs = [vec for name, vec in group1]
        group2_vecs = [vec for name, vec in group2]
        
        centroid1 = calculate_centroid(group1_vecs)
        centroid2 = calculate_centroid(group2_vecs)
        
        # For each reciprocal, count how many are closer to group1
        closer_to_group1 = 0
        for vec in reciprocal_vectors:
            dist1 = euclidean_distance(vec, centroid1)
            dist2 = euclidean_distance(vec, centroid2)
            if dist1 < dist2:
                closer_to_group1 += 1
        
        if perm == 0:
            # First permutation uses the "true" assignment
            observed_scores.append(closer_to_group1)
        else:
            null_scores.append(closer_to_group1)
    
    return observed_scores[0], null_scores

def bootstrap_test(fused_dets, reciprocal_vecs, pronouns, n_bootstrap=1000):
    """
    Fix 2: Bootstrap resampling to equalize group sizes and assess uncertainty
    """
    
    fused_det_vecs = [vec for name, vec in fused_dets]
    pronoun_vecs = [vec for name, vec in pronouns]
    reciprocal_vectors = [vec for name, vec in reciprocal_vecs]
    
    min_size = min(len(fused_det_vecs), len(pronoun_vecs))
    
    bootstrap_results = []
    
    for boot in range(n_bootstrap):
        random.seed(100 + boot)
        
        # Resample to equal sizes
        fused_sample = random.sample(fused_det_vecs, min_size)
        pronoun_sample = random.sample(pronoun_vecs, min_size)
        
        centroid_fused = calculate_centroid(fused_sample)
        centroid_pronoun = calculate_centroid(pronoun_sample)
        
        closer_to_fused = 0
        for vec in reciprocal_vectors:
            dist_fused = euclidean_distance(vec, centroid_fused)
            dist_pronoun = euclidean_distance(vec, centroid_pronoun)
            if dist_fused < dist_pronoun:
                closer_to_fused += 1
        
        bootstrap_results.append(closer_to_fused)
    
    return bootstrap_results

def standardized_distance_test(fused_dets, reciprocal_vecs, pronouns):
    """
    Fix 3: Standardize features to prevent some dimensions dominating
    """
    
    # Collect all vectors to compute overall mean and std
    all_vecs = [vec for name, vec in fused_dets + reciprocal_vecs + pronouns]
    
    # Calculate mean and std for each dimension
    n_dims = len(all_vecs[0])
    means = [sum(vec[i] for vec in all_vecs) / len(all_vecs) for i in range(n_dims)]
    stds = []
    for i in range(n_dims):
        variance = sum((vec[i] - means[i]) ** 2 for vec in all_vecs) / len(all_vecs)
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)
    
    # Standardize all vectors
    def standardize(vec):
        return [(vec[i] - means[i]) / stds[i] for i in range(len(vec))]
    
    fused_std = [standardize(vec) for name, vec in fused_dets]
    pronoun_std = [standardize(vec) for name, vec in pronouns]
    reciprocal_std = [standardize(vec) for name, vec in reciprocal_vecs]
    
    centroid_fused = calculate_centroid(fused_std)
    centroid_pronoun = calculate_centroid(pronoun_std)
    
    results = {}
    for i, (name, vec) in enumerate(reciprocal_vecs):
        vec_std = reciprocal_std[i]
        dist_fused = euclidean_distance(vec_std, centroid_fused)
        dist_pronoun = euclidean_distance(vec_std, centroid_pronoun)
        
        results[name] = {
            'dist_to_fused_det': dist_fused,
            'dist_to_pronoun': dist_pronoun,
            'closer_to_fused': dist_fused < dist_pronoun
        }
    
    return results

def simulate_null_hypothesis_equal_groups():
    """
    Improved null simulation with equal group sizes
    """
    random.seed(123)
    
    # All items from same distribution
    mixed_center = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # Create exactly equal-sized groups
    all_comparison = []
    for i in range(10):  # 5 "fused" + 5 "pronouns"
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        all_comparison.append((f"item_{i}", vec))
    
    fused_dets = all_comparison[:5]
    pronouns = all_comparison[5:]
    
    # Reciprocals from same distribution
    reciprocal_vecs = []
    for name in ['each_other', 'one_another']:
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        reciprocal_vecs.append((name, vec))
    
    return fused_dets, reciprocal_vecs, pronouns

def run_fixed_tests():
    """Test all fixes"""
    
    print("=== TESTING FIXES FOR OVER-SENSITIVITY ===\n")
    
    # Generate null hypothesis data with equal groups
    fused_null, reciprocals_null, pronouns_null = simulate_null_hypothesis_equal_groups()
    
    print("Fix 1: Equal group sizes + standardized distances")
    print("-" * 50)
    results_std = standardized_distance_test(fused_null, reciprocals_null, pronouns_null)
    
    closer_to_fused_count = sum(1 for r in results_std.values() if r['closer_to_fused'])
    for name, metrics in results_std.items():
        print(f"{name}: dist_to_fused={metrics['dist_to_fused_det']:.3f}, "
              f"dist_to_pronoun={metrics['dist_to_pronoun']:.3f}, "
              f"closer_to_fused={metrics['closer_to_fused']}")
    
    print(f"Result: {closer_to_fused_count}/2 closer to fused (expected ~1 under null)")
    
    print("\nFix 2: Bootstrap uncertainty assessment")
    print("-" * 50)
    bootstrap_results = bootstrap_test(fused_null, reciprocals_null, pronouns_null, n_bootstrap=100)
    mean_bootstrap = sum(bootstrap_results) / len(bootstrap_results)
    
    # Count how often we get extreme results (0 or 2)
    extreme_results = sum(1 for x in bootstrap_results if x == 0 or x == 2)
    print(f"Mean across bootstrap samples: {mean_bootstrap:.2f}/2")
    print(f"Extreme results (0 or 2): {extreme_results}/100 times ({extreme_results}%)")
    print(f"Expected under null: ~50% extreme results by chance")
    
    # All comparison items for permutation test
    all_comparison = fused_null + pronouns_null
    
    print("\nFix 3: Permutation test")
    print("-" * 50)
    observed, null_dist = permutation_test(reciprocals_null, all_comparison, n_permutations=100)
    null_extreme = sum(1 for x in null_dist if x == 0 or x == 2)
    
    print(f"Observed: {observed}/2 closer to 'fused' group")
    print(f"Null distribution extreme results: {null_extreme}/99 times ({null_extreme:.1f}%)")
    print(f"P-value approximation: {null_extreme/99:.3f}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Standardized distance: {closer_to_fused_count}/2 (better if closer to 1)")
    print(f"Bootstrap extremes: {extreme_results}% (better if closer to 50%)")
    print(f"Permutation extremes: {null_extreme:.1f}% (better if closer to 50%)")
    
    return results_std, bootstrap_results, null_dist

if __name__ == "__main__":
    run_fixed_tests()