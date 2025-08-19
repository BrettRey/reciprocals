#!/usr/bin/env python3
"""
Real data analysis: Testing reciprocals hypothesis using robust permutation method
"""

import csv
import math
import random

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def calculate_centroid(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def load_linguistic_data(filepath):
    """Load the linguistic feature matrix"""
    
    items = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            lemma = row[0]
            category = row[1]
            # Convert feature values to floats, skip first 2 columns (lemma, class)
            features = [float(x) for x in row[2:]]
            items.append({
                'lemma': lemma,
                'category': category,
                'features': features
            })
    
    return items, header[2:]  # Return items and feature names

def extract_comparison_groups(items):
    """Extract the specific groups for comparison"""
    
    # Fused determinatives (your comparison group)
    fused_det_names = ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']
    fused_dets = [item for item in items if item['lemma'] in fused_det_names]
    
    # Regular pronouns (for contrast)
    pronoun_names = ['he', 'him', 'himself', 'she', 'her', 'herself', 'they', 'them', 'themselves']
    pronouns = [item for item in items if item['lemma'] in pronoun_names and item['category'] == 'pronoun']
    
    # Reciprocals (test items)
    reciprocals = [item for item in items if item['lemma'] in ['each_other', 'one_another']]
    
    print(f"Found {len(fused_dets)} fused determinatives: {[item['lemma'] for item in fused_dets]}")
    print(f"Found {len(pronouns)} pronouns: {[item['lemma'] for item in pronouns]}")
    print(f"Found {len(reciprocals)} reciprocals: {[item['lemma'] for item in reciprocals]}")
    
    return fused_dets, pronouns, reciprocals

def permutation_test_reciprocals(fused_dets, pronouns, reciprocals, n_permutations=1000):
    """
    Test if reciprocals cluster with fused determinatives vs pronouns
    """
    
    if len(reciprocals) != 2:
        print(f"Warning: Expected 2 reciprocals, found {len(reciprocals)}")
    
    # Extract feature vectors
    fused_det_vecs = [item['features'] for item in fused_dets]
    pronoun_vecs = [item['features'] for item in pronouns]
    reciprocal_vecs = [item['features'] for item in reciprocals]
    
    # Calculate observed pattern
    fused_det_centroid = calculate_centroid(fused_det_vecs)
    pronoun_centroid = calculate_centroid(pronoun_vecs)
    
    observed_closer_to_fused = 0
    individual_results = []
    
    for item in reciprocals:
        vec = item['features']
        dist_to_fused = euclidean_distance(vec, fused_det_centroid)
        dist_to_pronoun = euclidean_distance(vec, pronoun_centroid)
        closer_to_fused = dist_to_fused < dist_to_pronoun
        
        if closer_to_fused:
            observed_closer_to_fused += 1
        
        effect_size = abs(dist_to_fused - dist_to_pronoun) / max(dist_to_fused, dist_to_pronoun)
        
        individual_results.append({
            'lemma': item['lemma'],
            'dist_to_fused': dist_to_fused,
            'dist_to_pronoun': dist_to_pronoun,
            'closer_to_fused': closer_to_fused,
            'effect_size': effect_size
        })
    
    mean_effect_size = sum(r['effect_size'] for r in individual_results) / len(individual_results)
    
    # Permutation test
    all_comparison = fused_dets + pronouns
    null_closer_to_fused_counts = []
    
    for perm in range(n_permutations):
        random.seed(42 + perm)
        
        # Randomly reassign items to "fused" vs "pronoun" groups
        shuffled = all_comparison.copy()
        random.shuffle(shuffled)
        
        perm_fused_group = shuffled[:len(fused_dets)]
        perm_pronoun_group = shuffled[len(fused_dets):]
        
        perm_fused_vecs = [item['features'] for item in perm_fused_group]
        perm_pronoun_vecs = [item['features'] for item in perm_pronoun_group]
        
        perm_fused_centroid = calculate_centroid(perm_fused_vecs)
        perm_pronoun_centroid = calculate_centroid(perm_pronoun_vecs)
        
        perm_closer_to_fused = 0
        for vec in reciprocal_vecs:
            perm_dist_fused = euclidean_distance(vec, perm_fused_centroid)
            perm_dist_pronoun = euclidean_distance(vec, perm_pronoun_centroid)
            if perm_dist_fused < perm_dist_pronoun:
                perm_closer_to_fused += 1
        
        null_closer_to_fused_counts.append(perm_closer_to_fused)
    
    # Calculate p-value (one-tailed)
    n_reciprocals = len(reciprocals)
    if observed_closer_to_fused >= n_reciprocals / 2:
        # Testing if reciprocals prefer fused determinatives
        extreme_count = sum(1 for x in null_closer_to_fused_counts if x >= observed_closer_to_fused)
    else:
        # Testing if reciprocals prefer pronouns
        extreme_count = sum(1 for x in null_closer_to_fused_counts if x <= observed_closer_to_fused)
    
    p_value = extreme_count / n_permutations
    
    return {
        'observed_closer_to_fused': observed_closer_to_fused,
        'total_reciprocals': n_reciprocals,
        'proportion_fused': observed_closer_to_fused / n_reciprocals,
        'p_value': p_value,
        'mean_effect_size': mean_effect_size,
        'individual_results': individual_results,
        'null_distribution': null_closer_to_fused_counts,
        'fused_det_centroid_distance': euclidean_distance(fused_det_centroid, pronoun_centroid)
    }

def interpret_results(results, strong_prior_prob=0.85):
    """
    Interpret results considering the strong prior against the hypothesis
    """
    
    interpretation = {
        'pattern': f"{results['observed_closer_to_fused']}/{results['total_reciprocals']} reciprocals closer to fused determinatives",
        'evidence_strength': 'weak',
        'bayesian_conclusion': 'insufficient evidence to overcome prior'
    }
    
    # Evidence strength based on effect size and p-value
    if results['mean_effect_size'] > 0.5 and results['p_value'] < 0.01:
        interpretation['evidence_strength'] = 'very strong'
    elif results['mean_effect_size'] > 0.3 and results['p_value'] < 0.05:
        interpretation['evidence_strength'] = 'strong'
    elif results['mean_effect_size'] > 0.2 and results['p_value'] < 0.1:
        interpretation['evidence_strength'] = 'moderate'
    elif results['mean_effect_size'] > 0.1:
        interpretation['evidence_strength'] = 'weak'
    else:
        interpretation['evidence_strength'] = 'very weak'
    
    # Bayesian interpretation considering strong prior
    if interpretation['evidence_strength'] in ['very strong', 'strong']:
        interpretation['bayesian_conclusion'] = 'evidence may overcome strong prior - hypothesis worth considering'
    elif interpretation['evidence_strength'] == 'moderate':
        interpretation['bayesian_conclusion'] = 'evidence weakens but does not overcome strong prior'
    else:
        interpretation['bayesian_conclusion'] = 'insufficient evidence to overcome strong prior from Cambridge Grammar'
    
    return interpretation

def run_reciprocals_analysis():
    """Main analysis function"""
    
    print("=== RECIPROCALS HYPOTHESIS TEST ===")
    print("Testing: Do reciprocals pattern with fused determinatives rather than pronouns?")
    print("Strong prior: Cambridge Grammar classifies them as pronouns (85% confidence)")
    print("=" * 80)
    
    # Load data
    items, feature_names = load_linguistic_data('matrix_clean.csv')
    print(f"\nLoaded {len(items)} linguistic items with {len(feature_names)} features")
    
    # Extract comparison groups
    fused_dets, pronouns, reciprocals = extract_comparison_groups(items)
    
    if len(fused_dets) == 0 or len(pronouns) == 0 or len(reciprocals) == 0:
        print("Error: Missing required comparison groups")
        return None
    
    print(f"\nComparison groups:")
    print(f"- Fused determinatives: {len(fused_dets)} items")
    print(f"- Pronouns: {len(pronouns)} items")  
    print(f"- Reciprocals (test items): {len(reciprocals)} items")
    
    # Run permutation test
    print(f"\nRunning permutation test (1000 permutations)...")
    results = permutation_test_reciprocals(fused_dets, pronouns, reciprocals)
    
    # Display results
    print(f"\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    print(f"Observed pattern: {results['observed_closer_to_fused']}/{results['total_reciprocals']} reciprocals closer to fused determinatives")
    print(f"Proportion preferring fused determinatives: {results['proportion_fused']:.1%}")
    print(f"P-value (permutation test): {results['p_value']:.3f}")
    print(f"Mean effect size: {results['mean_effect_size']:.3f}")
    print(f"Distance between group centroids: {results['fused_det_centroid_distance']:.3f}")
    
    print(f"\nIndividual reciprocal results:")
    for result in results['individual_results']:
        print(f"  {result['lemma']}:")
        print(f"    Distance to fused determinatives: {result['dist_to_fused']:.3f}")
        print(f"    Distance to pronouns: {result['dist_to_pronoun']:.3f}")
        print(f"    Closer to fused determinatives: {result['closer_to_fused']}")
        print(f"    Effect size: {result['effect_size']:.3f}")
    
    # Interpretation
    interpretation = interpret_results(results)
    
    print(f"\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    print(f"Evidence strength: {interpretation['evidence_strength']}")
    print(f"Bayesian conclusion: {interpretation['bayesian_conclusion']}")
    
    # Null distribution summary
    null_mean = sum(results['null_distribution']) / len(results['null_distribution'])
    null_extreme = sum(1 for x in results['null_distribution'] if x == 0 or x == len(reciprocals))
    
    print(f"\nNull distribution statistics:")
    print(f"  Mean under random assignment: {null_mean:.1f}/{len(reciprocals)}")
    print(f"  Extreme results (0 or {len(reciprocals)}): {null_extreme}/{len(results['null_distribution'])} ({null_extreme/len(results['null_distribution'])*100:.1f}%)")
    
    return results, interpretation

if __name__ == "__main__":
    results, interpretation = run_reciprocals_analysis()