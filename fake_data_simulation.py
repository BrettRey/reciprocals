#!/usr/bin/env python3
"""
Gelman-style fake data simulation for reciprocals hypothesis testing
Before looking at real data, simulate what we expect under each hypothesis
"""

import math
import random

def simulate_under_hypothesis_1():
    """
    Hypothesis 1: Reciprocals ARE fused determinatives
    Simulate: each_other and one_another should be very similar to 
    {someone, anyone, anything, everything, somebody, anybody}
    """
    
    # Simulate feature vectors for fused determinatives (high similarity)
    random.seed(42)
    n_features = 10  # simplified for simulation
    
    # Fused determinatives cluster (high internal similarity)
    fused_det_center = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    fused_dets = []
    for name in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']:
        # Add small random noise around the center
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in fused_det_center]
        fused_dets.append((name, vec))
    
    # Reciprocals should be very similar to fused determinatives
    reciprocal_vecs = []
    for name in ['each_other', 'one_another']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in fused_det_center]
        reciprocal_vecs.append((name, vec))
    
    # Regular pronouns should be different
    pronoun_center = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    pronouns = []
    for name in ['he', 'him', 'himself', 'she', 'her']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in pronoun_center]
        pronouns.append((name, vec))
    
    return fused_dets, reciprocal_vecs, pronouns

def simulate_under_hypothesis_2():
    """
    Hypothesis 2: Reciprocals ARE pronouns (Cambridge Grammar is right)
    Simulate: each_other and one_another should be similar to 
    {he, him, himself, etc.}
    """
    
    random.seed(42)
    n_features = 10
    
    # Fused determinatives cluster
    fused_det_center = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    fused_dets = []
    for name in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in fused_det_center]
        fused_dets.append((name, vec))
    
    # Pronoun cluster (including reciprocals)
    pronoun_center = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    
    # Regular pronouns
    pronouns = []
    for name in ['he', 'him', 'himself', 'she', 'her']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in pronoun_center]
        pronouns.append((name, vec))
    
    # Reciprocals should be similar to pronouns
    reciprocal_vecs = []
    for name in ['each_other', 'one_another']:
        vec = [max(0, min(1, x + random.gauss(0, 0.1))) for x in pronoun_center]
        reciprocal_vecs.append((name, vec))
    
    return fused_dets, reciprocal_vecs, pronouns

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

def calculate_centroid(vectors):
    """Calculate centroid of a list of vectors"""
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]

def calculate_similarities(fused_dets, reciprocal_vecs, pronouns):
    """Calculate similarity metrics"""
    
    # Extract just the vectors
    fused_det_vecs = [vec for _, vec in fused_dets]
    pronoun_vecs = [vec for _, vec in pronouns]
    
    # Calculate centroids
    fused_det_centroid = calculate_centroid(fused_det_vecs)
    pronoun_centroid = calculate_centroid(pronoun_vecs)
    
    # Distance from each reciprocal to each centroid
    results = {}
    for name, vec in reciprocal_vecs:
        dist_to_fused = euclidean_distance(vec, fused_det_centroid)
        dist_to_pronoun = euclidean_distance(vec, pronoun_centroid)
        
        results[name] = {
            'dist_to_fused_det': dist_to_fused,
            'dist_to_pronoun': dist_to_pronoun,
            'closer_to_fused': dist_to_fused < dist_to_pronoun
        }
    
    return results

def simulate_null_hypothesis():
    """
    NULL HYPOTHESIS: No real difference - both reciprocals and comparison groups 
    are drawn from the same underlying distribution. Test if we find false differences.
    """
    random.seed(123)  # Different seed for null test
    
    # All items drawn from same mixed distribution (no true clusters)
    mixed_center = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    # "Fused determinatives" - really just random noise around center
    fused_dets = []
    for name in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']:
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        fused_dets.append((name, vec))
    
    # "Pronouns" - same distribution  
    pronouns = []
    for name in ['he', 'him', 'himself', 'she', 'her']:
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        pronouns.append((name, vec))
    
    # "Reciprocals" - also same distribution
    reciprocal_vecs = []
    for name in ['each_other', 'one_another']:
        vec = [max(0, min(1, x + random.gauss(0, 0.2))) for x in mixed_center]
        reciprocal_vecs.append((name, vec))
    
    return fused_dets, reciprocal_vecs, pronouns

def run_simulation():
    """Run all simulations and check if our method can distinguish them"""
    
    print("=== FAKE DATA SIMULATION ===")
    print("Testing if our analysis can distinguish between hypotheses\n")
    
    print("Hypothesis 1: Reciprocals are fused determinatives")
    print("-" * 50)
    fused_dets_h1, reciprocals_h1, pronouns_h1 = simulate_under_hypothesis_1()
    results_h1 = calculate_similarities(fused_dets_h1, reciprocals_h1, pronouns_h1)
    
    for name, metrics in results_h1.items():
        print(f"{name}: dist_to_fused={metrics['dist_to_fused_det']:.3f}, "
              f"dist_to_pronoun={metrics['dist_to_pronoun']:.3f}, "
              f"closer_to_fused={metrics['closer_to_fused']}")
    
    print("\nHypothesis 2: Reciprocals are pronouns (Cambridge Grammar)")
    print("-" * 50)
    fused_dets_h2, reciprocals_h2, pronouns_h2 = simulate_under_hypothesis_2()
    results_h2 = calculate_similarities(fused_dets_h2, reciprocals_h2, pronouns_h2)
    
    for name, metrics in results_h2.items():
        print(f"{name}: dist_to_fused={metrics['dist_to_fused_det']:.3f}, "
              f"dist_to_pronoun={metrics['dist_to_pronoun']:.3f}, "
              f"closer_to_fused={metrics['closer_to_fused']}")
    
    print("\n=== SIMULATION RESULTS ===")
    print("Under Hypothesis 1 (reciprocals are fused determinatives):")
    h1_fused_wins = sum(1 for r in results_h1.values() if r['closer_to_fused'])
    print(f"  {h1_fused_wins}/2 reciprocals closer to fused determinatives")
    
    print("Under Hypothesis 2 (reciprocals are pronouns):")
    h2_pronoun_wins = sum(1 for r in results_h2.values() if not r['closer_to_fused'])
    print(f"  {h2_pronoun_wins}/2 reciprocals closer to pronouns")
    
    print(f"\nOur method can distinguish: {h1_fused_wins == 2 and h2_pronoun_wins == 2}")
    
    # Test for over-sensitivity (false positives)
    print("\n" + "="*50)
    print("NULL HYPOTHESIS TEST (checking for over-sensitivity)")
    print("All groups drawn from same distribution - should find NO pattern")
    print("-" * 50)
    
    fused_dets_null, reciprocals_null, pronouns_null = simulate_null_hypothesis()
    results_null = calculate_similarities(fused_dets_null, reciprocals_null, pronouns_null)
    
    for name, metrics in results_null.items():
        print(f"{name}: dist_to_fused={metrics['dist_to_fused_det']:.3f}, "
              f"dist_to_pronoun={metrics['dist_to_pronoun']:.3f}, "
              f"closer_to_fused={metrics['closer_to_fused']}")
    
    null_fused_wins = sum(1 for r in results_null.values() if r['closer_to_fused'])
    print(f"\nUnder NULL (no real difference): {null_fused_wins}/2 reciprocals closer to fused dets")
    print(f"Expected under null: ~1/2 (random chance)")
    print(f"Over-sensitive? {null_fused_wins == 2 or null_fused_wins == 0}")
    
    return results_h1, results_h2, results_null

if __name__ == "__main__":
    run_simulation()