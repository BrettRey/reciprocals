#!/usr/bin/env python3
"""
Corrected analysis: Looking at genuine linguistic features beyond morphology
"""

import csv

def load_and_analyze_features(filepath):
    """Load data and examine different types of features"""
    
    items = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[2:]
        
        for row in reader:
            lemma = row[0]
            category = row[1] 
            features = [float(x) for x in row[2:]]
            
            items.append({
                'lemma': lemma,
                'category': category,
                'features': features
            })
    
    return items, feature_names

def categorize_features(feature_names):
    """Categorize features into morphological, semantic, syntactic"""
    
    morphological = []
    semantic = []
    syntactic = []
    
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        
        # Morphological (word composition)
        if any(word in name_lower for word in ['each', 'other', 'one', 'another', 'any', 'some', 'thing', 'body', 'where', 'what', 'who', 'which', 'whose', 'him', 'her', 'his', 'their', 'our', 'my', 'your']):
            morphological.append((i, name))
        
        # Semantic features
        elif any(word in name_lower for word in ['definite', 'indefinite', 'universal', 'existential', 'negative', 'interrogative', 'reciprocal', 'reflexive', 'anaphoric', 'deictic', 'person', 'gender', 'dual', 'plural', 'singular', 'countable', 'temporal', 'locative', 'degree']):
            semantic.append((i, name))
        
        # Syntactic features  
        elif any(word in name_lower for word in ['function', 'appears', 'subject', 'object', 'determiner', 'head', 'modifier', 'complement', 'predicate', 'coordination', 'paradigm', 'inflect', 'case', 'genitive', 'accusative', 'nominative']):
            syntactic.append((i, name))
        
        else:
            # Catch remaining features
            if 'can_be' in name_lower or 'takes_' in name_lower or 'selects_' in name_lower or 'compatible_' in name_lower:
                syntactic.append((i, name))
            else:
                semantic.append((i, name))  # Default to semantic
    
    return morphological, semantic, syntactic

def analyze_reciprocals_by_feature_type(items, feature_names):
    """Analyze how reciprocals pattern across different feature types"""
    
    print("=== CORRECTED FEATURE ANALYSIS ===")
    print("Understanding reciprocal patterns across linguistic feature types")
    print("=" * 60)
    
    # Get reciprocals and comparison groups
    reciprocals = [item for item in items if item['lemma'] in ['each_other', 'one_another']]
    fused_dets = [item for item in items if item['lemma'] in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']]
    regular_pronouns = [item for item in items if item['category'] == 'pronoun' and item['lemma'] not in ['each_other', 'one_another']]
    
    print(f"Reciprocals: {len(reciprocals)}")
    print(f"Fused determinatives: {len(fused_dets)}")  
    print(f"Regular pronouns: {len(regular_pronouns)}")
    
    # Categorize features
    morphological, semantic, syntactic = categorize_features(feature_names)
    
    print(f"\nFeature categories:")
    print(f"  Morphological: {len(morphological)}")
    print(f"  Semantic: {len(semantic)}")
    print(f"  Syntactic: {len(syntactic)}")
    
    # Analyze each feature type
    for feature_type, feature_list in [('MORPHOLOGICAL', morphological), 
                                     ('SEMANTIC', semantic), 
                                     ('SYNTACTIC', syntactic)]:
        
        print(f"\n{'='*50}")
        print(f"{feature_type} FEATURES")
        print('='*50)
        
        if not feature_list:
            print("No features in this category")
            continue
            
        # Calculate average distances in this feature subspace
        feature_indices = [idx for idx, name in feature_list]
        
        # Extract feature subsets
        recip_vecs = [[item['features'][i] for i in feature_indices] for item in reciprocals]
        fused_vecs = [[item['features'][i] for i in feature_indices] for item in fused_dets]
        pronoun_vecs = [[item['features'][i] for i in feature_indices] for item in regular_pronouns[:10]]  # Sample
        
        # Calculate centroids
        def calculate_centroid(vecs):
            if not vecs:
                return None
            dim = len(vecs[0])
            return [sum(vec[i] for vec in vecs) / len(vecs) for i in range(dim)]
        
        def euclidean_distance(v1, v2):
            return sum((a - b)**2 for a, b in zip(v1, v2))**0.5
        
        fused_centroid = calculate_centroid(fused_vecs)
        pronoun_centroid = calculate_centroid(pronoun_vecs)
        
        if fused_centroid and pronoun_centroid:
            for recip in reciprocals:
                recip_vec = [recip['features'][i] for i in feature_indices]
                
                dist_to_fused = euclidean_distance(recip_vec, fused_centroid)
                dist_to_pronoun = euclidean_distance(recip_vec, pronoun_centroid)
                
                print(f"\n{recip['lemma']}:")
                print(f"  Distance to fused determinatives: {dist_to_fused:.3f}")
                print(f"  Distance to pronouns: {dist_to_pronoun:.3f}")
                print(f"  Closer to: {'fused determinatives' if dist_to_fused < dist_to_pronoun else 'pronouns'}")
                print(f"  Difference: {abs(dist_to_fused - dist_to_pronoun):.3f}")
        
        # Show some example features in this category
        print(f"\nExample {feature_type.lower()} features:")
        for i, (idx, name) in enumerate(feature_list[:5]):
            print(f"  {name}")
            
            # Show values for key items
            for item_type, items_list in [('Reciprocals', reciprocals), 
                                        ('Fused_Dets', fused_dets[:3]),
                                        ('Pronouns', regular_pronouns[:3])]:
                values = [item['features'][idx] for item in items_list]
                if values:
                    avg_val = sum(values) / len(values)
                    print(f"    {item_type}: {avg_val:.3f}")

def examine_key_syntactic_features(items, feature_names):
    """Look specifically at syntactic features that test your hypothesis"""
    
    print(f"\n{'='*60}")
    print("KEY SYNTACTIC FEATURES FOR RECIPROCAL HYPOTHESIS")
    print('='*60)
    
    # Key features for your theory
    key_features = [
        'functions_as_fused_determiner_head',
        'appears_in_subject', 
        'inflects_for_case',
        'paradigm_has_distinct_acc_form',
        'functions_as_predicative_complement',
        'accepts_genitive_apstrophe_s'
    ]
    
    reciprocals = [item for item in items if item['lemma'] in ['each_other', 'one_another']]
    fused_dets = [item for item in items if item['lemma'] in ['someone', 'anyone', 'anything', 'everything', 'somebody', 'anybody']]
    regular_pronouns = [item for item in items if item['category'] == 'pronoun' and item['lemma'] not in ['each_other', 'one_another']]
    
    for feature_name in key_features:
        if feature_name in feature_names:
            feat_idx = feature_names.index(feature_name)
            
            print(f"\n{feature_name}:")
            
            # Get values for each group
            recip_vals = [item['features'][feat_idx] for item in reciprocals]
            fused_vals = [item['features'][feat_idx] for item in fused_dets]
            pronoun_vals = [item['features'][feat_idx] for item in regular_pronouns]
            
            print(f"  Reciprocals: {recip_vals} (mean: {sum(recip_vals)/len(recip_vals):.3f})")
            print(f"  Fused determinatives: {fused_vals} (mean: {sum(fused_vals)/len(fused_vals):.3f})")
            print(f"  Regular pronouns: mean: {sum(pronoun_vals)/len(pronoun_vals):.3f}")
            
            # Which group do reciprocals match better?
            recip_mean = sum(recip_vals) / len(recip_vals)
            fused_mean = sum(fused_vals) / len(fused_vals)
            pronoun_mean = sum(pronoun_vals) / len(pronoun_vals)
            
            dist_to_fused = abs(recip_mean - fused_mean)
            dist_to_pronoun = abs(recip_mean - pronoun_mean)
            
            if dist_to_fused < dist_to_pronoun:
                print(f"  → Reciprocals closer to FUSED DETERMINATIVES")
            elif dist_to_pronoun < dist_to_fused:
                print(f"  → Reciprocals closer to PRONOUNS")
            else:
                print(f"  → Reciprocals equidistant")

if __name__ == "__main__":
    items, feature_names = load_and_analyze_features('matrix_clean.csv')
    analyze_reciprocals_by_feature_type(items, feature_names)
    examine_key_syntactic_features(items, feature_names)