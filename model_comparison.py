#!/usr/bin/env python3
"""
Model comparison: Which classification scheme fits better?
POST-HOC ANALYSIS (forking paths documented)
"""

import csv
import math
import random

def load_data(filepath):
    """Load linguistic data and separate reciprocals from others"""
    
    items = []
    reciprocals = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            lemma = row[0]
            category = row[1]
            features = [float(x) for x in row[2:]]
            
            item = {
                'lemma': lemma,
                'category': category,
                'features': features
            }
            
            if lemma in ['each_other', 'one_another']:
                reciprocals.append(item)
            else:
                items.append(item)
    
    return items, reciprocals, header[2:]

def prepare_training_data(items):
    """Prepare training data (non-reciprocals only)"""
    
    X = []  # Feature vectors
    y = []  # Labels (0 = determinative, 1 = pronoun)
    item_info = []
    
    for item in items:
        if item['category'] in ['determinative', 'pronoun']:
            X.append(item['features'])
            y.append(1 if item['category'] == 'pronoun' else 0)
            item_info.append({
                'lemma': item['lemma'],
                'category': item['category']
            })
    
    return X, y, item_info

def simple_logistic_regression(X, y, learning_rate=0.01, max_iterations=1000):
    """Simple logistic regression implementation"""
    
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    
    for iteration in range(max_iterations):
        # Forward pass
        total_loss = 0
        dw = [0.0] * n_features
        db = 0.0
        
        for i in range(len(X)):
            # Prediction
            z = sum(X[i][j] * weights[j] for j in range(n_features)) + bias
            z = max(-500, min(500, z))  # Prevent overflow
            prob = 1 / (1 + math.exp(-z))
            
            # Loss
            epsilon = 1e-15
            prob = max(epsilon, min(1 - epsilon, prob))
            loss = -(y[i] * math.log(prob) + (1 - y[i]) * math.log(1 - prob))
            total_loss += loss
            
            # Gradients
            error = prob - y[i]
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error
        
        # Update weights
        for j in range(n_features):
            weights[j] -= learning_rate * dw[j] / len(X)
        bias -= learning_rate * db / len(X)
        
        # Check convergence
        if iteration % 100 == 0:
            avg_loss = total_loss / len(X)
            if iteration > 0 and abs(avg_loss - prev_loss) < 1e-6:
                break
            prev_loss = avg_loss
    
    return weights, bias

def predict_probability(features, weights, bias):
    """Predict probability of being a pronoun"""
    z = sum(features[i] * weights[i] for i in range(len(features))) + bias
    z = max(-500, min(500, z))  # Prevent overflow
    return 1 / (1 + math.exp(-z))

def calculate_model_fit(X, y, weights, bias):
    """Calculate various model fit metrics"""
    
    total_log_likelihood = 0
    correct_predictions = 0
    
    for i in range(len(X)):
        prob = predict_probability(X[i], weights, bias)
        predicted_class = 1 if prob > 0.5 else 0
        
        # Accuracy
        if predicted_class == y[i]:
            correct_predictions += 1
        
        # Log-likelihood
        epsilon = 1e-15
        prob = max(epsilon, min(1 - epsilon, prob))
        log_likelihood = y[i] * math.log(prob) + (1 - y[i]) * math.log(1 - prob)
        total_log_likelihood += log_likelihood
    
    accuracy = correct_predictions / len(X)
    avg_log_likelihood = total_log_likelihood / len(X)
    
    return {
        'accuracy': accuracy,
        'log_likelihood': avg_log_likelihood,
        'n_samples': len(X)
    }

def cross_validate_model(X, y, k_folds=5):
    """Cross-validation to assess model stability"""
    
    # Simple k-fold cross validation
    fold_size = len(X) // k_folds
    accuracies = []
    log_likelihoods = []
    
    for fold in range(k_folds):
        # Split data
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < k_folds - 1 else len(X)
        
        X_test = X[start_idx:end_idx]
        y_test = y[start_idx:end_idx]
        X_train = X[:start_idx] + X[end_idx:]
        y_train = y[:start_idx] + y[end_idx:]
        
        # Train model
        weights, bias = simple_logistic_regression(X_train, y_train)
        
        # Test model
        metrics = calculate_model_fit(X_test, y_test, weights, bias)
        accuracies.append(metrics['accuracy'])
        log_likelihoods.append(metrics['log_likelihood'])
    
    return {
        'mean_accuracy': sum(accuracies) / len(accuracies),
        'std_accuracy': math.sqrt(sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) / len(accuracies)),
        'mean_log_likelihood': sum(log_likelihoods) / len(log_likelihoods),
        'std_log_likelihood': math.sqrt(sum((ll - sum(log_likelihoods)/len(log_likelihoods))**2 for ll in log_likelihoods) / len(log_likelihoods))
    }

def test_reciprocal_classification(X_train, y_train, reciprocals):
    """Test both classification schemes for reciprocals"""
    
    # Train model on non-reciprocal data
    weights, bias = simple_logistic_regression(X_train, y_train)
    
    results = {}
    
    for scenario in ['as_determinative', 'as_pronoun']:
        # Create test dataset with reciprocals
        X_test = X_train.copy()
        y_test = y_train.copy()
        
        # Add reciprocals with different labels
        for reciprocal in reciprocals:
            X_test.append(reciprocal['features'])
            if scenario == 'as_determinative':
                y_test.append(0)  # determinative = 0
            else:
                y_test.append(1)  # pronoun = 1
        
        # Calculate fit metrics
        metrics = calculate_model_fit(X_test, y_test, weights, bias)
        
        # Individual reciprocal predictions
        reciprocal_predictions = []
        for reciprocal in reciprocals:
            prob = predict_probability(reciprocal['features'], weights, bias)
            reciprocal_predictions.append({
                'lemma': reciprocal['lemma'],
                'prob_pronoun': prob,
                'prob_determinative': 1 - prob,
                'predicted_category': 'pronoun' if prob > 0.5 else 'determinative'
            })
        
        results[scenario] = {
            'metrics': metrics,
            'reciprocal_predictions': reciprocal_predictions
        }
    
    return results, weights, bias

def run_model_comparison():
    """Main model comparison analysis"""
    
    print("=== MODEL COMPARISON ANALYSIS ===")
    print("POST-HOC ANALYSIS: Testing reciprocal classification schemes")
    print("FORKING PATHS: Documented in analysis_preregistration.md")
    print("=" * 70)
    
    # Load data
    items, reciprocals, feature_names = load_data('matrix_clean.csv')
    print(f"\nLoaded {len(items)} non-reciprocal items and {len(reciprocals)} reciprocals")
    print(f"Features: {len(feature_names)} linguistic features")
    
    # Prepare training data (non-reciprocals only)
    X_train, y_train, train_info = prepare_training_data(items)
    
    # Count categories in training data
    n_determinatives = sum(1 for y in y_train if y == 0)
    n_pronouns = sum(1 for y in y_train if y == 1)
    print(f"\nTraining data: {n_determinatives} determinatives, {n_pronouns} pronouns")
    
    # Cross-validation on training data
    print(f"\nCross-validation performance on non-reciprocal data:")
    cv_results = cross_validate_model(X_train, y_train)
    print(f"  Mean accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
    print(f"  Mean log-likelihood: {cv_results['mean_log_likelihood']:.3f} ± {cv_results['std_log_likelihood']:.3f}")
    
    # Test reciprocal classification schemes
    print(f"\nTesting reciprocal classification schemes...")
    results, weights, bias = test_reciprocal_classification(X_train, y_train, reciprocals)
    
    print(f"\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    # Compare the two models
    model_a = results['as_determinative']
    model_b = results['as_pronoun']
    
    print(f"\nMODEL A: Reciprocals as DETERMINATIVES")
    print(f"  Overall accuracy: {model_a['metrics']['accuracy']:.3f}")
    print(f"  Log-likelihood: {model_a['metrics']['log_likelihood']:.3f}")
    print(f"  N samples: {model_a['metrics']['n_samples']}")
    
    print(f"\nMODEL B: Reciprocals as PRONOUNS (current)")
    print(f"  Overall accuracy: {model_b['metrics']['accuracy']:.3f}")
    print(f"  Log-likelihood: {model_b['metrics']['log_likelihood']:.3f}")
    print(f"  N samples: {model_b['metrics']['n_samples']}")
    
    # Model comparison
    accuracy_diff = model_a['metrics']['accuracy'] - model_b['metrics']['accuracy']
    ll_diff = model_a['metrics']['log_likelihood'] - model_b['metrics']['log_likelihood']
    
    print(f"\nMODEL COMPARISON:")
    print(f"  Accuracy difference (A - B): {accuracy_diff:+.4f}")
    print(f"  Log-likelihood difference (A - B): {ll_diff:+.4f}")
    
    if ll_diff > 0:
        print(f"  BETTER FIT: Model A (reciprocals as determinatives)")
    elif ll_diff < 0:
        print(f"  BETTER FIT: Model B (reciprocals as pronouns)")
    else:
        print(f"  EQUAL FIT: No meaningful difference")
    
    # Individual reciprocal predictions
    print(f"\n" + "="*50)
    print("INDIVIDUAL RECIPROCAL PREDICTIONS")
    print("="*50)
    print("(Based on classifier trained on non-reciprocal data)")
    
    for reciprocal in reciprocals:
        predictions = model_a['reciprocal_predictions']
        pred = next(p for p in predictions if p['lemma'] == reciprocal['lemma'])
        
        print(f"\n{pred['lemma']}:")
        print(f"  Probability of being pronoun: {pred['prob_pronoun']:.3f}")
        print(f"  Probability of being determinative: {pred['prob_determinative']:.3f}")
        print(f"  Predicted category: {pred['predicted_category']}")
    
    # Summary interpretation
    print(f"\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    if abs(ll_diff) < 0.01:
        interpretation = "No meaningful difference between models - both fit equally well"
    elif ll_diff > 0.01:
        interpretation = "Model A (determinatives) fits better, but difference may be small"
    else:
        interpretation = "Model B (pronouns) fits better - supports traditional classification"
    
    print(f"Model fit interpretation: {interpretation}")
    
    # Caveat about post-hoc analysis
    print(f"\nCAVEAT: This is post-hoc analysis after seeing clustering results.")
    print(f"Results should be interpreted cautiously due to multiple testing.")
    print(f"Consider this exploratory evidence, not confirmatory.")
    
    return results, cv_results

if __name__ == "__main__":
    results, cv_results = run_model_comparison()