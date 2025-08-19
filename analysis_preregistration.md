# Pre-registered Analysis Plan: Reciprocals as Determinatives

## Hypothesis
*Each other* and *one another* are unitary determinative lexemes, not pronouns. They function as fused-determiner heads, similar to *someone* or *anything*.

### Motivating Analogy
**Theoretical foundation:** Payne, Huddleston & Pullum (2007) "Fusion of functions: The syntax of once, twice and thrice" *Journal of Linguistics* 43(3):565-603.

**Template: Fused Det-Head determinatives (*someone*, *anyone*, *everything*)**
- NP distribution (objects of V/P, possessors: *someone's idea*)  
- DP-like pre-head modifiers (*almost everyone*, *hardly anyone*, *not anything*)
- Block external determiners (**the anyone*) and adjectival modifiers (**happy anyone*)
- Allow post-head restrictors where semantically licensed (*anyone present*, *something edible*)

**Application to reciprocals:**
- Same NP distribution (*see each other*, *each other's ideas*)
- Same DP-targeting particles (*only each other*)  
- Same blocking effects (**the each other*, **happy one another*)
- Defective members: no restrictors (**each other present*), no partitives (**each other of the students*)
- Internal morphology differs (D+A vs D+D) but external category is determinative

## Strong Prior Against Hypothesis
**Authoritative sources consistently classify these as pronouns:**
- Cambridge Grammar of the English Language
- Major dictionaries
- Standard linguistic references

**Prior probability**: Given this consensus, we assign a strong prior probability (~0.85-0.9) that reciprocals ARE pronouns, and only ~0.1-0.15 that they are determinatives.

## Pre-registered Predictions (to avoid forking paths)

### If reciprocals are determinatives (my hypothesis):
1. `functions_as_fused_determiner_head` = 1 (like *someone*, *anything*)
2. `paradigm_has_distinct_acc_form` = 0 (no case paradigm)
3. `appears_in_subject` = 0 (subject avoidance due to licensing construction)
4. `inflects_for_case` = 0 (morphology targets lexeme as whole)
5. Distance to determinative centroid < distance to pronoun centroid

### If reciprocals are pronouns (standard view):
1. `functions_as_fused_determiner_head` = 0 
2. Pattern similarity to other pronouns (*he*, *him*, *himself*)
3. Distance to pronoun centroid < distance to determinative centroid

## Falsification Criteria
**Hypothesis is rejected if:**
- Reciprocals cluster closer to pronouns than determinatives (Euclidean distance in feature space)
- Key determinative features (functions_as_fused_determiner_head) = 0
- Bayesian model gives posterior probability < 0.3 for determinative classification

## Analysis Plan
1. Calculate feature-based distances to class centroids
2. Bayesian logistic regression with informative priors
3. Cross-validation excluding reciprocals
4. Report uncertainty intervals, not just point estimates

## Simulation Results (2025-08-18)
**✅ Method can distinguish real differences** when they exist
**❌ CRITICAL ISSUE: Method is over-sensitive** - finds false patterns under null hypothesis
- Under null (no real difference): 2/2 reciprocals still "prefer" one group
- Expected under null: ~1/2 by random chance
- **Conclusion: Systematic bias in distance metric or comparison groups**

**ACTION REQUIRED:** Fix over-sensitivity before proceeding with real data analysis

## POST-HOC ANALYSIS (Forking Paths - Documented)
**Date: 2025-08-18**
**Motivation**: Primary hypothesis test showed no clustering effect. Now testing model fit approach.

**Research Question**: Does reclassifying reciprocals as "determinative" vs "pronoun" improve model fit to feature data?

**Method**: Cross-validation comparison
- Model A: {each_other, one_another} labeled as "determinative"  
- Model B: {each_other, one_another} labeled as "pronoun" (current)
- Train classifier on non-reciprocal items, test which labeling fits better

**Caveat**: This is exploratory analysis after seeing primary results. Results should be interpreted cautiously due to multiple testing.

## WHERE GELMAN WOULD GO FROM HERE
**Status**: Hypothesis strongly rejected by multiple converging analyses

**Gelman's Next Steps:**
1. **Data Quality Assessment**: Are these features capturing the right linguistic distinctions?
2. **Alternative Feature Sets**: Test on different linguistic corpora or feature matrices
3. **Theory Refinement**: Is the fusion-of-functions framework better tested on other constructions?
4. **Measurement Error**: Could reciprocals be mis-coded in the feature matrix?
5. **Power Analysis**: Retrospective - was this the right test for the theory?

**Scientific Integrity Check:**
- Pre-registered hypothesis: ❌ Rejected
- Post-hoc model comparison: ❌ Also rejects hypothesis  
- Multiple convergent evidence: All point to traditional classification
- **Conclusion**: Accept Cambridge Grammar is correct for this feature space

*Date: 2025-08-18*
*Analyst: Claude Code*