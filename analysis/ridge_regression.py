# analysis/ridge_regression.py
"""Ridge regression core functionality for AI Teams analyses.

Provides generalized ridge regression implementation that can be used
by various analysis modules with different predictor sets.

This module implements ridge regression with cross-validation for analyzing network structure 
effects on multi-agent performance. Key features include:
- Creates models of increasing complexity (baseline, network metrics, network-task interactions)
- Uses RidgeCV with 5-fold cross-validation to select optimal alpha values
- Calculates robust standard errors (HC2) for statistical inference
- Formats results for coefficient comparisons across models
- Returns residuals for subsequent mediation analysis
"""

import numpy as np
import pandas as pd
import time

# Import scikit-learn
from sklearn.linear_model import RidgeCV

# Import local modules
from analysis.robust_standard_errors import RobustStandardErrors

def run_ridge_regression(ddf, outcome, base_features, interactions=None):
    """Run ridge regression with generic feature sets.
    
    Args:
        ddf: Dask DataFrame
        outcome: Target variable name
        base_features: List of base feature columns to include
        interactions: List of (feature1, feature2) pairs for interactions or None
    
    Returns:
        Dict containing model results
    """
    # Start with base features
    all_features = base_features.copy()
    
    # Map to keep track of interaction terms and their components
    interaction_map = {}
    
    # Add interaction terms if specified
    interaction_dict = {}
    if interactions:
        print(f"    Generating {len(interactions)} interaction terms...")
        for feat1, feat2 in interactions:
            interaction_name = f"{feat1}_{feat2}"
            all_features.append(interaction_name)
            interaction_dict[interaction_name] = ddf[feat1] * ddf[feat2]
            interaction_map[interaction_name] = (feat1, feat2)
    
    print(f"    Starting ridge regression with {len(all_features)} features...")
    start_time = time.time()
    
    # Add interaction columns to dataframe if needed
    if interaction_dict:
        ddf = ddf.assign(**interaction_dict)
    
    # Select only necessary columns to reduce memory usage
    ddf_subset = ddf[all_features + [outcome]]
    
    # Compute the dataset once
    print("    Computing DataFrame subset...")
    df = ddf_subset.compute()
    
    # Extract features and target
    X = df[all_features]
    y = df[outcome]
    
    # Create and fit model with 5-fold cross-validation
    print("    Fitting model with 5-fold cross-validation...")
    alphas = np.logspace(-2, 2, 10)  # 10 values between 0.01 and 100
    model = RidgeCV(alphas=alphas, cv=5, fit_intercept=True)
    model.fit(X, y)
    
    # Get best alpha
    best_alpha = model.alpha_
    
    # Calculate R² score
    print("    Calculating R² score...")
    r2 = model.score(X, y)
    
    # Calculate AIC and BIC
    n = len(X)
    k = len(all_features) + 1  # +1 for intercept
    
    # Calculate RSS
    y_pred = model.predict(X)
    residuals = y - y_pred
    rss = np.sum(residuals**2)
    
    # Calculate AIC and BIC
    aic = n * np.log(rss/n) + 2 * k
    bic = n * np.log(rss/n) + np.log(n) * k
    
    # Calculate robust standard errors
    print("    Calculating robust standard errors...")
    rse = RobustStandardErrors(model, X, y, cov_type='HC2').fit()
    print("    RSE calculation results:")
    print(f"    - Has coef_: {hasattr(rse, 'coef_')}")
    print(f"    - Has std_errors_: {hasattr(rse, 'std_errors_')}")
    print(f"    - Has p_values_: {hasattr(rse, 'p_values_')}")
    print(f"    - Shape of X: {X.shape}")
    print(f"    - Condition number of X'X: {np.linalg.cond(X.T @ X)}")
    
    compute_time = time.time() - start_time
    print(f"    Completed in {compute_time:.2f} seconds. R² = {r2:.4f}, optimal alpha = {best_alpha:.6f}")
    
    # Collect results
    results = {
        'model': model,
        'features': all_features,
        'best_alpha': best_alpha,
        'coefficients': rse.coef_,
        'std_errors': rse.std_errors_,
        'p_values': rse.p_values_,
        'conf_int': rse.conf_int_,
        'r2_score': r2,
        'summary': rse.summary(),
        'n_samples': len(X),
        'aic': aic,
        'bic': bic,
        'interaction_map': interaction_map,
        'residuals': residuals
    }
    
    return results

def format_regression_results(models_dict):
    """Format results from models for comparison.
    
    Creates two outputs:
    1. A model comparison DataFrame with performance metrics
    2. A coefficient DataFrame with MultiIndex (Base Feature, Interacting Feature)
    """
    # Extract key information from each model
    comparison = pd.DataFrame({
        'Model': [m['model_name'] for m in models_dict.values()],
        'R²': [m['r2_score'] for m in models_dict.values()],
        'Alpha': [m['best_alpha'] for m in models_dict.values()],
        'AIC': ['{:.2f}'.format(m['aic']) for m in models_dict.values()],
        'BIC': ['{:.2f}'.format(m['bic']) for m in models_dict.values()],
        'Features': [len(m['features']) for m in models_dict.values()]
    })
    
    # Create coefficient comparison matrix with standard errors
    all_features = set()
    for model_results in models_dict.values():
        all_features.update(model_results['features'])
    
    # Ensure intercept is included
    all_features.add('intercept')
    
    # Combine all interaction maps
    all_interaction_maps = {}
    for model_results in models_dict.values():
        if 'interaction_map' in model_results:
            all_interaction_maps.update(model_results['interaction_map'])
    
    # Create tuples for MultiIndex: (Base Feature, Interacting Feature)
    index_tuples = []
    for feature in sorted(all_features):
        if feature == 'intercept':
            # Special case for intercept
            index_tuples.append(('intercept', None))
        elif feature in all_interaction_maps:
            # This is an interaction term, use the stored mapping
            feat1, feat2 = all_interaction_maps[feature]
            index_tuples.append((feat1, feat2))
        else:
            # This is a main effect
            index_tuples.append((feature, None))
    
    # Create MultiIndex
    multi_idx = pd.MultiIndex.from_tuples(index_tuples, names=['Base Feature', 'Interacting Feature'])
    
    # Initialize coefficient matrix with MultiIndex
    coef_matrix = pd.DataFrame(index=multi_idx)
    
    # Fill in coefficients and standard errors from each model
    for model_id, model_results in models_dict.items():
        # Get short model name
        model_name = model_id.replace('model', 'M')
        
        # Add coefficient, standard error, p-value, and significance columns
        coef_matrix[f"{model_name}_coef"] = np.nan
        coef_matrix[f"{model_name}_se"] = np.nan
        coef_matrix[f"{model_name}_pval"] = np.nan
        coef_matrix[f"{model_name}_sig"] = ''
        
        # Extract summary data
        summary = model_results['summary']
        
        # Map from original feature names to MultiIndex tuples
        feature_to_idx = {}
        for feature in all_features:
            if feature == 'intercept':
                feature_to_idx[feature] = ('intercept', None)
            elif feature in all_interaction_maps:
                feat1, feat2 = all_interaction_maps[feature]
                feature_to_idx[feature] = (feat1, feat2)
            else:
                feature_to_idx[feature] = (feature, None)
        
        # Fill available values
        for feature in list(all_features):
            if feature in summary.index:
                idx = feature_to_idx.get(feature)
                if idx:
                    coef_matrix.loc[idx, f"{model_name}_coef"] = summary.loc[feature, 'Coefficient']
                    coef_matrix.loc[idx, f"{model_name}_se"] = summary.loc[feature, 'Std Error (HC2)']
                    coef_matrix.loc[idx, f"{model_name}_pval"] = summary.loc[feature, 'P>|t|']
                    
                    # Add significance markers
                    pval = summary.loc[feature, 'P>|t|']
                    if pval < 0.001:
                        coef_matrix.loc[idx, f"{model_name}_sig"] = '***'
                    elif pval < 0.01:
                        coef_matrix.loc[idx, f"{model_name}_sig"] = '**'
                    elif pval < 0.05:
                        coef_matrix.loc[idx, f"{model_name}_sig"] = '*'
    
    return {
        'comparison': comparison,
        'coefficients': coef_matrix
    }