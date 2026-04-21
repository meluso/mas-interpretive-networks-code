# analysis/graph_type_ols_analysis.py

"""
Analyze graph type effects using separate sklearn regressions for each graph type.
Each regression shows how a specific network structure performs across different task types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import time

# Import local modules
from analysis.robust_standard_errors import RobustStandardErrors
from analysis.util import (
    get_task_difficulties,
    get_control_variables
)

def load_data_by_graph_type(study_name, dataset_name):
    """Load dataset and separate it by graph type."""
    # Set up paths
    data_dir = Path("data/preprocessed") / study_name
    dataset_path = data_dir / f"{dataset_name}.parquet"
    
    # Load data
    print(f"Loading data from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    # Extract unique graph types
    graph_types = df['graph_slug'].unique()
    
    # Create dictionary of dataframes by graph type
    graph_dfs = {graph: df[df['graph_slug'] == graph].copy() for graph in graph_types}
    
    print(f"Found {len(graph_types)} graph types with {len(df)} total observations")
    return graph_dfs, graph_types

def diagnose_multicollinearity(df, features, threshold=0.9):
    """
    Diagnose multicollinearity issues in feature set.
    
    Args:
        df: DataFrame for a specific graph type
        features: List of features being used in regression
        threshold: Correlation threshold to flag
    
    Returns:
        Dict with diagnostic information
    """
    print(f"\nMulticollinearity diagnosis for {len(df)} observations:")
    
    # Check for constant or near-constant features
    variance = df[features].var()
    low_var_features = variance[variance < 1e-6].index.tolist()
    if low_var_features:
        print(f"  Warning: Features with near-zero variance: {', '.join(low_var_features)}")
        
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                high_corr_pairs.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': corr
                })
    
    # Print high correlation pairs
    if high_corr_pairs:
        print(f"  Found {len(high_corr_pairs)} highly correlated pairs (|r| > {threshold}):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"    {pair['feature1']} & {pair['feature2']}: r = {pair['correlation']:.4f}")
    else:
        print(f"  No highly correlated pairs found (|r| > {threshold})")
    
    # Check condition number
    from numpy.linalg import svd
    X = df[features].values
    s = svd(X, compute_uv=False)
    condition_number = s[0] / s[-1]
    print(f"  Condition number: {condition_number:.2e}")
    
    if condition_number > 1e10:
        print("  CRITICAL: Extreme multicollinearity detected (condition number > 1e10)")
    elif condition_number > 1e4:
        print("  WARNING: Severe multicollinearity detected (condition number > 1e4)")
    
    # Return diagnostic data
    return {
        'low_variance': low_var_features,
        'high_correlations': high_corr_pairs,
        'condition_number': condition_number
    }

def run_sklearn_regression(df, outcome, features):
    """Run LinearRegression with robust standard errors for a single graph type.
    
    Args:
        df: DataFrame with data for a specific graph type
        outcome: Target outcome variable
        features: List of feature columns to use
        
    Returns:
        Dict with regression results
    """
    # Prepare X and y
    X = df[features]
    y = df[outcome]
    
    # Add constant for intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Fit model
    model = LinearRegression(fit_intercept=False).fit(X_with_intercept, y)
    
    # Calculate robust standard errors
    rse = RobustStandardErrors(model, X_with_intercept, y, cov_type='HC2').fit()
    
    # Separate intercept from other coefficients
    intercept = model.coef_[0]
    coefficients = model.coef_[1:]
    
    # Calculate R²
    r2 = model.score(X_with_intercept, y)
    
    # Calculate adjusted R²
    n = len(X)
    p = len(features) + 1  # +1 for intercept
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    # Extract summary information
    summary = rse.summary()
    
    # Return results
    return {
        'model': model,
        'intercept': intercept,
        'coefficients': coefficients,
        'feature_names': features,
        'r2': r2,
        'adj_r2': adj_r2,
        'n_samples': n,
        'rse_summary': summary,
        'std_errors': rse.std_errors_,
        'p_values': rse.p_values_,
        'conf_int': rse.conf_int_
    }

def run_all_graph_regressions(study_name, dataset_name, outcome_metric):
    """Run separate regressions for each graph type."""
    start_time = time.time()
    
    # Load data by graph type
    graph_dfs, graph_types = load_data_by_graph_type(study_name, dataset_name)
    
    # Get controls and task difficulties
    controls = get_control_variables(transformed=True)
    task_difficulties = [t for t in get_task_difficulties(transformed=True) 
                        if 'interdep' not in t]
    features = controls + task_difficulties
    
    # Combine features
    features = controls + task_difficulties
    
    # Setup results directory
    results_dir = Path(f"data/results/{study_name}/graph_type_ols")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run regression for each graph type
    models = {}
    coef_rows = []
    summary_rows = []
    
    print(f"Running regressions for {len(graph_types)} graph types")
    
    for graph in sorted(graph_types):
        df = graph_dfs[graph]
        n_samples = len(df)
        
        if n_samples < 10:
            print(f"  Skipping {graph} - insufficient data ({n_samples} samples)")
            continue
            
        print(f"\nAnalyzing {graph} with {n_samples} observations")
        
        # Run diagnostics before regression
        diagnostics = diagnose_multicollinearity(df, features)
        
        # Proceed with regression if appropriate
        if diagnostics['condition_number'] < 1e15:  # Still try unless extreme
            try:
                
                # Run regression
                results = run_sklearn_regression(df, outcome_metric, features)
                models[graph] = results
                
                # Store summary stats
                summary_rows.append({
                    'graph_type': graph,
                    'n_samples': n_samples,
                    'r_squared': results['r2'],
                    'adj_r_squared': results['adj_r2']
                })
                
                # Get feature names with intercept
                feature_names = ['Intercept'] + features
                
                # Extract coefficient values and statistics
                coef_values = np.concatenate([[results['intercept']], results['coefficients']])
                std_errors = results['std_errors_']
                p_values = results['p_values_']
                
                # Store coefficients
                for i, feat in enumerate(feature_names):
                    coef_rows.append({
                        'graph_type': graph,
                        'variable': feat,
                        'coefficient': coef_values[i],
                        'std_error': std_errors[i],
                        'p_value': p_values[i],
                        'significance': '***' if p_values[i] < 0.001 else
                                        '**' if p_values[i] < 0.01 else
                                        '*' if p_values[i] < 0.05 else ''
                    })
                
            except Exception as e:
                print(f"  Error analyzing {graph}: {str(e)}")
        else:
            raise RuntimeError('Multicollinear')
    
    # Create dataframes
    coef_df = pd.DataFrame(coef_rows)
    summary_df = pd.DataFrame(summary_rows)
    
    # Save results
    result_path = results_dir / f"{dataset_name}_{outcome_metric}"
    result_path.mkdir(parents=True, exist_ok=True)
    
    coef_df.to_csv(result_path / "coefficients.csv", index=False)
    summary_df.to_csv(result_path / "model_summaries.csv", index=False)
    
    # Print summary statistics
    print("\nModel performance by graph type (sorted by R²):")
    print(summary_df.sort_values('r_squared', ascending=False)[['graph_type', 'n_samples', 'r_squared', 'adj_r_squared']].head(10).to_string(index=False))
    
    # Calculate task importance overall
    task_importance = coef_df[coef_df['variable'].isin(task_difficulties)]
    task_significance = task_importance.groupby('variable').apply(
        lambda x: (x['p_value'] < 0.05).mean()
    ).reset_index(name='significance_rate')
    task_significance['avg_abs_coef'] = task_importance.groupby('variable')['coefficient'].apply(
        lambda x: abs(x).mean()
    ).values
    
    # Print task importance
    print("\nTask importance across graph types:")
    print(task_significance.sort_values('avg_abs_coef', ascending=False).to_string(index=False))
    
    # Calculate runtime
    total_time = time.time() - start_time
    print(f"\nAnalysis completed in {total_time:.1f} seconds")
    print(f"Results saved to {result_path}")
    
    return {
        'coefficients': coef_df,
        'summaries': summary_df,
        'models': models
    }

def run_all_analyses(study_name="aiteams01nm_20250128_223001"):
    """Run analyses for all datasets and outcomes."""
    # Define datasets and outcomes
    datasets = ["dataset1", "dataset2"]
    outcomes = ["convergence_performance", "convergence_step"]
    
    results = {}
    
    # Run analysis for each combination
    for dataset_name in datasets:
        results[dataset_name] = {}
        
        for outcome in outcomes:
            print(f"\nAnalyzing {outcome} with {dataset_name}")
            
            try:
                # Run regression analysis
                analysis_results = run_all_graph_regressions(
                    study_name, 
                    dataset_name, 
                    outcome
                )
                
                # Store results
                results[dataset_name][outcome] = analysis_results
                
            except Exception as e:
                print(f"Error analyzing {dataset_name} - {outcome}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    return results

if __name__ == "__main__":
    # Run analysis
    study_name = "aiteams01nm_20250128_223001"
    results = run_all_analyses(study_name)
    print("All analyses complete.")