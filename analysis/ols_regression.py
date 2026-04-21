# analysis/ols_regression.py
"""OLS regression analysis of network metrics and graph types.

Runs OLS regressions with robust standard errors to analyze how individual network metrics 
and graph types influence multi-agent performance, with both main effects only models and
models that include task difficulty interactions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
from sklearn.linear_model import LinearRegression

# Import local modules
from analysis.robust_standard_errors import RobustStandardErrors
from analysis.util import (
    get_task_difficulties,
    get_control_variables,
    get_network_metrics,
    get_outcome_metrics
)

def run_ols_regression_with_robust_errors(X, y):
    """Run OLS regression with robust standard errors and calculate performance metrics."""
    # Fit model with built-in intercept
    model = LinearRegression(fit_intercept=True).fit(X, y)
    
    # Suppress the specific warning about feature names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                               message="X does not have valid feature names, but LinearRegression was fitted with feature names")
        # Calculate robust standard errors
        rse = RobustStandardErrors(model, X, y, cov_type='HC2').fit()
    
    # Calculate R²
    r2 = model.score(X, y)
    
    # Calculate adjusted R²
    n = len(X)
    p = X.shape[1]  # Number of features (without intercept)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    # Extract summary information
    summary = rse.summary()
    
    return {
        'model': model,
        'r2': r2,
        'adj_r2': adj_r2,
        'n_samples': n,
        'summary': summary
    }

def add_coefficients_to_result(result_row, coef_dict, feature_groups):
    """Add coefficient and p-value pairs to result row for multiple feature groups."""
    
    # Add intercept
    result_row['intercept_coef'] = coef_dict.get('intercept', {}).get('coef', np.nan)
    result_row['intercept_p'] = coef_dict.get('intercept', {}).get('p_value', np.nan)
    
    # Add coefficients for each feature group
    for group_name, features in feature_groups.items():
        for feature in features:
            # Handle special cases for main effect variable (network_metric or graph_slug)
            if group_name == 'main_effect':
                result_row['main_effect_coef'] = coef_dict.get(feature, {}).get('coef', np.nan)
                result_row['main_effect_p'] = coef_dict.get(feature, {}).get('p_value', np.nan)
            else:
                result_row[f"{feature}_coef"] = coef_dict.get(feature, {}).get('coef', np.nan)
                result_row[f"{feature}_p"] = coef_dict.get(feature, {}).get('p_value', np.nan)
    
    return result_row

def run_ols_regression_for_single_network_metric(df, outcome, metric_name, dataset_name, include_interactions=True):
    """Run OLS regression for a single network metric with optional task difficulty interactions."""
    # Get control variables and task difficulties
    controls = get_control_variables(transformed=True)
    task_difficulties = get_task_difficulties(transformed=True)
    
    # Create feature list: controls, metric, tasks
    features = controls + [metric_name] + task_difficulties
    
    # Create interaction terms if requested
    interaction_terms = []
    if include_interactions:
        for task in task_difficulties:
            interaction_name = f"interaction_{task}"
            interaction_terms.append(interaction_name)
            df[interaction_name] = df[metric_name] * df[task]
        features.extend(interaction_terms)
    
    # Prepare X and y
    X = df[features].copy()
    y = df[outcome]
    
    # Run regression
    results = run_ols_regression_with_robust_errors(X, y)
    
    # Extract coefficient information from summary
    summary = results['summary']
    
    # Create coefficient dictionary
    coef_dict = {row_name: {'coef': row['Coefficient'], 
                          'p_value': row['P>|t|'],
                          'std_err': row['Std Error (HC2)']} 
                for row_name, row in summary.iterrows()}
    
    # Initialize row for the tidy dataframe
    result_row = {
        'dataset': dataset_name,
        'outcome': outcome,
        'main_effect_name': metric_name,
        'model_type': 'with_interactions' if include_interactions else 'main_effects_only',
        'r_squared': results['r2'],
        'adj_r_squared': results['adj_r2'],
        'n_samples': results['n_samples'],
    }
    
    # Define feature groups
    feature_groups = {
        'controls': controls,
        'main_effect': [metric_name],
        'task_difficulties': task_difficulties,
        'interactions': interaction_terms
    }
    
    # Add coefficients to result row
    result_row = add_coefficients_to_result(result_row, coef_dict, feature_groups)
    
    return result_row

def run_ols_regression_for_single_graph_slug(df, outcome, graph_slug, dataset_name, include_interactions=True):
    """Run OLS regression for a single graph type with optional task difficulty interactions."""
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Get control variables and task difficulties
    controls = ['log_agent_steplim']
    task_difficulties = get_task_difficulties(transformed=True)
    
    # Filter to only include teams of size 8 (using approximate matching)
    team_size_8_norm = 0.42857146
    df = df[np.isclose(df['norm_team_size'], team_size_8_norm, atol=1e-6)]
    df = df.drop('norm_team_size', axis=1)
    
    # Create dummy variable for this graph type
    dummy_name = f"graph_slug_{graph_slug}"
    df[dummy_name] = (df['graph_slug'] == graph_slug).astype(float)
    
    # Create feature list: controls, graph dummy, tasks
    features = controls + [dummy_name] + task_difficulties
    
    # Create interaction terms if requested
    interaction_terms = []
    if include_interactions:
        for task in task_difficulties:
            interaction_name = f"interaction_{task}"
            interaction_terms.append(interaction_name)
            df[interaction_name] = df[dummy_name] * df[task]
        features.extend(interaction_terms)
    
    # Prepare X and y
    X = df[features]
    y = df[outcome]
    
    # Run regression
    results = run_ols_regression_with_robust_errors(X, y)
    
    # Extract coefficient information from summary
    summary = results['summary']
    
    # Create coefficient dictionary
    coef_dict = {row_name: {'coef': row['Coefficient'], 
                          'p_value': row['P>|t|'],
                          'std_err': row['Std Error (HC2)']} 
                for row_name, row in summary.iterrows()}
    
    # Initialize row for the tidy dataframe
    result_row = {
        'dataset': dataset_name,
        'outcome': outcome,
        'main_effect_name': graph_slug,
        'model_type': 'with_interactions' if include_interactions else 'main_effects_only',
        'r_squared': results['r2'],
        'adj_r_squared': results['adj_r2'],
        'n_samples': results['n_samples'],
    }
    
    # Define feature groups
    feature_groups = {
        'controls': controls,
        'main_effect': [dummy_name],
        'task_difficulties': task_difficulties,
        'interactions': interaction_terms
    }
    
    # Add coefficients to result row
    result_row = add_coefficients_to_result(result_row, coef_dict, feature_groups)
    
    return result_row

def run_ols_regressions_for_all_network_metrics(study_name, dataset_name, outcome):
    """Run OLS regressions for all network metrics with and without task difficulty interactions."""
    start_time = time.time()
    
    # Set up paths
    data_dir = Path("data")
    dataset_path = data_dir / "preprocessed" / study_name / f"{dataset_name}.parquet"
    
    print(f"\nRunning OLS regressions for all network metrics - {dataset_name} - {outcome}")
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    # Get all network metrics for this dataset
    include_special = (dataset_name == "dataset2")
    network_metrics = get_network_metrics(include_special_metrics=include_special)
    
    # Run regression for each metric
    results_rows = []
    
    for metric in network_metrics:
        print(f"  Analyzing {metric}...")
        
        # Run regression without interactions
        result_row_main = run_ols_regression_for_single_network_metric(
            df, outcome, metric, dataset_name, include_interactions=False
        )
        results_rows.append(result_row_main)
        
        # Run regression with interactions
        result_row_interx = run_ols_regression_for_single_network_metric(
            df, outcome, metric, dataset_name, include_interactions=True
        )
        results_rows.append(result_row_interx)
    
    # Calculate runtime
    total_time = time.time() - start_time
    print(f"Network metrics analysis completed in {total_time:.1f} seconds")
    
    return pd.DataFrame(results_rows)

def run_ols_regressions_for_all_graph_slugs(study_name, dataset_name, outcome, reference_slug=None):
    """Run OLS regressions for all graph types with and without task difficulty interactions."""
    start_time = time.time()
    
    # Set up paths
    data_dir = Path("data")
    dataset_path = data_dir / "preprocessed" / study_name / f"{dataset_name}.parquet"
    
    print(f"\nRunning OLS regressions for all graph types - {dataset_name} - {outcome}")
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    # Get all graph slugs
    graph_slugs = sorted(df['graph_slug'].unique())
    
    # Set reference slug if not provided
    if reference_slug is None:
        reference_slug = graph_slugs[0]
    elif reference_slug not in graph_slugs:
        print(f"  Warning: Reference slug '{reference_slug}' not found in dataset. Using '{graph_slugs[0]}' instead.")
        reference_slug = graph_slugs[0]
    
    print(f"  Using '{reference_slug}' as reference category")
    
    # Remove reference slug from the list
    graph_slugs = [slug for slug in graph_slugs if slug != reference_slug]
    
    # Run regression for each graph type
    results_rows = []
    
    for graph_slug in graph_slugs:
        print(f"  Analyzing {graph_slug}...")
        
        # Run regression without interactions
        result_row_main = run_ols_regression_for_single_graph_slug(
            df, outcome, graph_slug, dataset_name, include_interactions=False
        )
        result_row_main['reference_slug'] = reference_slug
        results_rows.append(result_row_main)
        
        # Run regression with interactions
        result_row_interx = run_ols_regression_for_single_graph_slug(
            df, outcome, graph_slug, dataset_name, include_interactions=True
        )
        result_row_interx['reference_slug'] = reference_slug
        results_rows.append(result_row_interx)
    
    # Calculate runtime
    total_time = time.time() - start_time
    print(f"Graph slugs analysis completed in {total_time:.1f} seconds")
    
    return pd.DataFrame(results_rows)

def run_all_ols_regressions(study_name="aiteams01nm_20250128_223001", 
                          datasets=None, outcomes=None):
    """Run all OLS regressions (with and without interactions) for specified datasets and outcomes."""
    # Set defaults
    if datasets is None:
        datasets = ['dataset1', 'dataset2']
    if outcomes is None:
        outcomes = get_outcome_metrics()
    
    # Initialize result dataframes
    network_metric_results = []
    graph_slug_results = []
    
    # Run all analyses
    for dataset_name in datasets:
        for outcome in outcomes:
            print(f"\nAnalyzing {dataset_name} - {outcome}")
            
            # Run network metric regressions (both with and without interactions)
            nm_results = run_ols_regressions_for_all_network_metrics(
                study_name, dataset_name, outcome
            )
            network_metric_results.append(nm_results)
            
            # Run graph slug regressions (both with and without interactions)
            gs_results = run_ols_regressions_for_all_graph_slugs(
                study_name, dataset_name, outcome
            )
            graph_slug_results.append(gs_results)
    
    # Combine results into two main dataframes
    network_metric_df = pd.concat(network_metric_results, ignore_index=True)
    graph_slug_df = pd.concat(graph_slug_results, ignore_index=True)
    
    # Save results
    results_dir = Path("data") / "results" / study_name / "ols_regressions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV files
    network_metric_path = results_dir / "network_metric_regressions.csv"
    graph_slug_path = results_dir / "graph_slug_regressions.csv"
    
    network_metric_df.to_csv(network_metric_path, index=False)
    graph_slug_df.to_csv(graph_slug_path, index=False)
    
    print("\nAll OLS regression analyses complete.")
    print(f"Network metric results saved to: {network_metric_path}")
    print(f"Graph slug results saved to: {graph_slug_path}")
    
    return {
        'network_metrics': network_metric_df,
        'graph_slugs': graph_slug_df
    }

if __name__ == "__main__":
    # Run with default parameters
    results = run_all_ols_regressions()