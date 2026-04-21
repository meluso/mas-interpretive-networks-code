# analysis/graph_type_residual_effects.py
"""Mediation analysis examining whether network metrics fully explain graph type effects.

Implements Analysis 2.2: OLS regression of residuals from network property analysis 
on graph slugs to test for unexplained variance.
"""

import dask.dataframe as dd
import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.linear_model import LinearRegression

# Import from our other analysis modules
from analysis.ridge_regression import run_ridge_regression
from analysis.network_property_regression_effects import create_network_property_models
from analysis.robust_standard_errors import RobustStandardErrors

# Import utility functions
from analysis.util import (
    get_outcome_metrics,
    get_categorical_variables
)

def prepare_graph_dummy_variables(ddf):
    """Create dummy variables for graph slugs."""
    # Get unique values for graph_slug
    graph_slugs = ddf['graph_slug'].unique().compute().tolist()
    graph_slugs.sort()  # Ensure consistent reference category
    
    # Use first value as reference category
    reference_slug = graph_slugs[0]
    graph_slugs = graph_slugs[1:]
    
    print(f"  Using {reference_slug} as reference category")
    
    # Create dummy variables
    dummy_cols = {}
    for slug in graph_slugs:
        col_name = f"graph_slug_{slug}"
        dummy_cols[col_name] = (ddf['graph_slug'] == slug).astype(int)
    
    # Add dummy columns to dataframe
    return ddf.assign(**dummy_cols), graph_slugs, reference_slug

def run_residual_analysis(df, outcome, graph_slug_dummies, reference_slug):
    """Run OLS regression of residuals on graph slug dummies.
    
    Args:
        df: Dask DataFrame with graph slug dummy variables
        outcome: Target outcome to analyze
        graph_slug_dummies: List of graph slug dummy column names
        reference_slug: Reference category for graph slugs
        
    Returns:
        Dict containing OLS model results
    """
    print(f"  Running OLS regression of residuals on graph slugs")
    
    # Create feature matrix X
    X = df[graph_slug_dummies]
    
    # Use residuals as dependent variable
    y = df[f"residual_{outcome}"]
    
    # Fit OLS model
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    # Calculate R²
    r2 = model.score(X, y)
    
    # Calculate robust standard errors
    rse = RobustStandardErrors(model, X, y, cov_type='HC2').fit()
    
    # Create summary table with reference category
    summary = rse.summary().copy()
    
    # Add row for reference category
    reference_row = pd.DataFrame(
        [['reference', 0.0, 0.0, float('nan'), float('nan'), float('nan')]],
        index=[f'graph_slug_{reference_slug}'],
        columns=summary.columns
    )
    summary = pd.concat([reference_row, summary])
    
    # Add statistical significance markers
    summary['Significance'] = ''
    summary.loc[summary['P>|t|'] < 0.05, 'Significance'] = '*'
    summary.loc[summary['P>|t|'] < 0.01, 'Significance'] = '**'
    summary.loc[summary['P>|t|'] < 0.001, 'Significance'] = '***'
    
    print(f"  Residual analysis R² = {r2:.6f}")
    
    return {
        'model': model,
        'r2': r2,
        'summary': summary,
        'reference_slug': reference_slug
    }

def run_graph_type_residual_analysis(study_name, dataset_name, outcome):
    """Run mediation analysis for a specific dataset and outcome.
    
    Args:
        study_name: Name of the study
        dataset_name: Name of the dataset (dataset1 or dataset2)
        outcome: Target outcome variable name
        
    Returns:
        Dict containing analysis results
    """
    # Set up paths
    data_dir = Path("data")
    dataset_path = data_dir / "preprocessed" / study_name / f"{dataset_name}.parquet"
    results_dir = data_dir / "results" / study_name / "graph_type_residual_effects"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning mediation analysis for {outcome} with {dataset_name}")
    
    # Load dataset
    ddf = dd.read_parquet(dataset_path)
    
    # Step 1: Prepare graph slug dummy variables
    ddf, graph_slug_dummies, reference_slug = prepare_graph_dummy_variables(ddf)
    
    # Step 2: Run network property regression to get residuals
    print("  Running network property regression to obtain residuals")
    model_specs = create_network_property_models(dataset_name)
    
    # We only need the most complex model (Model 3) with all interactions
    model_spec = model_specs['model3']
    
    # Run regression to get residuals
    network_results = run_ridge_regression(
        ddf, 
        outcome, 
        model_spec['base_features'], 
        model_spec['interactions']
    )
    
    # Add residuals to DataFrame
    residuals = network_results['residuals']
    residual_col = f"residual_{outcome}"
    df = ddf.compute()
    df[residual_col] = residuals
    
    # Step 3: Run OLS regression of residuals on graph slugs
    residual_results = run_residual_analysis(
        df, 
        outcome,
        [f"graph_slug_{slug}" for slug in graph_slug_dummies],
        reference_slug
    )
    
    # Save results
    result_dir = results_dir / f"{dataset_name}_{outcome}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary table
    residual_results['summary'].to_csv(result_dir / "residual_analysis.csv")
    
    # Save R² comparison
    r2_comparison = pd.DataFrame({
        'Model': ['Network Properties', 'Residual (Graph Types)'],
        'R²': [network_results['r2_score'], residual_results['r2']],
        'Interpretation': [
            f"Variance explained by network properties",
            f"Unexplained variance attributable to graph types"
        ]
    })
    r2_comparison.to_csv(result_dir / "r2_comparison.csv", index=False)
    
    print(f"  Results saved to {result_dir}")
    
    return {
        'network_r2': network_results['r2_score'],
        'residual_r2': residual_results['r2'],
        'residual_summary': residual_results['summary']
    }

def run_all_residual_analyses(study_name):
    """Run all mediation analyses across datasets and outcomes."""
    start_time = time.time()
    
    # Define datasets and outcomes
    datasets = ["dataset1", "dataset2"]
    outcomes = get_outcome_metrics()
    
    all_results = {}
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        
        for outcome in outcomes:
            try:
                results = run_graph_type_residual_analysis(study_name, dataset_name, outcome)
                all_results[dataset_name][outcome] = results
                
                # Print interpretation
                network_r2 = results['network_r2']
                residual_r2 = results['residual_r2']
                
                print(f"\nMediation analysis results for {dataset_name} - {outcome}:")
                print(f"  Network properties explain {network_r2:.4f} of variance")
                print(f"  Graph types explain {residual_r2:.4f} of the residual variance")
                
                if residual_r2 < 0.01:
                    print("  Interpretation: Network properties fully mediate graph type effects")
                elif residual_r2 < 0.05:
                    print("  Interpretation: Network properties largely mediate graph type effects")
                else:
                    print("  Interpretation: Network properties partially mediate graph type effects")
                
            except Exception as e:
                print(f"Error analyzing {dataset_name} - {outcome}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nCompleted all mediation analyses in {total_time:.1f} seconds")
    
    return all_results

if __name__ == "__main__":
    study_name = "aiteams01nm_20250128_223001"
    results = run_all_residual_analyses(study_name)