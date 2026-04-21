# analysis/network_property_regression_effects.py
"""Analysis of network property effects on team performance using ridge regression.

Implements Analysis 2.1: Regression models examining how network metrics
affect team performance across different task types.
"""

import dask.dataframe as dd
import gc
import time
from pathlib import Path

# Import from our core regression module
from analysis.ridge_regression import run_ridge_regression, format_regression_results

# Import utility functions
from analysis.util import (
    get_task_difficulties,
    get_outcome_metrics,
    get_control_variables,
    get_network_metrics
)

def create_network_property_models(dataset_name):
    """Define models for network property analysis.
    
    Each model has base_features and interaction terms structured for increasing complexity.
    """
    # Include special metrics for dataset2 only
    include_special_metrics = (dataset_name == "dataset2")
    
    # Get feature sets with transform prefixes
    task_difficulties = get_task_difficulties()
    network_metrics = get_network_metrics(include_special_metrics)
    base_controls = get_control_variables()
    
    # Build top network metrics
    top_network_metrics \
        = [
            'norm_team_graph_pathlength'
            ] if include_special_metrics else [] \
        + [
            'norm_team_graph_density',
            'norm_team_graph_centrality_eigenvector_mean'
            ]
    
    # Define model formulas
    models = {
        'model1': {
            'base_features': base_controls + task_difficulties,
            'interactions': [],
            'name': 'Model 1: Controls + Tasks'
        },
        'model2': {
            'base_features': base_controls + task_difficulties + network_metrics,
            'interactions': [],
            'name': 'Model 2: Model 1 + Networks'
        },
        'model3': {
            'base_features': base_controls + task_difficulties + network_metrics,
            'interactions': [(n, t) for n in network_metrics for t in task_difficulties],
            'name': 'Model 3: Model 2 + Interactions'
        },
        'model4': {
            'base_features': base_controls + task_difficulties + top_network_metrics,
            'interactions': [(n, t) for n in top_network_metrics for t in task_difficulties],
            'name': 'Model 4: Reduced Features'
        }
    }
    
    return models

def run_network_property_analysis(study_name, dataset_name, outcome):
    """Run network property analysis for a specific dataset and outcome.
    
    Args:
        study_name: Name of the study
        dataset_name: Name of the dataset (dataset1 or dataset2)
        outcome: Target outcome variable name
    
    Returns:
        Dict containing model results and formatted coefficients
    """
    # Set up paths
    data_dir = Path("data")
    dataset_path = data_dir / "preprocessed" / study_name / f"{dataset_name}.parquet"
    results_dir = data_dir / "results" / study_name / "network_property_effects"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing network property effects for {outcome} with {dataset_name}")
    
    # Load dataset
    ddf = dd.read_parquet(dataset_path)
    
    # Get model specifications
    model_specs = create_network_property_models(dataset_name)
    
    # Run models
    models_results = {}
    
    for model_id, model_spec in model_specs.items():
        print(f"  Running {model_spec['name']}")
        
        # Run regression using model formula
        results = run_ridge_regression(
            ddf, 
            outcome, 
            model_spec['base_features'], 
            model_spec['interactions']
        )
        
        # Add model name
        results['model_name'] = model_spec['name']
        models_results[model_id] = results
        
        # Clear memory
        gc.collect()
    
    # Format results
    formatted = format_regression_results(models_results)
    
    # Save results
    result_dir = results_dir / f"{dataset_name}_{outcome}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    formatted['comparison'].to_csv(result_dir / "model_comparison.csv", index=False)
    formatted['coefficients'].reset_index().to_csv(result_dir / "coefficient_comparison.csv", index=False)
    
    print(f"  Results saved to {result_dir}")
    
    return {
        'models': models_results,
        'formatted': formatted
    }

def run_all_network_property_analyses(study_name):
    """Run network property analyses for all datasets and outcomes."""
    start_time = time.time()
    
    # Define datasets and outcomes
    datasets = ["dataset1", "dataset2"]
    outcomes = get_outcome_metrics()
    
    all_results = {}
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        
        for outcome in outcomes:
            try:
                results = run_network_property_analysis(study_name, dataset_name, outcome)
                all_results[dataset_name][outcome] = results
                
                # Log summary statistics
                comparison_df = results['formatted']['comparison']
                print(f"\nSummary of model performance for {dataset_name} - {outcome}:")
                print(comparison_df)
                
            except Exception as e:
                print(f"Error analyzing {dataset_name} - {outcome}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\nCompleted all network property analyses in {total_time:.1f} seconds")
    
    return all_results

if __name__ == "__main__":
    study_name = "aiteams01nm_20250128_223001"
    results = run_all_network_property_analyses(study_name)