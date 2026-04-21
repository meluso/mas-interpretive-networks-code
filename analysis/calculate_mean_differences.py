#!/usr/bin/env python3
# analysis/calculate_mean_differences.py
"""
Calculate relative descriptive statistics for AI teams by comparing graph types 
to baseline performance across team sizes and agent step limits.
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Import utility functions
from analysis.util import (
    get_outcome_metrics,
    conf_int
)

def calculate_relative_descriptive_statistics(study_name: str) -> Path:
    """
    Calculate relative performance statistics comparing different graph types
    to baseline measurements, grouped by team size and agent step limit.
    
    Args:
        study_name: Name of the study to analyze
        
    Returns:
        Path to the saved results file
    """
    start_time = time.time()
    
    # Set up paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw" / study_name / "trials"
    results_dir = data_dir / "results" / study_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "mean_differences.csv"
    
    print(f"Calculating relative descriptive statistics for {study_name}")
    
    # Get outcome metrics
    outcome_metrics = get_outcome_metrics()
    
    # Load only necessary columns to reduce memory usage
    essential_columns = [
        'team_size', 
        'agent_steplim', 
        'graph_slug',
        'team_graph_density',
        'team_graph_centrality_eigenvector_mean',
        *outcome_metrics
    ]
    
    # Read dataset using dask
    print("Loading dataset with dask...")
    df = dd.read_parquet(raw_dir / "*.parquet", columns=essential_columns)
    
    # Filter out teams of size 1 (these have no graph structure comparison)
    df = df[df['team_size'] > 1]
    
    print("Calculating baseline statistics (grouped by team_size and agent_steplim)...")
    
    # Group A: Calculate baseline statistics by team_size and agent_steplim
    baseline_stats = df.groupby(['team_size', 'agent_steplim'])[outcome_metrics].agg(['mean','count','std']).compute()
    
    print("Calculating statistics by graph type...")
    
    # Group B: Calculate statistics by team_size, agent_steplim, and graph_slug
    group_stats = df.groupby(['team_size', 'agent_steplim', 'graph_slug'])
    
    # Calculate mean, count, and std for each outcome metric + density
    mets_to_get = outcome_metrics + ['team_graph_density', 'team_graph_centrality_eigenvector_mean']
    graph_stats = group_stats[mets_to_get].agg(['mean', 'count', 'std']).compute()
    
    # Prepare result dataframe
    results = []
    
    print("Calculating relative differences and confidence intervals...")
    
    # Calculate differences and confidence intervals
    for idx, row in graph_stats.iterrows():
        team_size, agent_steplim, graph_slug = idx
        
        # Get baseline values for this team_size and agent_steplim
        baseline_idx = (team_size, agent_steplim)
        baseline_values = baseline_stats.loc[baseline_idx]
        
        # Prepare row data
        row_data = {
            'team_size': team_size,
            'agent_steplim': agent_steplim,
            'graph_slug': graph_slug,
            'count': row[f'{outcome_metrics[0]}', 'count'],  # Use count from first metric
            'density': row['team_graph_density', 'mean'],
            'centrality_eigenvector_mean': row['team_graph_centrality_eigenvector_mean', 'mean']
        }
        
        # Calculate statistics for each outcome metric
        for metric in outcome_metrics:
            # Get values for this metric
            metric_mean = row[metric, 'mean']
            metric_std = row[metric, 'std']
            metric_count = row[metric, 'count']
            
            # Get baseline values
            baseline_mean = baseline_values[metric, 'mean']
            baseline_std = baseline_values[metric, 'std']
            baseline_count = baseline_values[metric, 'count']
            
            # Calculate difference from baseline
            difference = metric_mean - baseline_mean
            
            # Calculate 95% confidence interval for the difference
            lower_ci, upper_ci = conf_int(
                difference, 
                metric_std, metric_count,
                baseline_std, baseline_count,  # Now using correct baseline values
                alpha=0.05
            )
            
            # Add to row data
            row_data[f'{metric}_mean'] = metric_mean
            row_data[f'{metric}_std'] = metric_std
            row_data[f'{metric}_diff'] = difference
            row_data[f'{metric}_lower_ci'] = lower_ci
            row_data[f'{metric}_upper_ci'] = upper_ci
            row_data[f'{metric}_significant'] = (lower_ci > 0) or (upper_ci < 0)
        
        results.append(row_data)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by team_size, agent_steplim, graph_slug
    results_df = results_df.sort_values(['team_size', 'agent_steplim', 'graph_slug'])
    
    # Save to CSV
    print(f"Saving results to {output_file}")
    results_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\nResults summary:")
    print(f"Total graph type configurations: {len(results_df)}")
    print(f"Team sizes: {sorted(results_df['team_size'].unique())}")
    print(f"Agent step limits: {sorted(results_df['agent_steplim'].unique())}")
    print(f"Graph types: {len(results_df['graph_slug'].unique())}")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.1f} seconds")
    
    return output_file

if __name__ == "__main__":
    study_name = "aiteams01nm_20250128_223001"  # Use the most recent study
    output_file = calculate_relative_descriptive_statistics(study_name)
    print(f"Results saved to: {output_file}")