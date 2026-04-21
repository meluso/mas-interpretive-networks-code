# analysis/network_properties_by_graph.py
"""Calculate network metrics by graph type and team size.

This module analyzes how different graph structures (graph_slug) exhibit 
different network properties across team sizes. Results are saved as:

data/results/{study_name}/network_properties/network_metrics_by_graph.parquet

This analysis helps explain why certain graph types perform differently
on various tasks by showing their inherent network properties.
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging

# Import from config
from config.metrics import NETWORK_METRICS

# Create logger
logger = logging.getLogger(__name__)

def calculate_network_properties(study_name):
    """Calculate network property statistics by graph type and team size.
    
    Args:
        study_name: Name of study to analyze
    
    Returns:
        Path to saved results file
    """
    start_time = time.time()
    
    # Set up paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw" / study_name / "trials"
    results_dir = data_dir / "results" / study_name / "network_properties"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "network_metrics_by_graph.parquet"
    
    print(f"Calculating network properties for {study_name}")
    
    # Load only necessary columns to reduce memory usage
    columns = ['graph_slug', 'team_size'] + list(NETWORK_METRICS.keys())
    
    # Read dataset using dask
    print("Loading dataset with dask...")
    df = dd.read_parquet(raw_dir / "*.parquet", columns=columns)
    
    # Filter out teams of size 1 (no meaningful network properties)
    df = df[df['team_size'] > 1]
    
    # Basic info
    n_partitions = df.npartitions
    print(f"Dataset has {n_partitions} partitions")
    
    # Group by graph_slug and team_size and calculate statistics
    print("Calculating statistics...")
    
    # Define aggregation functions for each network metric
    agg_dict = {metric: ['mean', 'std', 'min', 'max', 'median'] for metric in NETWORK_METRICS}
    
    # Perform aggregation
    stats = df.groupby(['graph_slug', 'team_size']).agg(agg_dict)
    
    # Compute and flatten the multi-index columns
    print("Computing results...")
    result = stats.compute()
    
    # Flatten the column multi-index for easier access
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    
    # Reset index to make graph_slug and team_size regular columns
    result = result.reset_index()
    
    # Add count of observations for each group
    counts = df.groupby(['graph_slug', 'team_size']).size().compute()
    result['count'] = result.apply(lambda row: counts.loc[row['graph_slug'], row['team_size']], axis=1)
    
    # Save to parquet
    print(f"Saving results to {output_file}")
    result.to_parquet(
        output_file,
        compression='zstd',
        compression_level=3,
        index=False
    )
    
    # Print summary
    print(f"\nResults summary:")
    print(f"Network properties by graph type: {len(result)} combinations")
    print(f"Graph types: {result['graph_slug'].nunique()}")
    print(f"Team sizes: {sorted(result['team_size'].unique())}")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.1f} seconds")
    
    return output_file

if __name__ == "__main__":
    # study_name = "aiteams01nm_20250128_223001"
    study_name = "aiteams01rw_20250321_215818"
    calculate_network_properties(study_name)