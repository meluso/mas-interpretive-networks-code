# analysis/create_dataset.py
"""Dataset creation utilities for AI teams analysis.

This module handles the creation of standardized datasets for analysis:
- Dataset 1: Includes all graph types with main network metrics only
- Dataset 2: Includes only graphs with valid values for all metrics

Both datasets apply standard transformations: log transforms for agent_steplim 
and team_fn_diff_peaks, and MinMax normalization for continuous variables.
"""

import dask.dataframe as dd
import dask.array as da
from pathlib import Path
from dask_ml.preprocessing import MinMaxScaler

# Import utility functions
from analysis.util import (
    get_task_difficulties,
    get_outcome_metrics,
    get_control_variables,
    get_all_network_metrics,
    get_main_network_metrics,
    get_special_network_metrics,
    get_variables_to_log_transform,
    get_categorical_variables
)

def create_dataset(study_name: str, drop_nans: bool = False) -> Path:
    """Create and save analysis-ready dataset with preprocessing applied."""
    # Set up paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw" / study_name / "trials"
    processed_dir = data_dir / "preprocessed" / study_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output file path
    output_name = f"dataset{'2' if drop_nans else '1'}"
    output_file = processed_dir / f"{output_name}.parquet"
    
    print(f"Creating {output_name} for {study_name}")
    
    # Get column groups
    task_metrics = get_task_difficulties(transformed=False)
    outcome_metrics = get_outcome_metrics()
    controls = get_control_variables(transformed=False)
    categorical_vars = get_categorical_variables()
    log_vars = get_variables_to_log_transform()
    
    # Determine which network metrics to include
    if drop_nans:
        # Dataset 2: Include all metrics but will drop rows with NaNs
        network_metrics = get_all_network_metrics(transformed=False)
    else:
        # Dataset 1: Include only main metrics, ignore special metrics with NaNs
        network_metrics = get_main_network_metrics(transformed=False)
    
    # Combine all columns to load (include trial_id for reference)
    columns_to_load = ['trial_id'] + controls + task_metrics + network_metrics + outcome_metrics + categorical_vars
    
    # Load the data
    df = dd.read_parquet(raw_dir / "*.parquet", columns=columns_to_load)
    
    # Filter out teams of size 1
    print("\tFiltering out teams of size 1...")
    df = df[df['team_size'] > 1]
    
    # Process based on dataset type - do this early to reduce computation
    if drop_nans:
        # Dataset 2: Drop rows with NaN values in special metrics
        special_metrics = get_special_network_metrics(transformed=False)
        df = df.dropna(subset=special_metrics)
    
    # 1. Apply log transformations to specified variables
    print("\tApplying log transformations...")
    for var in log_vars:
        df[f'log_{var}'] = da.log10(df[var])
    
    # 2. Apply MinMax normalization to continuous features using dask-ml
    print("\tApplying normalization...")
    
    # Filter continuous variables to only those in the dataframe
    # Exclude categorical variables from normalization
    continuous_vars = [var for var in list(set(controls + network_metrics + task_metrics) - set(log_vars) - set(categorical_vars)) 
                      if var in df.columns]
    
    # Process all continuous variables at once with MinMaxScaler
    X = df[continuous_vars]
    
    # Create and fit scaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    # Transform all columns
    normalized_df = scaler.transform(X)
    
    # Add normalized columns to original dataframe with norm_ prefix
    for var in continuous_vars:
        df[f'norm_{var}'] = normalized_df[var]

    # Drop original continuous columns since not needed, but keep categorical variables
    df = df.drop(log_vars + continuous_vars, axis=1)
    
    # Save data to processed directory with compression
    print(f"\tSaving preprocessed dataset to {output_file}...")
    df.to_parquet(
        output_file,
        compression='zstd',
        compression_level=3,
        write_index=False
    )
    
    print(f"Dataset {output_name} saved to {output_file}")
    
    return output_file

def create_all_datasets(study_name):
    """Create all dataset variants."""
    
    # Define datasets
    dataset2nanopt = {
        'Dataset1': False,
        'Dataset2': True
        }
    
    # Create list of output files
    output_files = []
    
    # Create datasets
    for _nan_opt in dataset2nanopt.values():
        output_files.append(create_dataset(study_name, drop_nans=_nan_opt))
    
    return output_files

if __name__ == "__main__":
    
    # Specify dataset
    study_name = "aiteams01da_20251202_192855"
    
    # Create both datasets
    output_files = create_all_datasets(study_name)
    
    # Print output files
    from pprint import pprint
    pprint(output_files)