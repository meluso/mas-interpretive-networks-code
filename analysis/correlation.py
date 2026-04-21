# analysis/correlation.py
"""Calculate correlation matrix for AI teams analysis.

Computes correlation coefficients and p-values for all numeric variables,
with log transformations applied to exponentially scaled variables.
Results are saved to parquet files.

Usage from iPython:
    from analysis.correlation import calculate_correlations
    calculate_correlations('aiteams01nm_20250128_223001')
"""

import numpy as np
import dask.dataframe as dd
from pathlib import Path
import pandas as pd
from scipy import stats

def transform_variables(df):
    """Add log-transformed versions of exponential variables."""
    return df.assign(
        log_agent_steplim=np.log10(df.agent_steplim),
        log_team_fn_diff_peaks=np.log10(df.team_fn_diff_peaks)
    )

def calculate_pvalues(corr_matrix, n_samples):
    """Calculate p-values for correlation matrix using t-distribution."""
    t_stat = corr_matrix * np.sqrt((n_samples-2)/(1-corr_matrix**2))
    pvalues = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples-2))
    return pvalues

def calculate_correlations(study_name):
    """Calculate correlation matrices and p-values, saving results to disk."""
    # Set up paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw" / study_name / "trials"
    output_dir = data_dir / "results" / study_name / "correlation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and transform data
    df = dd.read_parquet(raw_dir / "*.parquet")
    df = transform_variables(df)
    
    # Calculate correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    
    # Calculate p-values after computing correlation
    computed_corr = corr_matrix.compute()
    n_samples = len(df)
    pvalues = calculate_pvalues(computed_corr.values, n_samples)
    
    # Create DataFrame for p-values with same index/columns as correlation
    pvalue_df = pd.DataFrame(
        pvalues, 
        index=computed_corr.index, 
        columns=computed_corr.columns
    )
    
    # Save results
    corr_matrix.to_parquet(output_dir / "correlation_matrix.parquet")
    pvalue_df.to_parquet(output_dir / "pvalue_matrix.parquet")
    
    print(f"Correlation analysis complete for {study_name}")
    print(f"Results saved to {output_dir}")
    
    return computed_corr, pvalue_df

if __name__ == '__main__':
    study_name = 'aiteams01nm_20250128_223001'
    calculate_correlations(study_name)