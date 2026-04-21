# analysis/util.py

"""Utility functions for AI teams analysis."""

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as st
from typing import Dict, List, Tuple, Union

# Import from config modules
from config.metrics import (
    OUTCOME_METRICS, 
    NETWORK_METRICS, 
    OBJECTIVE_METRICS,
    VARIABLES_TO_LOG_TRANSFORM,
    SPECIAL_NETWORK_METRICS
)
from config.parameter_metadata import PARAMETER_METADATA

def get_variables_to_log_transform() -> List[str]:
    """Return list of variables that need log transformation."""
    return VARIABLES_TO_LOG_TRANSFORM

def get_normalized_variable_names(variables: List[str]) -> List[str]:
    """Return normalized variable names for a list of original variables."""
    log_vars = get_variables_to_log_transform()
    return [f'log_{var}' if var in log_vars else f'norm_{var}' for var in variables]

def get_all_network_metrics(transformed: bool = True) -> List[str]:
    """Get all network metrics from config.metrics, optionally with transformation prefixes."""
    # Exclude team_graph_centrality_degree_mean to avoid collinearity with density
    metrics = [m for m in NETWORK_METRICS if m != 'team_graph_centrality_degree_mean']
    return get_normalized_variable_names(metrics) if transformed else metrics

def get_special_network_metrics(transformed: bool = True) -> List[str]:
    """Get network metrics that may have validity constraints."""
    return get_normalized_variable_names(SPECIAL_NETWORK_METRICS) if transformed else SPECIAL_NETWORK_METRICS

def get_main_network_metrics(transformed: bool = True) -> List[str]:
    """Get network metrics valid across all graph types."""
    # Exclude team_graph_centrality_degree_mean from the main metrics
    special_metrics = get_special_network_metrics(transformed=False)
    all_metrics = [m for m in NETWORK_METRICS if m != 'team_graph_centrality_degree_mean']
    main_metrics = [m for m in all_metrics if m not in special_metrics]
    return get_normalized_variable_names(main_metrics) if transformed else main_metrics

def get_task_difficulties(transformed: bool = True) -> List[str]:
    """Get task difficulty metrics from config.metrics."""
    return get_normalized_variable_names(list(OBJECTIVE_METRICS.keys())) if transformed else list(OBJECTIVE_METRICS.keys())

def get_outcome_metrics() -> List[str]:
    """Get outcome metrics from config.metrics."""
    return [metric for metric in OUTCOME_METRICS if metric.startswith("convergence_")]

def get_categorical_variables() -> List[str]:
    """Get categorical variables that should be retained without transformation."""
    return ['graph_slug']

def get_control_variables(transformed: bool = True, include_slugs: bool = False) -> List[str]:
    """Get core control variables."""
    controls = ['team_size','agent_steplim']
    if include_slugs: controls.extend(get_categorical_variables())
    return get_normalized_variable_names(controls) if transformed else controls

def get_network_metrics(include_special_metrics: bool = True, transformed: bool = True) -> List[str]:
    """Get list of network metrics based on requirements."""
    return get_all_network_metrics(transformed) if include_special_metrics else get_main_network_metrics(transformed)

def load_parquet_dataset(dataset_path):
    """Load dataset from parquet file."""
    return pd.read_parquet(dataset_path)

# Existing functions remain the same
def conf_int(diff_means, s1, n1, s2, n2, alpha=0.05):
    """Calculate lower and upper confidence interval limits."""
    # degrees of freedom
    df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))  
    
    # t-critical value for CI
    t = st.t.ppf(1 - alpha/2, df)
    
    # Range of difference of means
    lower = diff_means - t * np.sqrt(s1**2 / n1 + s2**2 / n2)
    upper = diff_means + t * np.sqrt(s1**2 / n1 + s2**2 / n2)
    
    return lower, upper