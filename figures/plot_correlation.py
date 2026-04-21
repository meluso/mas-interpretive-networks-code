# figures/plot_correlation.py
"""Plot correlation matrix heatmap for AI teams analysis.

Creates a heatmap visualization of correlation coefficients with
statistical significance masking.

Usage from iPython:
    from figures.plot_correlation import plot_correlation_matrix
    plot_correlation_matrix('aiteams01nm_20250128_223001')
"""

import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Import figure settings utilities
from figures.util import set_fonts, fig_size, save_pub_fig, set_border

def load_correlation_data(study_name):
    """Load correlation and p-value matrices from disk."""
    data_dir = Path("data")
    input_dir = data_dir / "processed" / study_name / "correlation"
    
    # Load correlation matrix and p-values
    corr_matrix = pd.read_parquet(input_dir / "correlation_matrix.parquet")
    pval_matrix = pd.read_parquet(input_dir / "pvalue_matrix.parquet")
    
    return corr_matrix, pval_matrix

def plot_correlation_matrix(study_name, save=False, significance_level=0.01):
    """Create and display correlation matrix heatmap."""
    # Set up fonts
    set_fonts()
    
    # Load data
    corr_matrix, pval_matrix = load_correlation_data(study_name)
    
    # Create labels
    var2tick = {
        'agent_num_vars': 'Num. Variables per Agent',
        'agent_steplim': 'Rationality Bounds (linear)',
        'convergence_performance': 'Performance (at Convergence)',
        'convergence_step': 'Number of Steps (at Convergence)',
        'final_performance': 'Performance (at 100 Steps)',
        'log_agent_steplim': 'Rationality Bounds (log)',
        'log_team_fn_diff_peaks': 'Choose Task Difficulty (log)',
        'team_fn_diff_alignment': 'Negotiate Task Difficulty',
        'team_fn_diff_integral': 'Generate Task Difficulty',
        'team_fn_diff_interdep': 'Coordinate Task Difficulty',
        'team_fn_diff_peaks': 'Choose Task Difficulty (linear)',
        'team_graph_assortativity': 'Degree Assortativity',
        'team_graph_centrality_betweenness_mean': 'Betweenness Centrality (mean)',
        'team_graph_centrality_betweenness_stdev': 'Betweenness Centrality (std.)',
        'team_graph_centrality_degree_mean': 'Degree Centrality (mean)',
        'team_graph_centrality_degree_stdev': 'Degree Centrality (std.)',
        'team_graph_centrality_eigenvector_mean': 'Eignevector Centrality (mean)',
        'team_graph_centrality_eigenvector_stdev': 'Eigenvector Centrality (std.)',
        'team_graph_clustering': 'Clustering Coefficient',
        'team_graph_density': 'Density',
        'team_graph_diameter': 'Graph Diameter',
        'team_graph_nearest_neighbor_degree_mean': 'Nearest Neighbor Degree (mean)',
        'team_graph_nearest_neighbor_degree_stdev': 'Nearest Neighbor Degree (std.)',
        'team_graph_pathlength': 'Shortest Path Length (mean)',
        'team_size': 'Team Size',
        'trial_id': 'Trial ID'
     }
    
    # Rename columns and indexes
    corr_matrix = corr_matrix.rename(index=var2tick, columns=var2tick)
    pval_matrix = pval_matrix.rename(index=var2tick, columns=var2tick)
    
    # Generate masks for significance
    mask_signif = np.triu(np.ones_like(corr_matrix, dtype=bool)) | (pval_matrix > significance_level)
    
    # Create figure
    plt.figure(figsize=fig_size(frac_width=1.2, frac_height=0.75))
    
    # Plot heatmap with significance masking
    heatmap = sns.heatmap(corr_matrix, cmap='PRGn', vmin=-1, vmax=1, center=0, 
                annot=True, fmt='.2f', square=True, mask=mask_signif, 
                annot_kws={'size': 6})
    heatmap.set_ylabel('')
    
    # Format plot
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save if requested
    if save:
        save_pub_fig(f"correlation_matrix")
    
    plt.show()
    
    return plt.gcf()

if __name__ == '__main__':
    study_name = 'aiteams01nm_20250128_223001'
    plot_correlation_matrix(study_name, save=True)