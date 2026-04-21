# figures/plot_permutation_importances.py
"""Plot permutation importance results from random forest analysis.

Creates publication-quality plots showing feature importance with confidence intervals.

Usage from iPython:
    from figures.plot_permutation_importances import plot_permutation_importances
    plot_permutation_importances(study_name='aiteams01nm_20250128_223001', 
                                outcome='convergence_performance', 
                                dataset='dataset2')
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import figure settings utilities
from figures.util import set_fonts, fig_size, save_pub_fig, set_border

def load_permutation_importance(study_name, outcome, dataset):
    """Load permutation importance results from disk."""
    data_dir = Path("data/results") / study_name / "random_forest" / f"{dataset}_{outcome}"
    importance_file = data_dir / "permutation_importance.csv"
    
    if not importance_file.exists():
        raise FileNotFoundError(f"No permutation importance file found at {importance_file}")
    
    return pd.read_csv(importance_file)

def plot_permutation_importances(study_name='aiteams01nm_20250128_223001', 
                               outcome='convergence_performance', 
                               dataset='dataset2',
                               top_n=15,
                               save=False):
    """Plot permutation importance with confidence intervals.
    
    Args:
        study_name: Name of the study
        outcome: Target outcome variable
        dataset: Dataset name (dataset1 or dataset2)
        top_n: Number of top features to display
        save: Whether to save the figure to disk
    """
    # Set up fonts
    set_fonts()
    
    # Load importance data
    importance_df = load_permutation_importance(study_name, outcome, dataset)
    
    # Sort by importance (should already be sorted, but just to be sure)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Get top N features or all if fewer
    top_n = min(top_n, len(importance_df))
    top_features = importance_df.head(top_n)
    
    # Create plot with appropriate aspect ratio
    plt.figure(figsize=fig_size(frac_width=0.8, frac_height=0.6))
    
    # Define colors
    significant_color = "#71A33F"  # Green
    non_significant_color = "#B3B3B3"  # Gray
    error_bar_color = "#B3B3B3"  # Gray
    
    # Reverse the order of features for plotting (highest at top)
    y_pos = range(top_n-1, -1, -1)
    
    # Prepare data
    importances = top_features['Importance'].values
    lower_ci = top_features['Lower_CI'].values
    upper_ci = top_features['Upper_CI'].values
    
    # Plot error bars (all in gray)
    plt.errorbar(
        importances, 
        y_pos,
        xerr=np.vstack([
            importances - lower_ci,
            upper_ci - importances
        ]),
        fmt='none',  # No central marker
        ecolor=error_bar_color,
        capsize=5,
        zorder=1
    )
    
    # Labels
    var2label = {
        'norm_team_fn_diff_integral': 'Generate Task Difficulty',
        'log_agent_steplim': 'Rationality Bounds',
        'log_team_fn_diff_peaks': 'Choose Task Difficulty',
        'norm_team_graph_density': 'Density',
        'norm_team_fn_diff_alignment': 'Negotiate Task Difficulty',
        'norm_team_graph_centrality_eigenvector_mean': 'Eignevector Centrality (mean)',
        'norm_team_size': 'Team Size',
        'norm_team_graph_centrality_betweenness_stdev': 'Betweenness Centrality (std.)',
        'norm_team_graph_pathlength': 'Shortest Path Length (mean)',
        'norm_team_fn_diff_interdep': 'Coordinate Task Difficulty',
        'norm_team_graph_clustering': 'Clustering Coefficient',
        'norm_team_graph_nearest_neighbor_degree_mean': 'Nearest Neighbor Degree (mean)',
        'graph_slug_small_world_k4_p03': 'Small World ($k=4$, $p=0.3$)',
        'graph_slug_random_p03': 'Random Graph ($p=0.3$)',
        'graph_slug_star': 'Star Graph',
     }
    
    # Plot points with conditional coloring
    for i, row in enumerate(top_features.itertuples()):
        color = significant_color if row.Significant else non_significant_color
        plt.scatter(row.Importance, y_pos[i], color=color, s=50, zorder=2)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, zorder=0)
    
    # Labels and formatting
    plt.yticks(y_pos, [var2label[var] for var in top_features['Feature'].values])
    plt.xlabel('Permutation Importance')
    # plt.title(f'Feature Importance (for Convergence Performance)')
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    # Remove top and right spines
    set_border(plt.gca(), top=False, right=False)
    
    # Use log scale for x-axis if all values are positive
    if all(importances > 0) and all(lower_ci > 0):
        plt.xscale('log')
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        save_pub_fig(f"permutation_importance_{dataset}_{outcome}")
    
    return plt.gcf()

if __name__ == '__main__':
    study_name = 'aiteams01nm_20250128_223001'
    outcome = 'convergence_performance'
    dataset = 'dataset2'
    
    plot_permutation_importances(study_name, outcome, dataset, save=True)
    plt.show()