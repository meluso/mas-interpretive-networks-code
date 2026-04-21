#!/usr/bin/env python3
# figures/plot_graph_property_heatmap.py

"""
Create a heatmap visualization of network properties by graph type.
Displays graph types (rows) vs network metrics (columns) with color representing values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import figure settings utilities
from util import set_fonts, fig_size, save_pub_fig, set_border

# Set up fonts
set_fonts()

def load_network_properties(study_name):
    """Load network properties by graph data."""
    base_path = Path(f"data/results/{study_name}/network_properties")
    file_path = base_path / "network_metrics_by_graph.parquet"
    return pd.read_parquet(file_path)

def prepare_heatmap_data(df, team_size=8):
    """Prepare data for the heatmap visualization, filtering to specific team size."""
    # Filter to specific team size
    df_filtered = df[df['team_size'] == team_size].copy()
    
    # Get mean metrics (average across instances of each graph type)
    metrics_columns = ['team_graph_density_mean', 
                       'team_graph_centrality_eigenvector_mean_mean',
                       'team_graph_pathlength_mean',
                       'team_graph_centrality_eigenvector_stdev_mean',
                       'team_graph_clustering_mean',
                       'team_graph_nearest_neighbor_degree_stdev_mean',
                       'team_graph_nearest_neighbor_degree_mean_mean',
                       'team_graph_centrality_betweenness_mean_mean',
                       'team_graph_assortativity_mean',
                       'team_graph_centrality_betweenness_stdev_mean',
                       'team_graph_diameter_mean',
                       'team_graph_centrality_degree_stdev_mean']
    
    # Rename columns to simpler format for display
    column_names = {
        'team_graph_density_mean': 'Graph Density',
        'team_graph_centrality_eigenvector_mean_mean': 'Mean Eigenvector Centrality',
        'team_graph_pathlength_mean': 'Mean Shortest Path Length',
        'team_graph_centrality_eigenvector_stdev_mean': 'St. Dev. of Eigenvector Centrality',
        'team_graph_clustering_mean': 'Clustering Coefficient',
        'team_graph_nearest_neighbor_degree_stdev_mean': 'St. Dev. of Nearest Neighbor Degree',
        'team_graph_nearest_neighbor_degree_mean_mean': 'Mean Nearest Neighbor Degree',
        'team_graph_centrality_betweenness_mean_mean': 'Mean Betweenness Centrality',
        'team_graph_assortativity_mean': 'Degree Assortativity',
        'team_graph_centrality_betweenness_stdev_mean': 'St. Dev. of Betweenness Centrality',
        'team_graph_diameter_mean': 'Graph Diameter',
        'team_graph_centrality_degree_stdev_mean': 'St. Dev. of Degree Centrality'
    }
    
    # Select and rename columns
    pivot_df = df_filtered[['graph_slug'] + metrics_columns].copy()
    pivot_df = pivot_df.rename(columns=column_names)
    
    # Sort by density and mean eigenvector centrality
    pivot_df = pivot_df.sort_values(by=['Graph Density', 'Mean Eigenvector Centrality'], 
                                    ascending=[False, False])
    
    # Pivot to get graphs as rows and metrics as columns
    graph_slugs = pivot_df['graph_slug'].values
    pivot_df = pivot_df.drop(columns=['graph_slug'])
    
    # Normalize each column to [0, 1] scale for better visualization
    for col in pivot_df.columns:
        min_val = pivot_df[col].min()
        max_val = pivot_df[col].max()
        if max_val > min_val:  # Avoid division by zero
            pivot_df[col] = (pivot_df[col] - min_val) / (max_val - min_val)
    
    # Order columns as specified in the figure
    ordered_cols = [
        'Graph Density',
        'Mean Eigenvector Centrality',
        'Mean Shortest Path Length',
        'St. Dev. of Eigenvector Centrality',
        'Clustering Coefficient',
        'St. Dev. of Nearest Neighbor Degree',
        'Mean Nearest Neighbor Degree',
        'Mean Betweenness Centrality',
        'Degree Assortativity',
        'St. Dev. of Betweenness Centrality',
        'Graph Diameter',
        'St. Dev. of Degree Centrality'
    ]
    
    # Ensure all columns are in the dataframe
    ordered_cols = [col for col in ordered_cols if col in pivot_df.columns]
    pivot_df = pivot_df[ordered_cols]
    
    return pivot_df, graph_slugs

def simplify_graph_names(graph_slugs):
    """Create simplified graph names for display."""
    simplified = []
    for slug in graph_slugs:
        # Extract base graph type
        parts = slug.split('_')
        base_type = parts[0]
        
        # Format parameters if present
        params = []
        for part in parts[1:]:
            if part.startswith('p'):
                # Format probability (p03 -> p=0.3)
                value = part[1:]
                if len(value) == 2:
                    value = '0.' + value
                params.append(f"p={value}")
            elif part.startswith('k'):
                # Format k parameter
                params.append(f"k={part[1:]}")
            elif part.startswith('m'):
                # Format m parameter
                params.append(f"m={part[1:]}")
        
        # Combine parts
        if params:
            simplified.append(f"{base_type} ({', '.join(params)})")
        else:
            simplified.append(base_type)
    
    return simplified

def plot_network_heatmap(df, graph_names, output_path=None):
    """Create heatmap visualization of network properties by graph type."""
    # Set figure size
    plt.figure(figsize=fig_size(frac_width=1, frac_height=0.7))
    
    # Create custom colormap that matches project aesthetics
    # Use white/light gray for 0, dark green for 1, with smooth gradient
    from matplotlib.colors import LinearSegmentedColormap
    
    # Option 1: White to dark green (matches main effects color)
    colors = ["#f9f9f9", "#c5e0b4", "#71A33F", "#3c6b1f"]
    cmap = LinearSegmentedColormap.from_list("green_gradient", colors, N=100)
    
    # Create heatmap
    ax = sns.heatmap(df, cmap=cmap, annot=False, linewidths=0.5, 
                     cbar_kws={'label': 'Normalized Value'})
    
    # Set y-axis labels with adjusted alignment
    ax.set_yticklabels(graph_names, ha='right', rotation=0)
    
    # Set and rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add title
    plt.title('Network Properties by Graph Type')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return ax

def create_network_heatmap(study_name, team_size=8):
    """Main function to create network heatmap for a study."""
    # Load network properties data
    df = load_network_properties(study_name)
    
    # Prepare data for visualization
    heatmap_data, graph_slugs = prepare_heatmap_data(df, team_size)
    
    # Create simplified graph names
    graph_names = simplify_graph_names(graph_slugs)
    
    # Create visualization
    ax = plot_network_heatmap(heatmap_data, graph_names)
    
    # Save figure
    figure_name = f"network_heatmap_team{team_size}"
    save_pub_fig(figure_name)
    
    return ax

if __name__ == "__main__":
    # Example usage
    study_name = "aiteams01nm_20250128_223001"
    create_network_heatmap(study_name, team_size=8)
    plt.show()