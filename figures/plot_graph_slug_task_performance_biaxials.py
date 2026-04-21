# figures/plot_graph_slug_task_performance_biaxials.py
"""Plot biaxial graphs showing task conditional effects by graph type."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path
import matplotlib.cm as cm

# Import figure settings utilities
from figures.util import (
    set_fonts, set_border, arrow,
    filter_by_significance, GrayGn
)


def load_graph_properties(study_name):
    """Load network properties by graph data."""
    base_path = Path(f"data/results/{study_name}/network_properties")
    file_path = base_path / "network_metrics_by_graph.parquet"
    return pd.read_parquet(file_path)


def load_graph_effects(study_name):
    """Load OLS regression results for graph types."""
    results_dir = Path(f"data/results/{study_name}/ols_regressions")
    file_path = results_dir / "graph_slug_regressions.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"OLS regression results not found at {file_path}")
        
    return pd.read_csv(file_path)
    

def calculate_conditional_effects(graph_props, graph_effects, task, reference_slug='complete', 
                               density_col='team_graph_density_mean',
                               eigvec_col='team_graph_centrality_eigenvector_mean_mean',
                               path_col='team_graph_pathlength_mean'):
    """Calculate conditional effects for each graph type on a task dimension."""
    # Filter to relevant regression results
    task_effects = graph_effects[
        (graph_effects['model_type'] == 'with_interactions') & 
        (graph_effects['dataset'] == 'dataset2') &
        (graph_effects['outcome'] == 'convergence_performance')
    ]
    
    # Filter out insignificant effects
    task_effects = filter_by_significance(task_effects)
    
    # Extract the interaction column name for this task
    interaction_col = f"interaction_{task}_coef"
    
    # Extract intercept (should be same in all rows)
    intercept = task_effects['intercept_coef'].iloc[0]
    
    # Find main effect column for this task
    task_name_parts = task.split('_')
    task_identifier = task_name_parts[-1]  # 'interdep' or 'alignment'
    
    # Find task main effect column 
    task_main_cols = [col for col in task_effects.columns 
                     if task_identifier in col and 'coef' in col and 'interaction' not in col]
    
    if not task_main_cols:
        raise ValueError(f"No main effect column found for task {task}")
    
    task_main_col = task_main_cols[0]
    task_main_effect = task_effects[task_main_col].iloc[0]
    
    # Get all unique graph slugs from properties data
    all_graph_slugs = graph_props['graph_slug'].unique()
    
    # Create output DataFrame
    result_data = []
    
    # Process all graph types from properties data
    for graph_slug in all_graph_slugs:
        # Get properties for this graph
        graph_prop_rows = graph_props[graph_props['graph_slug'] == graph_slug]
        if len(graph_prop_rows) == 0:
            continue
            
        props = graph_prop_rows.iloc[0]
        
        # Calculate conditional effect based on graph type
        if graph_slug == reference_slug:
            # For reference graph, use 0 as it's the reference point
            conditional_effect = 0.0
        else:
            # For other graphs, find their specific row if it exists
            graph_effect_rows = task_effects[task_effects['main_effect_name'] == graph_slug]
            
            if len(graph_effect_rows) == 0:
                continue
                
            # Get main effect and interaction
            graph_row = graph_effect_rows.iloc[0]
            main_effect = graph_row['main_effect_coef']
            
            # Get interaction effect if it exists
            if interaction_col in graph_row:
                interaction_effect = graph_row[interaction_col]
            else:
                raise ValueError(f"Interaction column {interaction_col} not found")
                
            # Calculate conditional effect (only main effect + interaction)
            conditional_effect = main_effect + interaction_effect
        
        # Add to results
        result_data.append({
            'graph_slug': graph_slug,
            'density': props[density_col],
            'eigenvector': props[eigvec_col],
            'pathlength': props[path_col],
            'effect': conditional_effect
        })
    
    return pd.DataFrame(result_data)


def add_point_annotations(ax, points_data, annotations_list):
    """Add annotations connecting from edge of scatter points."""
    for annotation in annotations_list:
        slug = annotation['slug']
        
        # Find the point data
        point = points_data[points_data['graph_slug'] == slug]
        if len(point) == 0:
            print(f"Point with slug '{slug}' not found")
            continue
            
        # Get point information
        x = point['density'].values[0]
        y = point['y'].values[0]
        size = point['size'].values[0]
        
        # Calculate radius in points
        radius = 0.6*np.sqrt(size)
        
        # Get annotation inputs
        dx, dy = annotation.get('xy_offset', (0.1, 0))
        
        # Get ha and va
        ha = {1: 'left', -1: 'right', 0: 'center'}[np.sign(dx)]
        va = {1: 'bottom', -1: 'top', 0: 'center'}[np.sign(dy)]
        
        # Create annotation with shrinkA set to radius
        ax.annotate(
            annotation['text'],
            xy=(x, y),  # Point to annotate
            xytext=(x + dx, y + dy),  # Text position
            fontsize=6,
            color='#888888',
            ha=ha,
            va=va,
            arrowprops=dict(
                arrowstyle='-',  # Just a line, no arrow
                color='#AAAAAA',
                shrinkA=0,  
                shrinkB=radius + annotation.get('extra', 0),
                connectionstyle='arc3,rad=0'
            ),
        )


def plot_single_biaxial(graph_props, graph_effects, ax, task, metric,
                      min_size=20, max_size=200, 
                      density_col='team_graph_density_mean',
                      eigvec_col='team_graph_centrality_eigenvector_mean_mean',
                      path_col='team_graph_pathlength_mean',
                      reference_slug='complete'):
    """Plot a single biaxial plot showing task-conditional effects."""
    # Calculate conditional effects for all graph types
    merged_data = calculate_conditional_effects(
        graph_props, 
        graph_effects, 
        task,
        reference_slug=reference_slug,
        density_col=density_col,
        eigvec_col=eigvec_col,
        path_col=path_col
    )
    print(f"Task Dim: {task}, Net Met: {metric}")
    print(merged_data.sort_values('effect'))
    print('\n')
    
    # Get the y-axis value based on metric
    if metric == 'eigenvector':
        y_values = merged_data['eigenvector']
        y_label = 'Decentralization\n(eig. cent. avg.)'
    else:  # pathlength
        y_values = merged_data['pathlength']
        y_label = 'Path Length\n(avg. path len.)'
    
    # Calculate point sizes (normalize to min_size-max_size range)
    effects = merged_data['effect']
    max_abs_effect = 2.0  # Match the heatmap limits
    sizes = np.abs(effects) / max_abs_effect * (max_size - min_size) + min_size
    
    # Map effects to color using GrayGn colormap
    norm_effects = (effects / max_abs_effect + 1) / 2  # Map from [-1, 1] to [0, 1]
    point_colors = [GrayGn(val) for val in norm_effects]
    
    # Create scatter plot
    ax.scatter(
        merged_data['density'], 
        y_values, 
        s=sizes, 
        c=point_colors,
        alpha=0.8,
        edgecolors='#666666',
        linewidths=0.5,
        clip_on=False,
        zorder=5
    )
    
    # Add y column and size to the data for use in annotations
    merged_data['y'] = y_values
    merged_data['size'] = sizes
    
    # Set border
    set_border(ax, top=False, right=False, bottom=False, left=False)
    
    # Set limits and scales
    ax.set_xlim(0, 1)

    # Add grid and remove ticks
    ax.grid(True, color='#CCCCCC')
    ax.tick_params(size=0)
    
    # Add task and metric-specific labels
    if metric == 'eigenvector':
        # Set y limits
        ax.set_ylim(0.3, 0.36)
        
        # Set top labels
        if task == 'norm_team_fn_diff_interdep':
            ax.text(0.5, 1.05, 'Coordinate', transform=ax.transAxes, 
                  ha='center', va='bottom', fontsize=8)
        elif task == 'norm_team_fn_diff_alignment':
            ax.text(0.5, 1.05, 'Negotiate', transform=ax.transAxes, 
                  ha='center', va='bottom', fontsize=8)
            
        # Set x ticks
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], [])
    
    # Add specialists/generalists annotations only on the bottom row
    if metric == 'pathlength':
        # Set x labels
        ax.set_xlabel('Network Density')
        
        # Set x ticks
        xticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(xticks, xticks)
        
        # Set y limits
        ax.set_ylim(1, 3)
        
    # Add y-axis label only on left side
    if task == 'norm_team_fn_diff_interdep':
        ax.set_ylabel(y_label)
    
    return merged_data


def add_biaxial_legend(fig, legend_ax, effect_data):
    """Add unified legend for all biaxial plots."""
    # Set axis properties
    legend_ax.set_axis_off()
    
    # Maximum effect magnitude across all plots
    max_effect = 2.0  # Match the heatmap limits
    
    # Create effect values for legend
    effect_values = np.linspace(-max_effect/2, max_effect/2, 9)
    
    # Map effects to color using GrayGn colormap
    norm_effects = (effect_values / max_effect + 1) / 2  # Map from [-1, 1] to [0, 1]
    point_colors = [GrayGn(val) for val in norm_effects]
    
    # Calculate sizes based on absolute effect magnitude
    sizes = np.abs(effect_values) / max_effect * 180 + 20  # Scale to match plot sizes
    
    # Position the legend points at y=0.5
    x_pos = np.linspace(0.2, 0.8, len(effect_values))
    
    # Adjust limits
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    
    # Draw sample points
    for i, (size, effect, color) in enumerate(zip(sizes, effect_values, point_colors)):
        legend_ax.scatter(x_pos[i], 0.5, s=size, c=[color], edgecolors='#666666', linewidths=0.5)
        legend_ax.text(x_pos[i], 0, f'{effect_values[i]:.2f}', size=6, ha='center', va='top')
        
    # Draw arrows and annotation text
    arrow(legend_ax, (0.15, 0.5), (0, 0.5))
    legend_ax.text(0.075, 0.75, 'Performs\nWorse', ha='center', va='bottom', size=6, color='#666666')
    arrow(legend_ax, (0.85, 0.5), (1, 0.5))
    legend_ax.text(0.925, 0.75, 'Performs\nBetter', ha='center', va='bottom', size=6, color='#666666')
    
    # Add Title
    legend_ax.set_title('Effect Size (St. Devs. of Performance)', va='bottom')
    legend_ax.text(0.5, 1, '(task conditional effects vs. complete graph reference)', size=6, ha='center')
    
    return legend_ax


def plot_biaxial_grid(study_name, fig, gs_cell, team_size=8):
    """Create 2x2 grid of biaxial plots showing task effects."""
    # Set up fonts
    set_fonts()
    
    # Load data
    graph_props = load_graph_properties(study_name)
    graph_effects = load_graph_effects(study_name)
    
    # Filter to team size
    graph_props = graph_props[graph_props['team_size'] == team_size]
    
    # Create nested GridSpec for biaxial plots
    nested_gs = GridSpecFromSubplotSpec(4, 2, subplot_spec=gs_cell, 
                                       height_ratios=[1, 1, 0.15, 0.2], 
                                       wspace=0.25, hspace=0.25)
    
    # Create axes for the biaxial plots
    ax_top_left = fig.add_subplot(nested_gs[0, 0])
    ax_top_right = fig.add_subplot(nested_gs[0, 1], sharey=ax_top_left)
    ax_bottom_left = fig.add_subplot(nested_gs[1, 0], sharex=ax_top_left)
    ax_bottom_right = fig.add_subplot(nested_gs[1, 1], sharex=ax_top_right, sharey=ax_bottom_left)
    
    # Hide shared y-axis ticks for right column
    plt.setp(ax_top_right.get_yticklabels(), visible=False)
    plt.setp(ax_bottom_right.get_yticklabels(), visible=False)
    
    # Set task dimensions to plot
    tasks = {
        'coordinate': 'norm_team_fn_diff_interdep',
        'negotiate': 'norm_team_fn_diff_alignment'
    }
    
    # Plot each biaxial plot
    data_tl = plot_single_biaxial(graph_props, graph_effects, ax_top_left, tasks['coordinate'], 'eigenvector')
    data_tr = plot_single_biaxial(graph_props, graph_effects, ax_top_right, tasks['negotiate'], 'eigenvector')
    data_bl = plot_single_biaxial(graph_props, graph_effects, ax_bottom_left, tasks['coordinate'], 'pathlength')
    data_br = plot_single_biaxial(graph_props, graph_effects, ax_bottom_right, tasks['negotiate'], 'pathlength')
    
    # Collect effect data
    effect_data = {
        'eigenvector_coordinate': data_tl,
        'eigenvector_negotiate': data_tr,
        'pathlength_coordinate': data_bl,
        'pathlength_negotiate': data_br
    }
    
    # Define annotations for each plot
    tl_annotations = [
        {'slug': 'random_p07', 'text': 'Random\n(high probab.)', 'xy_offset': (0.001, -0.008), 'extra': -1},
        {'slug': 'hypercube', 'text': 'Hypercube', 'xy_offset': (-0.1, 0.00), 'extra': -1}
    ]
    
    tr_annotations = [
        {'slug': 'random_p03', 'text': 'Random\n(low probab.)', 'xy_offset': (0.1, -0.001), 'extra': -1},
        {'slug': 'complete', 'text': 'Complete', 'xy_offset': (-0.05, 0.002)}
    ]
    
    bl_annotations = [
        {'slug': 'star', 'text': 'Star', 'xy_offset': (-0.1, -0.001), 'extra': -1}
    ]
    
    br_annotations = [
        {'slug': 'tree', 'text': 'Tree', 'xy_offset': (0.1, 0.1)},
        {'slug': 'random_p07', 'text': 'Random\n(high probab.)', 'xy_offset': (-0.1, -0.001), 'extra': -1},
    ]
    
    # Add annotations to each plot
    add_point_annotations(ax_top_left, data_tl, tl_annotations)
    add_point_annotations(ax_top_right, data_tr, tr_annotations)
    add_point_annotations(ax_bottom_left, data_bl, bl_annotations) 
    add_point_annotations(ax_bottom_right, data_br, br_annotations)
    
    # Create legend axis and add legend
    legend_ax = fig.add_subplot(nested_gs[3, :])
    add_biaxial_legend(fig, legend_ax, effect_data)
    
    return effect_data