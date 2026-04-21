# figures/plot_network_metric_task_performance_heatmap.py
"""
Plot network metrics and their interactions with task difficulties as a heatmap.

Creates a heatmap visualization showing:
- First column: Network metric main effects from main_effects_only model (scaled for visibility)
- Other columns: Conditional effects (M + D + M*D) from with_interactions model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path

# Import figure settings utilities
from config.metrics import NETWORK_METRICS
from figures.util import filter_by_significance, set_fonts, fig_size, save_pub_fig, set_border, GrayGn

def load_ols_regression_data(study_name, dataset_name, outcome_metric):
    """Load OLS regression results for network metrics."""
    results_dir = Path(f"data/results/{study_name}/ols_regressions")
    file_path = results_dir / "network_metric_regressions.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"OLS regression results not found at {file_path}")
        
    # Read data
    df = pd.read_csv(file_path)
    
    # Filter to keep only significant effects
    df = filter_by_significance(df)
    
    return df

def get_task_order_mapping():
    """Get mapping from task difficulty metrics to their dimension labels."""
    return {
        'norm_team_fn_diff_integral': 'Generate',
        'log_team_fn_diff_peaks': 'Choose',
        'norm_team_fn_diff_interdep': 'Coordinate',
        'norm_team_fn_diff_alignment': 'Negotiate'
    }

def calculate_conditional_effects(interx_df, network_metric, task_diff):
    """Calculate conditional effect of network metric and task difficulty."""
    # Find the row for this network metric
    row = interx_df[(interx_df['main_effect_name'] == network_metric)]
    
    if len(row) == 0:
        return np.nan
    
    row = row.iloc[0]
    
    # Extract main effect of network metric
    network_effect = row['main_effect_coef']
    
    # Extract interaction effect
    interaction_effect = row[f"interaction_{task_diff}_coef"]
    
    # Sum for conditional effect
    conditional_effect = network_effect + interaction_effect
    
    return conditional_effect

def format_network_metric_name(name):
    """Format network metric name for display by removing prefixes."""
    clean_name = name.replace('norm_', '').replace('log_', '')
    if clean_name in NETWORK_METRICS:
        return NETWORK_METRICS[clean_name]['label']
    return clean_name

def create_heatmap_data(df, dataset_name, outcome_metric):
    """Create data matrix for heatmap visualization."""
    # Filter to specified dataset and outcome
    filtered_df = df[(df['dataset'] == dataset_name) & (df['outcome'] == outcome_metric)].sort_values(by='main_effect_coef')
    
    # Split into main effects only and interaction models
    main_df = filtered_df[filtered_df['model_type'] == 'main_effects_only']
    interx_df = filtered_df[filtered_df['model_type'] == 'with_interactions']
    
    # Get unique network metrics
    network_metrics = main_df['main_effect_name'].unique()
    
    # Get task difficulties
    task_order = get_task_order_mapping()
    task_difficulties = list(task_order.keys())
    
    # Initialize data matrix
    n_metrics = len(network_metrics)
    n_columns = 1 + len(task_difficulties)  # Main effect + task interactions
    data = np.zeros((n_metrics, n_columns))
    
    # Fill data matrix
    for i, metric in enumerate(network_metrics):
        # Main effect (from main_effects_only model)
        main_row = main_df[main_df['main_effect_name'] == metric]
        data[i, 0] = main_row['main_effect_coef'].iloc[0] if len(main_row) > 0 else np.nan
        
        # Conditional effects with each task (from with_interactions model)
        for j, task in enumerate(task_difficulties):
            data[i, j+1] = calculate_conditional_effects(interx_df, metric, task)
    
    return data, network_metrics, task_difficulties, task_order


def add_task_columns_header(ax, n_rows):
    """Add a simple header with line for task columns."""
    # Get the current tick positions
    tick_pos = ax.get_xticks()
    
    # Calculate center and width of task columns area
    left_edge = 2  # First task column
    right_edge = 6  # One past last task column
    x_min = tick_pos[left_edge] - 0.55
    x_max = tick_pos[right_edge-1] + 0.2
    width = x_max - x_min
    x_center = x_min + width/2
    y_bar = n_rows + 2.3
    y_text = y_bar + 0.2
    
    # Add text
    ax.text(
        x_center,
        y_text,
        "Task Conditional Effects",
        ha='center',
        va='bottom',
        fontsize=7,
        color='#666666'
    )
    
    # Add horizontal line
    ax.plot(
        [x_min, x_max],  # x points
        [y_bar, y_bar],  # y points
        color='#DDDDDD',
        linewidth=5,
        clip_on=False
    )

def create_heatmap(data, network_metrics, task_difficulties, task_order, ax, cbar_ax):
    """Create heatmap visualization of network metric effects with dual scales."""
    
    # === SETUP ===
    # Format labels
    metric_labels = [format_network_metric_name(m) for m in network_metrics]
    visible_labels = ['Main Effect']
    for task in task_difficulties:
        visible_labels.append(f"{task_order[task]}")
    
    # === DATA PREPARATION ===
    # Insert a narrow gap column and scale main effects
    scaled_data = np.insert(data, 1, np.nan, axis=1).copy()
    data = np.insert(data, 1, np.nan, axis=1).copy()
    scaling_factor = 1
    scaled_data[:, 0] *= scaling_factor  # Scale only the first column (main effects)
    
    # Determine color scale limits
    actual_max = 2
    vmax_cond = np.ceil(actual_max * 2) / 2  # Round up to nearest 0.5
    vmin_cond = -vmax_cond
    
    # === CREATE CUSTOM GRID ===
    n_rows, n_cols = scaled_data.shape
    
    # X coordinates with a narrow gap after first column
    gap_width = 0.3  # Width of the gap column
    x_edges = np.zeros(n_cols + 1)
    x_edges[0] = 0
    x_edges[1] = 1  # First column (normal width)
    x_edges[2] = 1 + gap_width  # After narrow gap
    for i in range(3, n_cols + 1):
        x_edges[i] = x_edges[2] + (i - 2)
    
    # Y coordinates (normal spacing)
    y_edges = np.arange(n_rows + 1)
    
    # === CREATE HEATMAP ===
    heatmap = ax.pcolormesh(
        x_edges, y_edges, scaled_data,
        cmap=GrayGn,
        edgecolors='white',
        linewidth=0.5,
        vmin=vmin_cond,
        vmax=vmax_cond
    )
    ax.set_aspect('equal')
    
    # Add value annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]  # Use original (unscaled) data for annotations
            if not np.isnan(value):
                x_center = (x_edges[j] + x_edges[j+1]) / 2
                y_center = i + 0.5
                text_color = 'white' if abs(scaled_data[i, j]) > vmax_cond * 0.5 else 'black'
                ax.text(x_center, y_center, f"{value:.2f}", 
                        ha='center', va='center', color=text_color, size=6)
    
    # === CONFIGURE AXES ===
    # Set up tick positions (centered in each cell)
    tick_positions = [0.5]  # Main column center
    for j in range(2, n_cols):
        tick_positions.append((x_edges[j] + x_edges[j+1]) / 2)
            
    # Add subtle background
    add_task_columns_header(ax, n_rows)
    
    # Configure axis
    ax.tick_params(bottom=False, top=False, left=False, labelbottom=False, labeltop=True)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(visible_labels, ha='left', rotation=90, va='center', rotation_mode='anchor')
    ax.set_yticks(np.arange(len(metric_labels)) + 0.5, labels=metric_labels)
    
    # Set limits and labels
    ax.set_xlim(0, x_edges[-1])
    ax.set_ylim(0, n_rows)
    ax.set_ylabel('Network Property (with metric)')
    set_border(ax)
    
    # === CREATE COLORBAR ===
    # Create evenly spaced tick values
    cond_tick_values = np.linspace(vmin_cond, vmax_cond, 5)
    cond_tick_labels = [f"{tick:.1f}" for tick in cond_tick_values]
    main_tick_labels = [f"{tick/scaling_factor:.1f}" for tick in cond_tick_values]
    
    # Create colorbar directly in the provided axis
    cbar = plt.colorbar(heatmap, cax=cbar_ax, orientation='horizontal')
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)
    cbar.set_ticks(cond_tick_values)
    cbar.set_ticklabels(cond_tick_labels)
    
    # Manually adjust the colorbar position for better height/width ratio
    pos = cbar_ax.get_position()
    # Format: [x0, y0, width, height]
    new_height = pos.height * 0.3  # Make height 30% of original
    x_pos = pos.x0 - 0.13
    y_offset = -0.0    # Move down to create space
    new_pos = [x_pos, pos.y0 + y_offset, pos.width, new_height]
    cbar_ax.set_position(new_pos)
    
    # Title - position above the colorbar with reduced padding
    cbar.ax.set_title('Effect Size (St. Devs. of Performance)', 
                     fontsize=8, pad=10)

def plot_network_metric_task_heatmap(study_name, dataset_name, outcome_metric, fig, gs_cell):
    """Create heatmap visualization of network metric effects.
    
    Args:
        study_name: Name of the study
        dataset_name: Dataset name (dataset1 or dataset2)
        outcome_metric: Outcome metric to analyze
        fig: Figure to draw on (required)
        gs_cell: GridSpec cell to use (required)
    """
    # Set up fonts
    set_fonts()
    
    # Create nested GridSpec for internal layout (2×2 grid)
    # This gives us control of colorbar placement
    nested_gs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_cell, 
        height_ratios=[1, 6], width_ratios=[1, 1]
        )
    
    # Create colorbar axis in top-left cell
    cbar_ax = fig.add_subplot(nested_gs[0, 0])
    
    # Create heatmap axis spanning other cells
    heatmap_ax = fig.add_subplot(nested_gs[1:, 0:])
    
    # Load regression data
    df = load_ols_regression_data(study_name, dataset_name, outcome_metric)
    
    # Create heatmap data
    data, network_metrics, task_difficulties, task_order = create_heatmap_data(
        df, dataset_name, outcome_metric
    )
    
    # Calculate and print average effect magnitudes for console output
    col_avg_magnitudes = np.nanmean(np.abs(data), axis=0)
    task_cond_avg_magnitude = np.nanmean(col_avg_magnitudes[1:])
    magnitudes_text = \
        f"Average effect magnitudes: Main={col_avg_magnitudes[0]:.3f}, " +\
        f"Tasks={[f'{x:.3f}' for x in col_avg_magnitudes[1:]]}, " +\
        f"Overall_Task_Avg={task_cond_avg_magnitude:.3f}"
    print(magnitudes_text)
    
    # Create figure
    create_heatmap(data, network_metrics, task_difficulties, task_order,
                   heatmap_ax, cbar_ax)
    