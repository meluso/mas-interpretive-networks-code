#!/usr/bin/env python3
# figures/plot_mean_differences.py
"""
Plot mean performance differences vs network density for teams of size 8
across different agent step limits in three separate subplots.
"""

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
import pprint

# Import figure settings utilities
from util import set_fonts, fig_size, save_pub_fig, set_border, GREENS_3, arrow, GREENS_4

# Set up fonts
set_fonts()

def load_mean_differences(study_name: str) -> pd.DataFrame:
    """Load mean differences data from CSV file."""
    file_path = Path(f"data/results/{study_name}/mean_differences.csv")
    if not file_path.exists():
        raise FileNotFoundError(f"Mean differences file not found at {file_path}")
    return pd.read_csv(file_path)

def filter_data_for_plot(df: pd.DataFrame, team_size: int = 8) -> pd.DataFrame:
    """Filter data for specific team size and organize by agent step limit."""
    # Filter for team size 8
    team_data = df[df['team_size'] == team_size].copy()
    
    # Make sure we have required columns
    required_columns = ['agent_steplim', 'graph_slug', 'density', 
                        'convergence_performance_mean',  # Need group mean
                        'convergence_performance_diff', 
                        'convergence_performance_lower_ci', 
                        'convergence_performance_upper_ci']
    
    # Calculate baseline mean for each agent_steplim
    # Baseline = group_mean - difference
    team_data['baseline_mean'] = team_data['convergence_performance_mean'] - team_data['convergence_performance_diff']
    
    return team_data

def print_plot_data(step_limit, step_data, y, lower_ci, upper_ci):
    """Print DataFrame with plotted values and confidence intervals indexed by graph_slug"""
    # Create DataFrame with plotted values and confidence intervals
    plot_df = pd.DataFrame({
        'Density': step_data['density'].values,
        'Plotted_Value': y.values,
        'Lower_CI': lower_ci.values,
        'Upper_CI': upper_ci.values
    }, index=step_data['graph_slug'].values)
    
    print(f"\nPlot data for agent_steplim={step_limit}:")
    pd.set_option('display.precision', 3)  # Set display precision for better readability
    print(plot_df.sort_values('Plotted_Value', ascending=False))  # Sort by plotted value

def plot_mean_differences(study_name: str = 'aiteams01nm_20250128_223001', team_size: int = 8, use_percent: bool = True, axes_only: bool = False):
    """
    Plot convergence performance mean differences vs network density 
    for teams of specified size across different agent step limits,
    with each step limit in a separate subplot.
    
    Args:
        study_name: Name of the study
        team_size: Team size to plot (default: 8)
    """
    # Load data
    df = load_mean_differences(study_name)
    
    # Filter data for plotting
    plot_data = filter_data_for_plot(df, team_size)
    
    # Define step limits to plot
    step_limits = [0.1, 0.01, 0.001]
    
    # Define green colors for different step limits
    # Using colors from our lollipop plots
    colors = GREENS_3 # Dark, medium, and light green
    
    # Create figure with three subplots side by side
    fig, axes = plt.subplots(1, len(step_limits), figsize=fig_size(frac_width=1.0, frac_height=0.3), 
                             sharey=True, sharex=False)
    
    # Track min and max y values for consistent y-limits
    y_min, y_max = float('inf'), float('-inf')
    
    # Plot each step limit in its own subplot
    for i, step_limit in enumerate(step_limits):
        ax = axes[i]
        
        # Filter data for this step limit
        step_data = plot_data[np.isclose(plot_data['agent_steplim'], step_limit)]
        
        if len(step_data) == 0:
            print(f"No data found for agent_steplim={step_limit}")
            continue
            
        # Sort by density for better line plotting
        step_data = step_data.sort_values('density')
        
        # Get x and y values
        x = step_data['density']
        
        if use_percent:
            # Properly calculate percentage differences
            baseline_means = step_data['baseline_mean']
            print(f"")
            
            # Calculate percentage difference and CIs correctly
            y = step_data['convergence_performance_diff'] / baseline_means * 100
            
            # Convert absolute CIs to percentage CIs
            lower_ci = step_data['convergence_performance_lower_ci'] / baseline_means * 100
            upper_ci = step_data['convergence_performance_upper_ci'] / baseline_means * 100
        else:
            # Use absolute difference
            y = step_data['convergence_performance_diff']
            lower_ci = step_data['convergence_performance_lower_ci']
            upper_ci = step_data['convergence_performance_upper_ci']
        
        # Update min and max y values
        y_min = min(y_min, np.nanmin(lower_ci))
        y_max = max(y_max, np.nanmax(upper_ci))
            
        # Calculate error bar heights
        yerr = np.vstack([y - lower_ci, upper_ci - y])
        
        # Set background color
        ax.set_facecolor('#f9f9f9')
        
        # Draw horizontal zero line
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.6)
        
        # Add grid
        ax.grid(True, color='lightgray', linestyle='-', alpha=0.5, zorder=0)
        
        # Plot points and error bars in a single call using errorbar
        if not axes_only:
            ax.errorbar(x, y, yerr=yerr, fmt='o', color=colors[i],
                       ecolor='#888888', capsize=4, capthick=1.5, 
                       ms=5, mec='#888888', mew=0.5, alpha=0.75, zorder=3)
        
        # Print the plotted data
        print_plot_data(step_limit, step_data, y, lower_ci, upper_ci)
        
        # Add annotations for key graph types
        for graph_type in ['empty', 'complete']:
            graph_data = step_data[step_data['graph_slug'] == graph_type]
            is_empty = graph_type == 'empty'
            
            if len(graph_data) > 0:
                x_val = graph_data['density'].values[0]
                
                if use_percent:
                    # Use properly calculated percentage for annotation
                    baseline = graph_data['baseline_mean'].values[0]
                    diff = graph_data['convergence_performance_diff'].values[0]
                    y_val = diff / baseline * 100
                else:
                    y_val = graph_data['convergence_performance_diff'].values[0]
                
                # Add annotation with offset
                y_offset = 0
                x_offset = -0.03
                horiz_align = 'left' if is_empty else 'right'
                
                # Adjust position 
                if graph_type == 'empty':
                    x_offset *= -1
                
                if not axes_only:
                    ax.annotate(
                        f'{graph_type}', 
                        (x_val, y_val),
                        xytext=(x_val + x_offset, y_val + y_offset),
                        fontsize=8,
                        alpha=0.8,
                        ha=horiz_align,
                        va='center'
                        )
        
        # Set title for each subplot with step limit
        ax.set_title(f'Rationality Bound $=\pm{step_limit}$')
        
        # First subplot
        if i == 0:
        
            # Set axis labels only on relevant subplots
            if use_percent:
                ax.set_ylabel('% Difference in Performance\nVs. All Graphs Average')
            else:
                ax.set_ylabel('Difference in Performance\nFrom All Graphs Average')
            
            # Add annotation arrows
            ht = 0.05
            
            # More specialists
            arrow(ax, (0.35, ht), (0.1, ht), '')
            ax.text(x=0.225, y=0.08, s='More\nSpecialists',
                size=8, clip_on=False, va='bottom', ha='center', color='#888888',
                transform=ax.transAxes)
            
            # More generalists
            arrow(ax, (0.65, ht), (0.9, ht), '')
            ax.text(x=0.775, y=0.08, s='More\nGeneralists',
                size=8, clip_on=False, va='bottom', ha='center', color='#888888',
                transform=ax.transAxes)
            
            # Performs better
            arrow(ax, (0.9, 0.65), (0.9, 0.85))
            ax.text(x=0.87, y=0.75, s='Performs\nBetter',
                size=8, clip_on=False, va='center', ha='right', color='#888888',
                transform=ax.transAxes)
            
            # Average line
            ax.text(x=0.06, y=0.48, s='All Graph\nAverage',
                size=6, clip_on=False, va='center', ha='left', color='#888888',
                transform=ax.transAxes)
        
        # Set xlabel for all subplots
        ax.set_xlabel('Network Density')
        
        # Set x limits
        ax.set_xlim(-0.05, 1.05)
        
        # Set border
        set_border(ax, top=False, right=False)
        
        # Scale
        # ax.set_xscale('symlog', linthresh=0.1)
        ax.tick_params(size=0)
    
    # Set consistent y-limits for all subplots with padding
    y_padding = (y_max - y_min) * 0.1
    for ax in axes:
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Adjust spacing
    plt.tight_layout()
    # fig.subplots_adjust(top=0.85)
    
    # Save figure
    figure_name = f"network_density_performance_team{team_size}_subplots"
    save_pub_fig(figure_name)
    
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Plot using percentage differences
    plot_mean_differences(team_size=8, axes_only=False)