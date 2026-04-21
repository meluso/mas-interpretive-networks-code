#!/usr/bin/env python3
# figures/plot_performance_means.py
"""Plot convergence performance vs convergence time with density coloring."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from figures.util import (
    set_fonts, 
    fig_size, 
    save_pub_fig, 
    set_border,
    get_gray_green_cmap,
    arrow,
    get_optimizer
)

set_fonts()


def load_convergence_data(study_name, team_size=8):
    """Load convergence statistics from mean differences file."""
    data_path = Path(f"data/results/{study_name}/mean_differences.csv")
    df = pd.read_csv(data_path)
    return df[df['team_size'] == team_size].copy()


def plot_convergence_performance(study_name, team_size=8):
    """Plot convergence time vs performance colored by density.
    
    Args:
        study_name: Name of study to analyze
        team_size: Team size to filter
    """
    # Load data
    df = load_convergence_data(study_name, team_size)
    
    # Define rationality bounds to plot
    bounds = {
        0.1: 'Loosely Bounded',
        0.01: 'Moderately Bounded',
        0.001: 'Tightly Bounded'
        }
    
    # Create figure with space for colorbar
    fig = plt.figure(figsize=fig_size(frac_width=1, frac_height=0.25))
    
    # Create grid spec with extra space for colorbar
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.15], wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Get colormap
    cmap = get_gray_green_cmap()
    
    # Store scatter for colorbar
    scatter_obj = None
    
    for idx, (bound, title) in enumerate(bounds.items()):
        ax = axes[idx]
        
        # Filter to rationality bound
        bound_data = df[np.isclose(df['agent_steplim'], bound)]
        
        # Extract values
        x = bound_data['convergence_step_mean']
        y = bound_data['convergence_performance_mean']
        colors = bound_data['density']
        
        # Set white background for better contrast
        ax.set_facecolor('#f9f9f9')
        
        # Add grid
        ax.grid(True, color='lightgray', linestyle='-', alpha=0.4, zorder=0)
        
        # Plot scatter with styling from biaxials
        scatter = ax.scatter(
            x, y,
            c=colors,
            cmap=cmap,
            s=25,
            alpha=0.8,
            edgecolors='#666666',
            linewidths=0.5,
            vmin=0,
            vmax=1,
            zorder=3
        )
        
        if idx == 0:
            scatter_obj = scatter
        
        # Define which graphs to annotate and their positions for each bound
        if bound == 0.1:
            # Panel A: empty RIGHT, complete BELOW, star LEFT, random_p03 RIGHT
            annotations_config = {
                'empty': {'xytext': (8, 0), 'ha': 'left', 'va': 'center'},
                'complete': {'xytext': (0, -8), 'ha': 'center', 'va': 'top'},
                'star': {'xytext': (-8, 0), 'ha': 'right', 'va': 'center'},
                'random_p03': {'xytext': (0, 8), 'ha': 'center', 'va': 'bottom'}
            }
        elif bound == 0.01:
            # Panel B: empty LEFT, complete BELOW, star ABOVE, random_p03 RIGHT
            annotations_config = {
                'empty': {'xytext': (-8, 0), 'ha': 'right', 'va': 'center'},
                'complete': {'xytext': (0, -8), 'ha': 'center', 'va': 'top'},
                'star': {'xytext': (0, 8), 'ha': 'center', 'va': 'bottom'},
                'random_p03': {'xytext': (8, 0), 'ha': 'left', 'va': 'center'}
            }
        else:  # 0.001
            # Panel C: empty LEFT, complete RIGHT, star RIGHT
            annotations_config = {
                'empty': {'xytext': (-8, 0), 'ha': 'right', 'va': 'center'},
                'complete': {'xytext': (8, 0), 'ha': 'left', 'va': 'center'},
                'star': {'xytext': (-8, 0), 'ha': 'right', 'va': 'center'},
                'wheel': {'xytext': (4, 8), 'ha': 'center', 'va': 'bottom'}
            }
        
        # Create label mapping for display names
        label_map = {
            'empty': 'empty',
            'complete': 'complete',
            'star': 'star',
            'random_p03': 'random\n(low probab.)'
        }
        
        # Annotate all configured points
        for graph_slug, config in annotations_config.items():
            row_data = bound_data[bound_data['graph_slug'] == graph_slug]
            
            if len(row_data) > 0:
                x_val = row_data['convergence_step_mean'].values[0]
                y_val = row_data['convergence_performance_mean'].values[0]
                
                # Calculate radius from scatter point size
                point_size = 50
                radius = 0.6 * np.sqrt(point_size)
                
                # Get display label
                display_label = label_map.get(graph_slug, graph_slug)
                
                ax.annotate(
                    display_label,
                    xy=(x_val, y_val),
                    xytext=config['xytext'],
                    textcoords='offset points',
                    fontsize=6,
                    color='#888888',
                    ha=config['ha'],
                    va=config['va'],
                    arrowprops=dict(
                        arrowstyle='-',
                        color='#AAAAAA',
                        shrinkA=0,
                        shrinkB=radius - 1,
                        connectionstyle='arc3,rad=0'
                    )
                )
        
        # Set title and labels
        ax.set_title(f'{title} ($\pm{100*bound}\%$)')
        ax.set_xlabel('Convergence Time (steps)')
        
        # Add panel label
        panel_labels = ['A', 'B', 'C']
        ax.text(
            -0.15, 1.05,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=10,
            va='bottom',
            ha='right'
        )
        
        if idx == 0:
            ax.set_ylabel('Convergence Performance')
            
            # Add "Performs Better" arrow
            arrow_y = 0.85
            arrow(ax, (0.9, 0.65), (0.9, arrow_y))
            ax.text(
                x=0.87, y=0.75, 
                s='Performs\nBetter',
                size=8, 
                clip_on=False, 
                va='center', 
                ha='right', 
                color='#888888',
                transform=ax.transAxes
            )
        
        # Add padding to axis limits for better visibility
        x_range = x.max() - x.min()
        ax.set_xlim(x.min() - 0.12*x_range, x.max() + 0.12*x_range)
        
        y_range = y.max() - y.min()
        ax.set_ylim(y.min() - 0.12*y_range, y.max() + 0.12*y_range)
        
        # Increase x-axis tick frequency
        ax.locator_params(axis='x', nbins=5)
        
        # Set borders
        set_border(ax, top=False, right=False)
        ax.tick_params(size=0)
    
    # Add colorbar in dedicated space
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(scatter_obj, cax=cbar_ax)
    cbar.set_label('Network Density', fontsize=8, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=7, length=0)
    
    # Remove colorbar spines
    cbar.outline.set_visible(False)
    
    # Save figure
    optimizer = get_optimizer(study_name)
    figure_name = f"performance_means_{optimizer}_team{team_size}"
    save_pub_fig(figure_name, bbox_inches='tight')
    
    plt.show()
    
    return fig


def print_summary_stats(study_name, team_size=8):
    """Print summary statistics for convergence patterns."""
    df = load_convergence_data(study_name, team_size)
    bounds = [0.1, 0.01, 0.001]
    
    print("="*80)
    print("CONVERGENCE PERFORMANCE SUMMARY")
    print("="*80)
    
    for bound in bounds:
        bound_data = df[np.isclose(df['agent_steplim'], bound)]
        
        # Calculate correlations
        corr_time_perf = bound_data['convergence_step_mean'].corr(
            bound_data['convergence_performance_mean']
        )
        corr_density_perf = bound_data['density'].corr(
            bound_data['convergence_performance_mean']
        )
        corr_density_time = bound_data['density'].corr(
            bound_data['convergence_step_mean']
        )
        
        print(f"\nRationality Bound = {bound}")
        print(f"  r(time, perf): {corr_time_perf:+.3f}")
        print(f"  r(density, perf): {corr_density_perf:+.3f}")
        print(f"  r(density, time): {corr_density_time:+.3f}")
        
        # Extract extremes
        empty_row = bound_data[bound_data['density'] == 0]
        complete_row = bound_data[bound_data['density'] == 1]
        
        if len(empty_row) > 0 and len(complete_row) > 0:
            empty_time = empty_row['convergence_step_mean'].values[0]
            empty_perf = empty_row['convergence_performance_mean'].values[0]
            complete_time = complete_row['convergence_step_mean'].values[0]
            complete_perf = complete_row['convergence_performance_mean'].values[0]
            
            time_diff_pct = ((empty_time / complete_time) - 1) * 100
            perf_diff_pct = ((empty_perf / complete_perf) - 1) * 100
            
            print(f"  Empty: time={empty_time:.1f}, perf={empty_perf:.3f}")
            print(f"  Complete: time={complete_time:.1f}, perf={complete_perf:.3f}")
            print(f"  Time Δ: {time_diff_pct:+.1f}%")
            print(f"  Perf Δ: {perf_diff_pct:+.1f}%")
        
        # Print all graph types
        print(f"\n  All topologies (sorted by density):")
        for _, row in bound_data.sort_values('density').iterrows():
            print(f"    {row['graph_slug']:<20s} density={row['density']:.3f}  "
                  f"perf={row['convergence_performance_mean']:.3f}  "
                  f"time={row['convergence_step_mean']:.1f}")


if __name__ == '__main__':
    study_name = 'aiteams01nm_20250128_223001'
    # study_name = 'aiteams01rw_20250321_215818'
    print_summary_stats(study_name)
    plot_convergence_performance(study_name)