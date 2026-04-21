"""Generate simplified figures for methodology overview panel."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from figures.util import save_pub_fig


def create_convergence_plot():
    """Generate stylized performance convergence trajectory."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    # Generate smooth convergence curve (ends at trial 80, x-axis to 100)
    iterations = np.linspace(0, 80, 800)  # Only generate up to trial 80
    # Exponential convergence with consistent noise, converges to ~0.85 by trial 80
    np.random.seed(42)
    performance = 0.70 * (1 - np.exp(-iterations / 25)) + 0.15
    # Add consistent smooth noise
    noise = 0.01 * np.sin(iterations * 0.3) * np.exp(-iterations / 100)
    performance = performance + noise
    
    # Plot convergence trajectory (single green line) - stops at trial 80
    ax.plot(iterations, performance, color='#2d6e3e', linewidth=2, label='Team performance')
    
    # Mark convergence performance (at trial 80 - the end of the line)
    final_perf = performance[-1]
    ax.plot(80, final_perf, 'o', color='#2d6e3e', markersize=8, zorder=10)
    
    # Dashed horizontal line showing convergence performance
    ax.axhline(final_perf, color='#2d6e3e', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Label on the right side next to the dashed line
    ax.text(102, final_perf, 'Convergence\nPerformance', 
            ha='left', va='center', fontsize=12, color='#2d6e3e', fontweight='bold')
    
    # Styling
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_ylabel('Performance', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')
    fig.patch.set_alpha(0)
    
    plt.tight_layout()
    save_pub_fig('methods_convergence')


def create_heatmap_preview():
    """Generate simplified heatmap showing gradient pattern."""
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # Create 3 rows x 4 columns mini heatmap
    # Values ranging from negative (gray) to positive (green)
    data = np.array([
        [-0.5, 0.8, 0.9, -0.6],
        [0.1, 0.3, 0.2, 0.05],
        [-0.2, -0.3, -0.4, 0.2]
    ])
    
    # Create custom colormap (gray to green, matching their style)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#808080', '#e8e8e8', '#ffffff', '#c8e6c9', '#66bb6a', '#2d6e3e']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('gray_green', colors, N=n_bins)
    
    # Plot heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Add values in cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Labels
    row_labels = ['Network\nDensity', 'Triangle\nDensity', 'Path\nLength']
    col_labels = ['Generate', 'Choose', 'Coordinate', 'Negotiate']
    
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=90, ha='center')
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add colorbar without border
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Effect Size\n(St. Devs.)', fontsize=8, rotation=0, 
                   ha='left', va='center', labelpad=15)
    cbar.ax.tick_params(labelsize=7, length=0)
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    save_pub_fig('methods_heatmap', transparent=True)


def create_means_plot_preview():
    """Generate simplified convergence scatter plot matching new figure style."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    # Generate representative data simulating convergence time vs performance
    np.random.seed(42)
    
    # Simulate ~15 points with varying density
    n_points = 15
    densities = np.linspace(0, 1, n_points)
    
    # Create pattern: specialists (low density) perform better but slower
    # This mimics the general pattern we see in the data
    conv_times = 15 + 25 * densities**2 + np.random.normal(0, 2, n_points)
    performance = 0.92 - 0.20 * densities + 0.10 * densities**2 + np.random.normal(0, 0.015, n_points)
    
    # Create custom colormap (gray to green)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#808080', '#e8e8e8', '#ffffff', '#c8e6c9', '#66bb6a', '#2d6e3e']
    cmap = LinearSegmentedColormap.from_list('gray_green', colors_list, N=100)
    
    # Set white background
    ax.set_facecolor('white')
    
    # Add grid
    ax.grid(True, color='lightgray', linestyle='-', alpha=0.4, zorder=0)
    
    # Create scatter plot
    scatter = ax.scatter(
        conv_times, 
        performance,
        c=densities,
        cmap=cmap,
        s=50,
        alpha=0.8,
        edgecolors='#666666',
        linewidths=0.5,
        vmin=0,
        vmax=1,
        zorder=3
    )
    
    # Add simplified annotations for extremes
    # Lowest density (specialist)
    ax.text(conv_times[0] + 2, performance[0], 'specialist',
            fontsize=6, color='#888888', ha='left', va='center')
    
    # Highest density (generalist)  
    ax.text(conv_times[-1], performance[-1] + 0.02, 'generalist',
            fontsize=6, color='#888888', ha='left', va='center')
    
    # Styling
    ax.set_xlabel('Convergence Time (steps)', fontsize=10)
    ax.set_ylabel('Performance', fontsize=9)
    ax.set_xlim(conv_times.min() - 5, conv_times.max() + 5)
    ax.set_ylim(performance.min() - 0.02, performance.max() + 0.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(size=0)
    fig.patch.set_alpha(0)
    
    plt.tight_layout()
    save_pub_fig('methods_means')


def main():
    """Generate all three methodology figure elements."""
    
    # Generate figures
    create_convergence_plot()
    create_heatmap_preview()
    create_means_plot_preview()


if __name__ == '__main__':
    main()