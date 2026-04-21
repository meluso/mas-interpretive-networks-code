# figures/plot_joint_task_effects.py
"""Plot joint figure of network-task performance relationships.

Creates a figure with two subfigures:
1. Network metrics heatmap on the left
2. 2x2 grid of biaxial plots on the right showing task-conditional effects by graph type
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import figure settings utilities
from figures.util import set_fonts, fig_size, save_pub_fig, get_optimizer

# Import component plots
from figures.plot_network_metric_task_performance_heatmap import plot_network_metric_task_heatmap
from figures.plot_graph_slug_task_performance_biaxials import plot_biaxial_grid


def plot_joint_task_effects(study_name='aiteams01nm_20250128_223001', 
                          dataset_name='dataset2',
                          outcome_metric='convergence_performance',
                          save=True):
    """Create joint figure with heatmap and biaxial plots."""
    # Set up fonts
    set_fonts()
    
    # Create single figure with good margins
    fig = plt.figure(figsize=fig_size(frac_width=1.0, frac_height=0.5))
    
    # Create main GridSpec - just define the top-level division
    main_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.15, left=0.2, right=0.98, bottom=0.05, top=0.95)
    
    # Call component functions with main GridSpec cells
    # Each function will create its own nested GridSpec as needed
    plot_network_metric_task_heatmap(study_name, dataset_name, outcome_metric, 
                                    fig=fig, gs_cell=main_gs[0, 0])
    
    plot_biaxial_grid(study_name=study_name, 
                     fig=fig, gs_cell=main_gs[0, 1])
    
    # Create figure-wide annotations
    fig.text(0.025, 0.95, "A", size=12)
    fig.text(0.55, 0.95, "B", size=12)
    
    # Save figure
    optimizer = get_optimizer(study_name)
    figure_name = f"joint_task_effects_{optimizer}"
    if save: save_pub_fig(figure_name)
    
    return fig

if __name__ == "__main__":
    fig = plot_joint_task_effects()
    plt.show()