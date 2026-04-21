# figures/plot_objective.py
"""Visualization tools for objective functions."""

import numpy as np
import matplotlib.pyplot as plt
from models.objectives.objective import Sin2, Sin2sphere, Sin2root, Losqr_hiroot, Hisqr_loroot

def get_default_opts(objective_class, weight_type='unweighted'):
    """Get default options for an objective class."""
    opts = {'weight': 'degree' if weight_type == 'weighted' else 'node'}
    
    # Add frequency option for classes that need it
    if objective_class in [Sin2, Sin2sphere, Sin2root]:
        opts['frequency'] = 'uniform'
    
    # Add exponent option for classes that need it
    if objective_class in [Losqr_hiroot, Hisqr_loroot]:
        opts['exponent'] = 'uniform'
        
    return opts

def plot_objective_surface(objective_class, num_vars=1, degrees=[1, 5],
                           x_range=(0, 1), n_points=100,
                           weight_type='unweighted', **kwargs):
    """
    Create a 3D surface plot of an objective function for two variables.
    Domain is [0,1] x [0,1].
    """
    # Ensure we only use 2 degrees for 2D visualization
    degrees = np.array(degrees[:2])
    
    # Create options dictionary with defaults
    opts = get_default_opts(objective_class, weight_type)
    opts.update(kwargs)
    
    # Create the objective function instance
    obj = objective_class(opts, num_vars, degrees)
    
    # Create the grid of x values
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(x_range[0], x_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate Z values
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            x_vals = np.array([X[i,j], Y[i,j]])
            Z[i,j] = obj(x_vals)
    
    # Create the 3D plot
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    
    # Add colorbar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title with LaTeX formatting
    ax.yaxis.set_rotate_label(False)
    # ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r'$x_1$', labelpad=-7)
    ax.set_ylabel(r'$x_2$', labelpad=-7)
    # ax.set_zlabel(r'$f(x)$', labelpad=-7, )
    ax.text(x=0.85, y=1, z=1.2, s='$g(x_1,x_2)$', size=8)
    weight_label = 'weighted by degree' if weight_type == 'weighted' else 'unweighted'
    # ax.set_title(f'{objective_class.__name__} ({weight_label})\nDegrees: {list(degrees)}')
    
    # Set background to white
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    
    return fig, ax

def set_default_style():
    """Set consistent style for all objective function plots."""
    plt.rcParams['figure.figsize'] = [2, 2]
    plt.rcParams['axes.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6

def save_objective_plot(fig, filename, dpi=300):
    """Save objective function plot with consistent formatting."""
    fig.savefig(f'figures/objectives/{filename}', 
                dpi=dpi, 
                bbox_inches='tight',
                facecolor='white')

if __name__ == '__main__':
    from models.objectives.objective import Ackley
    from figures.util import set_fonts, save_publication_fig
    set_fonts()
    set_default_style()
    fig, ax = plot_objective_surface(Ackley)
    ax.tick_params(axis='both', pad=-3)
    plt.show()
    fig.savefig('figures/objectives/ackley.svg', dpi=600, bbox_inches='tight', transparent=True)