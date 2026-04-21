"""Network generators for team graphs.

This module provides functionality for generating different types of networks
for use in team simulations. All generators enforce size constraints of either
1, 2, or powers of 2 up to 32 nodes.
"""

import networkx as nx
import numpy as np

def validate_graph_inputs(graph_type, size):
    """Validates that graph size meets requirements.
    
    Args:
        graph_type (string): Type of network
        size (int): Number of nodes requested
        
    Raises:
        ValueError: If graph type is not in GRAPH_GENERATORS
        ValueError: If size not in [1, 2, 4, 8, 16, 32]
    """
    
    # Check for the graph type
    if graph_type not in GRAPH_GENERATORS:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    # Check that the size is valid
    valid_sizes = [1, 2, 4, 8, 16, 32]
    if size not in valid_sizes:
        raise ValueError(f"Size must be in {valid_sizes}")
    
    # Check that specific sizes are specific types
    if size == 1 and graph_type != 'empty':
        raise ValueError(f"Graphs of size 1 can only be empty, not {graph_type}.")
    if size == 2 and graph_type != 'empty' and graph_type != 'complete':
        raise ValueError(f"Graphs of size 2 can only be empty or complete, not {graph_type}.")
    

def get_trivial():
    """Returns a single-node graph."""
    G = nx.Graph()
    G.add_node(0)
    return G

def get_path():
    """Returns a 2-node path graph."""
    return nx.path_graph(2)

def get_complete(size, **opts):
    """Returns a complete graph of given size."""
    return nx.complete_graph(size)

def get_empty(size, **opts):
    """Returns an empty graph of given size."""
    return nx.empty_graph(size)

def get_hypercube(size, **opts):
    """Returns a hypercube graph of given size."""
    G = nx.hypercube_graph(int(np.log2(size)))
    return nx.convert_node_labels_to_integers(G)

def get_small_world(size, k=2, p=0.0, **opts):
    """Returns a small world graph of given size."""
    return nx.watts_strogatz_graph(size, k, p)

def get_power(size, m=2, p=0.3, **opts):
    """Returns a power law graph of given size."""
    return nx.powerlaw_cluster_graph(size, m, p)

def get_random(size, p=0.3, **opts):
    """Returns an Erdős-Rényi random graph of given size."""
    return nx.gnp_random_graph(size, p)

def get_ring_cliques(team_size, **team_graph_opts):
    '''Returns an instance of a ring of cliques graph.'''
    
    # Return a ring of cliques with cliques of 4 if >4 team size
    if team_size > 4:
        clique_size = 4
        num_cliques = int(team_size // clique_size)  
        return nx.ring_of_cliques(num_cliques, clique_size)

    # Otherwise, return the equivalent ring for 4 or less
    return get_small_world(team_size)

def get_star(size, **opts):
    """Returns a star graph of given size."""
    return nx.star_graph(size - 1)

def get_tree(size, **opts):
    """Returns a random tree of given size."""
    return nx.random_labeled_tree(size)

def get_wheel(size, **opts):
    """Returns a wheel graph of given size."""
    return nx.wheel_graph(size)

def get_windmill(size, **opts):
    """Returns a modified windmill graph of given size."""
        
    # Pre-computed optimal parameters for each size
    # Format: (p, k) where p is number of cliques and k is clique size
    OPTIMAL_PARAMS = {
        4: (2, 3),    # 2 cliques of size 2 (after central removal)
        8: (4, 3),    # 4 cliques of size 3 (after central removal)
        16: (4, 5),   # 4 cliques of size 4 (after central removal)
        32: (4, 9)    # 4 cliques of size 8 (after central removal)
    }
    
    p, k = OPTIMAL_PARAMS[size]
    G = nx.windmill_graph(p, k)
    G.remove_node(size)  # Remove last node
    return nx.convert_node_labels_to_integers(G)

# Main interface
GRAPH_GENERATORS = {
    'complete': get_complete,
    'empty': get_empty,
    'hypercube': get_hypercube,
    'power': get_power,
    'random': get_random,
    'ring_cliques': get_ring_cliques,
    'small_world': get_small_world,
    'star': get_star,
    'tree': get_tree,
    'wheel': get_wheel,
    'windmill': get_windmill,
}

def get_graph(graph_type, size, **opts):
    """Main interface for graph generation.
    
    Args:
        graph_type (str): Type of graph to generate
        size (int): Number of nodes
        **opts: Additional options passed to specific generators
        
    Returns:
        networkx.Graph: Generated graph
        
    Raises:
        ValueError: If graph_type not recognized or size invalid
    """
    
    # Validate the graph inputs
    validate_graph_inputs(graph_type, size)
        
    return GRAPH_GENERATORS[graph_type](size, **opts)