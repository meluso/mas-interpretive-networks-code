# tests/models/networks/test_generators.py
"""Tests for network generators."""

import pytest
import networkx as nx
import numpy as np
from models.networks.generators import (
    validate_graph_inputs,
    get_graph,
    GRAPH_GENERATORS
)

@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("graph_type", GRAPH_GENERATORS.keys())
def test_graph_sizes(graph_type, size):
    """Test that all generators produce correct size graphs."""
    
    if size == 1 and (graph_type != 'empty'):
        with pytest.raises(ValueError):
            _call_graph(graph_type, size)
            
    elif size == 2 and (graph_type != 'empty') and (graph_type != 'complete'):
        with pytest.raises(ValueError):
            _call_graph(graph_type, size)
    
    else:
        _call_graph(graph_type, size)
            
        
            
def _call_graph(graph_type, size):
    G = get_graph(graph_type, size)
    assert len(G) == size
    assert isinstance(G, nx.Graph)
    

def test_invalid_graph_type():
    """Test error on invalid graph type."""
    with pytest.raises(ValueError):
        get_graph("invalid_type", 4)

def test_graph_properties():
    """Test specific properties of each graph type."""
    # Complete graph should have n(n-1)/2 edges
    for n in [4, 8]:
        G = get_graph('complete', n)
        assert len(G.edges) == (n * (n-1)) // 2
        
    # Star graph should have n-1 edges
    for n in [4, 8]:
        G = get_graph('star', n)
        assert len(G.edges) == n-1
        
    # Tree should have n-1 edges
    for n in [4, 8]:
        G = get_graph('tree', n)
        assert len(G.edges) == n-1
        assert nx.is_tree(G)
        
    # Wheel should have 2n-2 edges
    for n in [4, 8]:
        G = get_graph('wheel', n)
        assert len(G.edges) == 2*n - 2

def test_random_reproducibility():
    """Test that random graphs are reproducible with same seed."""
    import random
    import numpy as np
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    
    # Generate first set
    G1_power = get_graph('power', 8)
    G1_random = get_graph('random', 8)
    G1_tree = get_graph('tree', 8)
    
    # Reset seeds
    random.seed(42)
    np.random.seed(42)
    
    # Generate second set
    G2_power = get_graph('power', 8)
    G2_random = get_graph('random', 8)
    G2_tree = get_graph('tree', 8)
    
    # Compare
    assert nx.is_isomorphic(G1_power, G2_power)
    assert nx.is_isomorphic(G1_random, G2_random)
    assert nx.is_isomorphic(G1_tree, G2_tree)

def test_hypercube_structure():
    """Test hypercube dimensional properties."""
    for n in [4, 8, 16]:
        G = get_graph('hypercube', n)
        # Number of edges should be n*log2(n)/2
        assert len(G.edges) == int(n * np.log2(n) / 2)
        # All nodes should have same degree = log2(n)
        assert all(d == np.log2(n) for _, d in G.degree())

def test_small_world_params():
    """Test small world parameter handling."""
    G1 = get_graph('small_world', 8, k=2, p=0.0)
    G2 = get_graph('small_world', 8, k=2, p=1.0)
    
    # p=0 should be regular ring lattice
    assert all(d == 2 for _, d in G1.degree())
    
    # p=1 should be different from p=0
    assert not nx.is_isomorphic(G1, G2)

def test_power_params():
    """Test power law parameter handling."""
    G1 = get_graph('power', 8, m=1, p=0.0)
    G2 = get_graph('power', 8, m=2, p=0.5)
    
    # m=1 should give tree-like structure
    assert len(G1.edges) == 7  # n-1 edges
    
    # Different parameters should give different graphs
    assert not nx.is_isomorphic(G1, G2)

def test_windmill_structure():
    """Test windmill graph properties."""
    for n in [4, 8, 16]:
        G = get_graph('windmill', n)
        # Should be connected
        assert nx.is_connected(G)
        # Should have correct number of nodes
        assert len(G) == n

if __name__ == '__main__':
    pytest.main([__file__])