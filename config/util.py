# config/util.py

from typing import Dict, List, Union, Any, Tuple
import numpy as np

# Datatypes for storage
STORAGE_DTYPES = {
    # Integers
    'num_trial': 'uint16',
    'team_size': 'uint8',
    'num_steps': 'uint8',
    'agent_num_vars': 'uint8',
    'convergence_step': 'uint8',
    
    # Floats
    'agent_steplim': 'float32',
    'team_graph_centrality_degree_mean': 'float32',
    'team_graph_centrality_degree_stdev': 'float32',
    'team_graph_centrality_betweenness_mean': 'float32',
    'team_graph_centrality_betweenness_stdev': 'float32',
    'team_graph_centrality_eigenvector_mean': 'float32',
    'team_graph_centrality_eigenvector_stdev': 'float32',
    'team_graph_nearest_neighbor_degree_mean': 'float32',
    'team_graph_nearest_neighbor_degree_stdev': 'float32',
    'team_graph_assortativity': 'float32',
    'team_graph_clustering': 'float32',
    'team_graph_density': 'float32',
    'team_graph_pathlength': 'float32',
    'team_graph_diameter': 'float32',
    'team_fn_diff_peaks': 'float64',
    'team_fn_diff_integral': 'float32',
    'team_fn_diff_alignment': 'float32',
    'team_fn_diff_interdep': 'float32',
    'convergence_performance': 'float32',
    'final_performance': 'float32',
    
    # Categories
    'agent_optim_type': 'category',
    'team_graph_slug': 'category',
    'fn_slug': 'category'
}

def validate_input_dict(input_dict: Dict[str, Any]) -> None:
    """
    Validates an input dictionary against predefined rules and conditions.
    Raises AssertionError if any validation fails.
    
    Args:
        input_dict: Dictionary to validate where each value can be a single value or list
        
    Raises:
        AssertionError: If any validation condition is not met
    """
    # Define options for objective functions
    weight_opts = ['node', 'degree']
    frequency_opts = ['uniform', 'degree']
    exponent_opts = ['uniform', 'degree']
    
    # Define option groups
    weight = {'weight': weight_opts}
    weight_and_frequency = {'weight': weight_opts, 'frequency': frequency_opts}
    weight_and_exponent = {'weight': weight_opts, 'exponent': exponent_opts}
    
    valid = {
        'team_size': [1, 2, 4, 8, 16, 32],
        'team_graph_type': {
            'complete': None,
            'hypercube': None,
            'power': ['m','p'],
            'random': ['p'],
            'ring_cliques': None,
            'small_world': ['k','p'],
            'star': None,
            'tree': None,
            'wheel': None,
            'windmill': None
        },
        'agent_steplim': (0,1),
        'agent_optim_type': ['random_walk', 'dual_annealing', 'nelder_mead', 'lbfgsb'],
        'fn_type': {
            'average': weight,
            'sphere': weight,
            'root': weight,
            'sin2': weight_and_frequency,
            'sin2sphere': weight_and_frequency,
            'sin2root': weight_and_frequency,
            'losqr_hiroot': weight_and_exponent,
            'hisqr_loroot': weight_and_exponent,
            'max': None,
            'min': None,
            'median': None,
            'kth_power': weight,
            'kth_root': weight,
            'ackley': None
        },
        'num_steps': list(range(1,1001)),
        'num_trials': list(range(1,1001))
    }
    
    def validate_value(key: str, value: Any, valid_options: Union[List, Dict, Tuple, None]) -> None:
        """Helper function to validate a single value against its valid options."""
        # Check if value is a list
        values = value if isinstance(value, list) else [value]
        
        for val in values:
            if valid_options is None:
                # If valid_options is None, any value is acceptable
                continue
                
            elif isinstance(valid_options, tuple):
                # For range validation (a,b]
                assert valid_options[0] < val <= valid_options[1], \
                    f"Value {val} for key '{key}' must be in range ({valid_options[0]}, {valid_options[1]}]"
                    
            elif isinstance(valid_options, list):
                # For explicit list of valid options
                assert val in valid_options, \
                    f"Value {val} for key '{key}' must be one of {valid_options}"
                    
            elif isinstance(valid_options, dict):
                # For nested dictionary validation
                assert val in valid_options, \
                    f"Value {val} for key '{key}' must be one of {list(valid_options.keys())}"
                    
                # If the valid option has sub-options, validate them
                if valid_options[val] is not None:
                    # Get the corresponding sub-dictionary from input_dict
                    sub_dict = input_dict.get(val)
                    if sub_dict:
                        # Validate each key-value pair in the sub-dictionary
                        for sub_key, sub_value in sub_dict.items():
                            sub_valid_options = valid_options[val][sub_key]
                            validate_value(f"{key}.{val}.{sub_key}", sub_value, sub_valid_options)
    
    # Validate that all input keys exist in valid dictionary
    for key in input_dict:
        assert key in valid, f"Invalid key: '{key}' not found in valid options"
    
    # Validate each key-value pair
    for key, value in input_dict.items():
        validate_value(key, value, valid[key])

def create_graph_slug(graph_type: str, graph_opts: dict) -> str:
    """Create a slug string from graph parameters.
    
    Args:
        graph_type: Type of graph (e.g. 'small_world')
        graph_opts: Dictionary of graph options or None
        
    Returns:
        String slug combining type and parameters
        
    Examples:
        >>> create_graph_slug('small_world', {'k': 2, 'p': 0.3})
        'small_world_k2_p03'
        >>> create_graph_slug('complete', None)
        'complete'
    """
    if graph_opts is None:
        return graph_type
        
    # Extract parameters and format as needed
    m = graph_opts.get('m', '')
    k = graph_opts.get('k', '')
    p = str(graph_opts.get('p', '')).replace('.', '')
    
    # Build slug parts, filtering out empty strings
    parts = [graph_type]
    if m:
        parts.append(f"m{m}")
    if k:
        parts.append(f"k{k}")
    if p:
        parts.append(f"p{p}")
        
    return '_'.join(parts)

def create_fn_slug(fn_type: str, fn_opts: dict) -> str:
    """Create a slug string from function parameters.
    
    Args:
        fn_type: Type of function (e.g. 'sin2')
        fn_opts: Dictionary of function options or None
        
    Returns:
        String slug combining type and parameters
        
    Examples:
        >>> create_fn_slug('sin2', {'weight': 'node', 'frequency': 'uniform'})
        'sin2_node_uniform'
        >>> create_fn_slug('sphere', None)
        'sphere'
    """
    if fn_opts is None:
        return fn_type
        
    # Extract parameters with descriptive prefixes
    weight = f"w{fn_opts.get('weight', '').capitalize()}" if 'weight' in fn_opts else ''
    freq = f"f{fn_opts.get('frequency', '').capitalize()}" if 'frequency' in fn_opts else ''
    exp = f"e{fn_opts.get('exponent', '').capitalize()}" if 'exponent' in fn_opts else ''
    
    # Build slug parts, filtering out empty strings
    parts = [fn_type]
    if weight:
        parts.append(weight)    # e.g. 'wNode', 'wDegree'
    if freq:
        parts.append(freq)      # e.g. 'fUniform', 'fDegree' 
    if exp:
        parts.append(exp)       # e.g. 'eUniform', 'eDegree'
        
    return '_'.join(parts)