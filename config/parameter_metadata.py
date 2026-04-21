# config/parameter_metadata.py

"""Parameter metadata for AI teams simulation.

This module provides human-readable labels and descriptions for simulation parameters,
complementing the metric definitions in metrics.py.
"""

# Parameter metadata
PARAMETER_METADATA = {
    # Team parameters
    'team_size': {
        'label': 'Team Size',
        'description': 'Number of agents in the team',
        'type': 'parameter'
    },
    'team_graph_type': {
        'label': 'Network Type',
        'description': 'Structure of connections between agents',
        'type': 'parameter'
    },
    'team_graph_slug': {
        'label': 'Network Configuration',
        'description': 'Full network specification with parameters',
        'type': 'parameter'
    },
    
    # Agent parameters
    'agent_num_vars': {
        'label': 'Variables per Agent',
        'description': 'Number of variables controlled by each agent',
        'type': 'parameter'
    },
    'agent_steplim': {
        'label': 'Agent Step Limit',
        'description': 'Maximum step size for variable updates',
        'type': 'parameter'
    },
    'agent_optim_type': {
        'label': 'Optimization Algorithm',
        'description': 'Algorithm used by agents for optimization',
        'type': 'parameter'
    },
    
    # Function parameters
    'fn_type': {
        'label': 'Function Type',
        'description': 'Type of objective function',
        'type': 'parameter'
    },
    'fn_slug': {
        'label': 'Function Configuration',
        'description': 'Full function specification with parameters',
        'type': 'parameter'
    },
    
    # Transformed variables
    'log_agent_steplim': {
        'label': 'Log Agent Step Limit',
        'description': 'Log-transformed agent step limit',
        'type': 'transformed'
    },
    'log_team_fn_diff_peaks': {
        'label': 'Log Number of Local Optima',
        'description': 'Log-transformed number of local maxima',
        'type': 'transformed'
    },
    
    # Normalized variables (prefix patterns)
    'norm_': {
        'prefix': True,
        'label_prefix': 'Normalized ',
        'description_prefix': 'Min-max normalized '
    }
}

# Graph type metadata
GRAPH_TYPE_METADATA = {
    'complete': {
        'label': 'Complete Network',
        'description': 'All agents connected to all other agents'
    },
    'hypercube': {
        'label': 'Hypercube Network',
        'description': 'Regular structure with connections along dimensional axes'
    },
    'power': {
        'label': 'Power Law Network',
        'description': 'Scale-free network with preferential attachment'
    },
    'random': {
        'label': 'Random Network',
        'description': 'Erdős-Rényi random graph'
    },
    'ring_cliques': {
        'label': 'Ring of Cliques',
        'description': 'Dense groups connected in a ring structure'
    },
    'small_world': {
        'label': 'Small World Network',
        'description': 'Watts-Strogatz small world network'
    },
    'star': {
        'label': 'Star Network',
        'description': 'Central hub connected to all other agents'
    },
    'tree': {
        'label': 'Tree Network',
        'description': 'Hierarchical structure without cycles'
    },
    'wheel': {
        'label': 'Wheel Network',
        'description': 'Central hub with peripheral agents in a ring'
    },
    'windmill': {
        'label': 'Windmill Network',
        'description': 'Central hub connected to multiple cliques'
    }
}

# Function type metadata
FUNCTION_TYPE_METADATA = {
    'average': {
        'label': 'Average',
        'description': 'Simple sum function (aligned)'
    },
    'sphere': {
        'label': 'Sphere',
        'description': 'Sum of squares function (aligned)'
    },
    'root': {
        'label': 'Root',
        'description': 'Sum of square roots function (aligned)'
    },
    'sin2': {
        'label': 'Sine Squared',
        'description': 'Sum of squared sines (multiple local optima)'
    },
    'sin2sphere': {
        'label': 'Sine Squared + Sphere',
        'description': 'Combined sine squared and sphere functions'
    },
    'sin2root': {
        'label': 'Sine Squared + Root',
        'description': 'Combined sine squared and root functions'
    },
    'losqr_hiroot': {
        'label': 'Low Square High Root',
        'description': 'Function varying with node degree (not aligned)'
    },
    'hisqr_loroot': {
        'label': 'High Square Low Root',
        'description': 'Function varying with node degree (not aligned)'
    },
    'max': {
        'label': 'Maximum',
        'description': 'Maximum value function (not aligned)'
    },
    'min': {
        'label': 'Minimum',
        'description': 'Minimum value function (not aligned)'
    },
    'median': {
        'label': 'Median',
        'description': 'Median value function (not aligned)'
    },
    'kth_power': {
        'label': 'K-th Power',
        'description': 'Variables to power of their degree (aligned)'
    },
    'kth_root': {
        'label': 'K-th Root',
        'description': 'Variables to root of their degree (aligned)'
    },
    'ackley': {
        'label': 'Ackley',
        'description': 'Multi-modal test function (not aligned)'
    }
}