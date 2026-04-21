# config/metrics.py

"""Defines the metrics used throughout the simulations."""

# Define outcome metrics
OUTCOME_METRICS = {
    'convergence_step': {
        'label': 'Steps to Convergence',
        'description': 'Number of steps until performance stabilizes',
        'type': 'outcome'
    },
    'convergence_performance': {
        'label': 'Performance at Convergence',
        'description': 'Team objective value when convergence is reached',
        'type': 'outcome'
    },
    'final_performance': {
        'label': 'Final Performance',
        'description': 'Team performance at the end of simulation',
        'type': 'outcome'
    }
}

# Define network metrics
NETWORK_METRICS = {
    'team_graph_centrality_degree_mean': {
        'label': 'Indiv. Connections (deg. cent. avg.)',
        'description': 'Average number of connections per node',
        'type': 'network'
    },
    'team_graph_centrality_degree_stdev': {
        'label': 'Connect. Variation (deg. cent. std.)',
        'description': 'Variation in number of connections across nodes',
        'type': 'network'
    },
    'team_graph_centrality_betweenness_mean': {
        'label': 'Intermediaries (bet. cent. avg.)',
        'description': 'Average frequency of nodes lying on shortest paths',
        'type': 'network'
    },
    'team_graph_centrality_betweenness_stdev': {
        'label': 'Varied Intermediaries (bet. cent. std.)',
        'description': 'Variation in nodes serving as information bridges',
        'type': 'network'
    },
    'team_graph_centrality_eigenvector_mean': {
        'label': 'Decentralization (eig. cent. avg.)',
        'description': 'Degree to which influence is distributed across nodes',
        'type': 'network'
    },
    'team_graph_centrality_eigenvector_stdev': {
        'label': 'Centralization (eig. cent. std.)',
        'description': 'Concentration of influence in specific nodes',
        'type': 'network'
    },
    'team_graph_nearest_neighbor_degree_mean': {
        'label': 'Neighbor Connect. (nnd avg.)',
        'description': 'Average degree of neighboring nodes',
        'type': 'network'
    },
    'team_graph_nearest_neighbor_degree_stdev': {
        'label': 'Neighbor Connect. Var. (nnd std.)',
        'description': 'Variation in connection patterns of neighbors',
        'type': 'network'
    },
    'team_graph_clustering': {
        'label': 'Triangle Density (clust. coeff.)',
        'description': 'Tendency of nodes to form tightly connected groups',
        'type': 'network'
    },
    'team_graph_density': {
        'label': 'Network Density',
        'description': 'Ratio of existing connections to possible connections',
        'type': 'network'
    },
    'team_graph_assortativity': {
        'label': 'Connection Homophily (deg. assort.)',
        'description': 'Tendency of nodes to connect to similar degree nodes',
        'type': 'network'
    },
    'team_graph_pathlength': {
        'label': 'Path Length (avg. path len.)',
        'description': 'Average shortest path between all node pairs',
        'type': 'network'
    },
    'team_graph_diameter': {
        'label': 'Max. Distance (diameter)',
        'description': 'Maximum shortest path between any node pair',
        'type': 'network'
    }
}

# Define objective function metrics
OBJECTIVE_METRICS = {
    'team_fn_diff_integral': {
        'label': 'Integral Difficulty',
        'description': 'Area above the function (global difficulty)',
        'type': 'task'
    },
    'team_fn_diff_peaks': {
        'label': 'Number of Local Optima',
        'description': 'Number of local maxima in the objective function',
        'type': 'task'
    },
    'team_fn_diff_alignment': {
        'label': 'Task Alignment',
        'description': 'Similarity of objective values across variables',
        'type': 'task'
    },
    'team_fn_diff_interdep': {
        'label': 'Task Interdependence',
        'description': 'Degree of variable interactions in the task',
        'type': 'task'
    }
}

# Derived metrics (those that should be transformed)
VARIABLES_TO_LOG_TRANSFORM = [
    'agent_steplim',
    'team_fn_diff_peaks'
]

# Special network metrics (those that may have validity constraints)
SPECIAL_NETWORK_METRICS = [
    'team_graph_pathlength',
    'team_graph_diameter',
    'team_graph_assortativity'
]