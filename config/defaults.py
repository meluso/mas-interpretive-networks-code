# config/defaults.py

"""Default configuration settings for simulations."""

# Storage settings
DEFAULT_DATA_DIR = "data/raw"

# Simulation settings
DEFAULT_NUM_STEPS = 25
DEFAULT_MC_TRIALS = 100

# Campaign settings
DEFAULT_RANDOM_SEED = None
DEFAULT_NUM_TRIALS = 100

# System settings
DEFAULT_NUM_PROCESSES = 4  # For parallel execution
    
# Set default graphs and options
graph2opts = {
    'complete': None,
    'empty': None,
    'hypercube': None,
    'power': {'m': [2, 4],'p': [0.3, 0.7]},
    'random': {'p': [0.3, 0.7]},
    'ring_cliques': None,
    'small_world': {'k': [2, 4], 'p': [0.3, 0.7]},
    'star': None,
    'tree': None,
    'wheel': None,
    'windmill': None
    }
    
# Define options for objective functions
weight_opts = ['node','degree']
frequency_opts = ['uniform','degree']
exponent_opts = ['degree']

# Define option groups
weight = {'weight': weight_opts}
weight_and_frequency = {'weight': weight_opts, 'frequency': frequency_opts}
weight_and_exponent = {'weight': weight_opts, 'exponent': exponent_opts}

# Set default objective functions and options
fn2opts = {
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
}