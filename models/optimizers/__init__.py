# models/optimizers/__init__.py
from models.optimizers.base import BaseOptimizer
from models.optimizers.random_walk import RandomWalkOptimizer
from models.optimizers.scipy_wrappers import (
    DualAnnealingOptimizer,
    NelderMeadOptimizer,
    LBFGSBOptimizer
)

# Registry of available optimizers
OPTIMIZERS = {
    'random_walk': RandomWalkOptimizer,
    'dual_annealing': DualAnnealingOptimizer,
    'nelder_mead': NelderMeadOptimizer,
    'lbfgsb': LBFGSBOptimizer,
}

def create_optimizer(optimizer_type, **kwargs):
    """Factory function to create optimizer instances."""
    if optimizer_type not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return OPTIMIZERS[optimizer_type](**kwargs)
