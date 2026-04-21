# models/optimizers/base.py
from abc import ABC, abstractmethod
import numpy as np
from numpy.random import default_rng

class BaseOptimizer(ABC):
    """Abstract base class for optimization strategies.
    
    Handles optimization of vector-valued variables in [0,1]^n domain.
    """
    
    def __init__(self, objective_fn=None, rng=None):
        """Initialize optimizer with objective function and optional RNG.
        
        Args:
            objective_fn (callable): Function to optimize
            rng (numpy.random.Generator, optional): Random number generator
        """
        self.objective_fn = objective_fn
        self.rng = rng if rng is not None else default_rng()
    
    @abstractmethod
    def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
        """Execute one optimization step.
        
        Args:
            current_x (np.ndarray): Current position vector in [0,1]^n
            current_fx (float): Current objective value
            step_limit (float): Maximum allowed step size
            neighborhood_state (list, optional): States of neighboring agents
            
        Returns:
            tuple: (new_x, new_fx) containing the new position and its objective value
        """
        pass
    
    def _evaluate(self, x, neighborhood_state=None):
        """Evaluate objective function with neighborhood state.
        
        Args:
            x (np.ndarray): Position to evaluate
            neighborhood_state (list, optional): States of neighboring agents
            
        Returns:
            float: Objective value
        """
        nbhd_state = [x] + neighborhood_state[1:] if neighborhood_state else [x]
        return float(self.objective_fn(np.concatenate(nbhd_state)))
    
    def _get_bounds(self, current_x, step_limit):
        """Get optimization bounds based on current position and step limit.
        
        Args:
            current_x (np.ndarray): Current position
            step_limit (float): Maximum step size
            
        Returns:
            tuple: (lower_bounds, upper_bounds) arrays
        """
        if step_limit >= 1:
            return np.zeros_like(current_x), np.ones_like(current_x)
        
        lower_bounds = np.maximum(0, current_x - step_limit)
        upper_bounds = np.minimum(1, current_x + step_limit)
        return lower_bounds, upper_bounds

    def _objective_to_minimize(self, x, neighborhood_state=None):
        """Convert maximization objective to minimization form.
        
        Args:
            x (np.ndarray): Position to evaluate
            neighborhood_state (list, optional): States of neighboring agents
            
        Returns:
            float: Negative of objective value (for minimization)
        """
        return -self._evaluate(x, neighborhood_state)
