# models/optimizers/random_walk.py
from models.optimizers.base import BaseOptimizer

class RandomWalkOptimizer(BaseOptimizer):
    """Implements random walk optimization strategy with uniform sampling.
    
    Uses component-wise step limits for simplicity and efficiency, sampling
    uniformly from a box of width 2*step_limit centered at the current position,
    constrained to remain within the [0,1]^n domain.
    """
    
    def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
        """Execute one optimization step using random walk with hill climbing.
        
        Args:
            current_x (np.ndarray): Current position vector in [0,1]^n
            current_fx (float): Current objective value
            step_limit (float): Maximum allowed step size per component
            neighborhood_state (list, optional): States of neighboring agents
            
        Returns:
            tuple: (new_x, new_fx) for best position found
        """
        # Get bounds for this step
        lower_bounds, upper_bounds = self._get_bounds(current_x, step_limit)
        
        # Sample new position uniformly from constrained box
        proposed_x = self.rng.uniform(lower_bounds, upper_bounds)
        
        # Evaluate proposal
        proposed_fx = self._evaluate(proposed_x, neighborhood_state)
        
        # Return best position (hill climbing)
        if proposed_fx > current_fx:
            return proposed_x, proposed_fx
        return current_x, current_fx
