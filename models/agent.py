# models/agent.py
from numpy import all, concatenate, ndarray
from typing import Optional, Dict, Callable

class Agent:
    """An agent controlling a subset of team decision variables."""
    
    def __init__(self, 
                 num_vars: int,
                 optimizer_type: str,
                 objective_fn: Callable,
                 optimizer_opts: Optional[Dict] = None,
                 step_limit: float = 0.1,
                 rng = None,
                 track_trajectories: bool = False):
        """Initialize an agent with its objective function and optimizer.
        
        Args:
            num_vars: Number of variables this agent controls
            optimizer_type: Type of optimizer to use ('random_walk', 'dual_annealing', etc.)
            objective_fn: Function to optimize
            optimizer_opts: Optional parameters for the optimizer
            step_limit: Maximum step size for variable updates
            rng: Random number generator
            track_trajectories: Whether to store optimization trajectories
        """
        from models.optimizers import create_optimizer
        
        self.num_vars = num_vars
        self.step_limit = step_limit
        self.objective_fn = objective_fn
        self.track_trajectories = track_trajectories
        self.trajectories = [] if track_trajectories else None
        
        # Initialize optimizer with options
        self.optimizer = create_optimizer(
            optimizer_type,
            objective_fn=objective_fn,
            **(optimizer_opts or {}),
            rng=rng
        )
        
        # Initialize state
        self.x = rng.random(size=num_vars)
        self.fx = None
    
    def step(self, nbhd_state) -> ndarray:
        """Execute one optimization step using the agent's optimizer.
        
        Args:
            nbhd_state: List of states [agent_state, neighbor1_state, ...]
        
        Returns:
            ndarray: New position vector
        """
        # Get current objective value for optimizer
        nbhd_x = concatenate(nbhd_state)
        current_fx = self.objective_fn(nbhd_x)
        
        # Execute optimization step
        new_x, new_fx = self.optimizer.step(
            self.x,
            current_fx,
            self.step_limit,
            nbhd_state
        )
        
        # Validate solution is within bounds
        if not all((new_x >= 0) & (new_x <= 1)):
            raise ValueError(
                f"Optimizer returned solution outside [0,1] bounds: {new_x}. "
                f"This indicates a problem with the optimization algorithm's bound constraints."
                )
        
        # Store trajectory if tracking enabled
        if self.track_trajectories:
            self.trajectories.append({
                'x': new_x.copy(),
                'fx': new_fx
            })
            
        return new_x
    
    def set_x(self, x):
        """Set the agent's x."""
        self.x = x
    
    def get_x(self):
        """Get the agent's x."""
        return self.x
    
    def set_fx(self, nbhd_state):
        """Set the agent's f(x) based on neighborhood state."""
        if not isinstance(nbhd_state, ndarray):
            nbhd_state = concatenate(nbhd_state)
        self.fx = self.objective_fn(nbhd_state)
    
    def get_fx(self):
        """Get the agent's f(x)."""
        return self.fx
    
    @property
    def state(self) -> Dict:
        """Get current agent state."""
        return {
            'x': self.x,
            'fx': self.fx,
            'trajectories': self.trajectories if self.track_trajectories else None
        }
