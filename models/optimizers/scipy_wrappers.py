# models/optimizers/scipy_wrappers.py
import numpy as np
from scipy import optimize
from .base import BaseOptimizer

class DualAnnealingOptimizer(BaseOptimizer):
    """Wrapper for scipy's dual annealing implementation.
    
    Parameters that can be set via optimizer_opts:
        maxiter (int): Maximum number of iterations (default: 100)
        initial_temp (float): Initial temperature (default: 5230.0)
        restart_temp_ratio (float): Temperature restart ratio (default: 2e-5)
        visit (float): Parameter for visiting distribution (default: 2.62)
        accept (float): Parameter for acceptance distribution (default: -5.0)
    """
    
    def __init__(self, objective_fn=None, rng=None, **kwargs):
        super().__init__(objective_fn=objective_fn, rng=rng)
        self.minimize_kwargs = {
            'maxiter': 25,
            'initial_temp': 5230.0,
            'restart_temp_ratio': 2e-5,
            'visit': 2.62,
            'accept': -5.0,
            'seed': None,
        }
        if kwargs:
            self.minimize_kwargs.update(kwargs)
        
    def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
        """Execute one optimization step using dual annealing."""
        # Get bounds for this step
        lower_bounds, upper_bounds = self._get_bounds(current_x, step_limit)
        bounds = list(zip(lower_bounds, upper_bounds))
        
        # Run optimization
        result = optimize.dual_annealing(
            self._objective_to_minimize,
            bounds=bounds,
            x0=current_x,
            args=(neighborhood_state,),
            **self.minimize_kwargs
        )
        
        # Return best solution found
        new_x = result.x
        new_fx = -result.fun
        return new_x, new_fx

class NelderMeadOptimizer(BaseOptimizer):
    """Wrapper for scipy's Nelder-Mead implementation.
    
    Parameters that can be set via optimizer_opts:
        maxiter (int): Maximum number of iterations (default: 100)
        xatol (float): Absolute error in x between iterations (default: 1e-4)
        fatol (float): Absolute error in f(x) between iterations (default: 1e-4)
        adaptive (bool): Use adaptive parameters (default: True)
    """
    
    def __init__(self, objective_fn=None, rng=None, **kwargs):
        super().__init__(objective_fn=objective_fn, rng=rng)
        self.minimize_kwargs = {
            'options': {
                'maxiter': 100,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        }
        if kwargs:
            self.minimize_kwargs['options'].update(kwargs)
    
    def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
        """Execute one optimization step using Nelder-Mead."""
        # Get bounds for this step
        lower_bounds, upper_bounds = self._get_bounds(current_x, step_limit)
        bounds = optimize.Bounds(lower_bounds, upper_bounds)
        
        # Run optimization
        result = optimize.minimize(
            self._objective_to_minimize,
            current_x,
            method='Nelder-Mead',
            bounds=bounds,
            args=(neighborhood_state,),
            **self.minimize_kwargs
        )
        
        # Return best solution found
        new_x = result.x
        new_fx = -result.fun
        return new_x, new_fx

class LBFGSBOptimizer(BaseOptimizer):
    """Wrapper for scipy's L-BFGS-B implementation.
    
    Parameters that can be set via optimizer_opts:
        maxiter (int): Maximum number of iterations (default: 100)
        ftol (float): Function tolerance stopping criterion (default: 1e-6)
        gtol (float): Gradient tolerance stopping criterion (default: 1e-6)
        maxls (int): Maximum number of line search steps (default: 20)
    """
    
    def __init__(self, objective_fn=None, rng=None, **kwargs):
        super().__init__(objective_fn=objective_fn, rng=rng)
        self.minimize_kwargs = {
            'options': {
                'maxiter': 100,
                'ftol': 1e-6,
                'gtol': 1e-6,
                'maxls': 20,
            }
        }
        if kwargs:
            self.minimize_kwargs['options'].update(kwargs)
    
    def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
        """Execute one optimization step using L-BFGS-B."""
        # Get bounds for this step
        lower_bounds, upper_bounds = self._get_bounds(current_x, step_limit)
        bounds = optimize.Bounds(lower_bounds, upper_bounds)
            
        def gradient(x, *args):
            eps = np.sqrt(np.finfo(float).eps)
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (
                    self._objective_to_minimize(x_plus, neighborhood_state) - 
                    self._objective_to_minimize(x_minus, neighborhood_state)
                    ) / (2 * eps)
            return grad
        
        # Run optimization
        result = optimize.minimize(
            self._objective_to_minimize,
            current_x,
            jac=gradient,
            method='L-BFGS-B',
            bounds=bounds,
            args=(neighborhood_state,),  # Added to pass neighborhood state
            **self.minimize_kwargs
        )
        
        # Return best solution found
        new_x = result.x
        new_fx = -result.fun
        return new_x, new_fx
