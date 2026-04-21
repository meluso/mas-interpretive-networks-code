# models/objectives/options.py
"""
Created on Thu Nov 11 19:38:08 2021

@author: John Meluso
"""

# Import libraries
from numpy import dot, float64

def ensure_scalar(val, operation='dot product'):
    """Ensure return value is a scalar for objective function results.
    Raises informative error if aggregation failed."""
    if hasattr(val, 'ndim') and val.ndim > 0:
        if val.size == 1:
            return float64(val.item())
        raise ValueError(f"Failed to aggregate {operation} result to scalar:" \
                         " got array of size {val.size}")
    return float64(val)


class Weight(object):
    def __init__(self, nn, kk, divisor):
        '''Initialize function class that normalizes an input by the node
        count and a range equivalent to the divisor input.'''
        self.nn = nn
        self.kk = kk
        self.divisor = divisor


class NodeWeight(Weight):
    def __init__(self, nn, kk, divisor=1):
        super().__init__(nn, kk, divisor)
    
    def __call__(self, xx):
        """Return normalized result weighted equally.
        
        Args:
            xx: Vector of variables
            nn: Total number of variables 
            kk: Not used but kept for interface consistency
        """
        return ensure_scalar(sum(xx)/(self.divisor * self.nn))
    

class DegreeWeight(Weight):
    def __init__(self, nn, kk, divisor=1):
        super().__init__(nn, kk, divisor)
    
    def __call__(self, xx):
        """Return normalized result weighted by node degrees.
        
        Args:
            xx: Vector of variables (e.g. [x11,x12, x21,x22, x31,x32, x41,x42])
            nn: Total number of variables
            kk: Vector of node degrees (e.g. [k1, k2, k3, k4])
            
        Returns:
            float: Normalized weighted sum
        """
        # Calculate weights
        weights = self.kk + 1
        # print(f"Xs: {xx}")
        # print(f"Weights: {weights}")
        
        # Calculate sum of weights for normalization
        total_weight = sum(self.kk + 1)
        
        # Weighted sum normalized by total possible weight
        return ensure_scalar(dot(weights, xx)/(self.divisor * total_weight))
