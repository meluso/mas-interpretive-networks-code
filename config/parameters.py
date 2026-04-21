# config/parameters.py
"""Parameter space management for AI teams simulation."""

from typing import Any, Dict, List, Optional
from itertools import product

from config.defaults import (
    fn2opts,
    graph2opts,
    weight,
    weight_and_exponent,
    weight_and_frequency
)


class Parameters:
    """Manages parameter space for simulation."""
    
    
    def __init__(self):
        """Initialize the base parameter space."""
        
        # Team parameters
        self.team_size = [4, 8, 16]
        self.team_graph2opts = graph2opts
        
        # Agent parameters
        self.agent_steplim = [0.001, 0.01, 0.1, 1.0]
        self.agent_optim_type = [
            # 'random_walk',
            # 'dual_annealing',
            'nelder_mead',
            # 'lbfgsb'
            ]
        
        # Function parameters
        self.fn_type2opts = fn2opts
        
        # Campaign parameters
        self.num_trials = 250
        self.num_steps = 100
    
    
    def get_dict(self) -> Dict[str, any]:
        """Get parameters as a dictionary."""
        
        return {
            'num_trial': list(range(self.num_trials)),
            'team_size': self.team_size,
            'team_graph2opts': self.team_graph2opts,
            'agent_steplim': self.agent_steplim,
            'agent_optim_type': self.agent_optim_type,
            'fn_type2opts': self.fn_type2opts,
            'num_steps': self.num_steps
            }
        
        
    def get_combinations(self,
                        partition_id: Optional[int] = None,
                        total_partitions: Optional[int] = None) -> List[Dict]:
        """Get parameter combinations for execution.
        
        Args:
            subset: Optional parameter subset in either format
            partition_id: Optional partition index
            total_partitions: Optional total number of partitions
            
        Returns:
            List of parameter dictionaries for execution
        """
        
        # Create a dictionary defining the parameter space
        parameters = self.get_dict()
        
        # Generate all combinations from the dictionary
        combinations = self._generate_combinations(parameters)
        
        # Handle partitioning if requested
        if partition_id is not None and total_partitions is not None:
            return self._get_partition(combinations, partition_id, total_partitions)
        elif partition_id is not None or total_partitions is not None:
            assert partition_id is None, \
                f"partition_id is {partition_id} but total_partitions is unspecified"
            assert total_partitions is None, \
                f"total_partitions is {total_partitions} but partition_id is unspecified"
        
        return combinations

    def _get_partition(self, combinations: List[Dict],
                      partition_id: int, total_partitions: int) -> List[Dict]:
        """Get subset of combinations for specified partition.
        
        Args:
            combinations: Full list of parameter combinations
            partition_id: Index of desired partition
            total_partitions: Total number of partitions
            
        Returns:
            List of parameter combinations for partition
        """
        n = len(combinations)
        partition_size = n // total_partitions
        start = partition_id * partition_size
        end = start + partition_size if partition_id < total_partitions - 1 else n
        return combinations[start:end]

    def _generate_combinations(self,input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate all possible combinations of values from input dictionary,
        handling None values and nested dictionaries appropriately.
        
        Args:
            input_dict: Dictionary where values can be:
                - Lists of options
                - None
                - Nested dictionaries to be treated as combinatorial units
            
        Returns:
            List of dictionaries, each containing one combination of values
        """
        # Process input dictionary to handle different value types
        processed_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # Generate all combinations for nested dictionary
                processed_dict[key] = self._generate_nested_combinations(value)
            elif value is None:
                # Convert None to single-item list
                processed_dict[key] = [value]
            elif isinstance(value, list):
                # Keep lists as is
                processed_dict[key] = value
            else:
                # Convert single values to single-item list
                processed_dict[key] = [value]
        
        # Get the keys and processed values lists
        keys = list(processed_dict.keys())
        value_lists = [processed_dict[key] for key in keys]
        
        # Generate all combinations using itertools.product
        combinations = list(product(*value_lists))
        
        # Convert tuples to dictionaries
        result = []
        for combo in combinations:
            combo_dict = {keys[i]: combo[i] for i in range(len(keys))}
            result.append(combo_dict)
            
        return result
    
    def _generate_nested_combinations(self, nested_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate all possible combinations from a nested dictionary structure,
        treating None as a valid value option.
        
        Args:
            nested_dict: Dictionary with potentially nested values and lists
            
        Returns:
            List of dictionaries, each representing one complete combination
            
        Example:
            >>> nested = {
            ...     'complete': None,
            ...     'power': {'m': [2], 'p': [0.3, 0.7]}
            ... }
            >>> generate_nested_combinations(nested)
            [
                {'complete': None},
                {'power': {'m': 2, 'p': 0.3}},
                {'power': {'m': 2, 'p': 0.7}}
            ]
        """
        result = []
        
        for key, value in nested_dict.items():
            if value is None:
                # Create a combination with just this None value
                result.append({key: None})
                
            elif isinstance(value, dict):
                # For nested dictionaries with their own parameter combinations
                sub_combinations = []
                
                # Get all parameter names and their possible values
                param_names = list(value.keys())
                param_values = []
                for param in param_names:
                    param_val = value[param]
                    if isinstance(param_val, list):
                        param_values.append(param_val)
                    else:
                        param_values.append([param_val])
                
                # Generate all combinations of parameter values
                for combo in product(*param_values):
                    param_dict = {param_names[i]: combo[i] for i in range(len(param_names))}
                    sub_combinations.append({key: param_dict})
                
                if not result:
                    result = sub_combinations
                else:
                    # We're now treating each top-level key as mutually exclusive,
                    # so we just extend the results instead of combining
                    result.extend(sub_combinations)
                    
            elif isinstance(value, list):
                # For direct list values
                sub_combinations = [{key: val} for val in value]
                if not result:
                    result = sub_combinations
                else:
                    result.extend(sub_combinations)
                    
            else:
                # For single values
                if not result:
                    result = [{key: value}]
                else:
                    result.append({key: value})
        
        return result
    
    
class ParametersSize01(Parameters):
    """Special case of parameters for teams of size=1 only."""
    
    def __init__(self):
        super().__init__()
        
        # Set team size
        self.team_size = [1]
        
        # Set allowed graph type
        self.team_graph2opts = {
            'empty': None
            }
    
        
class ParametersSize02(Parameters):
    """Special case of parameters for teams of size=2 only."""
    
    def __init__(self):
        super().__init__()
        
        # Set team size
        self.team_size = [2]
        
        # Set allowed graph type
        self.team_graph2opts = {
            'complete': None,
            'empty': None
            }