# config/factory_implementations.py
"""Factory implementations for parameter set variants.

This module contains concrete implementations of ParametersFactory for different
parameter configurations. Each factory class is decorated with @register_parameters
to automatically register itself with ParametersRegistry.

To add a new parameter set:
1. Create a new class inheriting from ParametersFactory
2. Decorate with @register_parameters("your_variant_name") 
3. Override create_parameters() to return customized Parameters instance
"""

from config.defaults import weight, weight_and_exponent, weight_and_frequency
from config.parameters_factory import ParametersFactory
from config.parameters import Parameters, ParametersSize01, ParametersSize02
from config.registry import register_parameters

@register_parameters("default")
class DefaultParametersFactory(ParametersFactory):
    """ Default factory with all parameters set to default values."""
    def create_parameters(self) -> Parameters:
        return Parameters()
    
#%% Small Test Parameter Sets

@register_parameters("campaignTestDefault")
class TestDefaultParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = Parameters()
        params.team_graph2opts = {
            'complete': None,
            'small_world': {'k': [2, 4], 'p': [0.3, 0.7]}
            }
        params.agent_steplim = [0.001, 1.0]
        params.fn_type2opts = {
            'root': weight,
            'sin2sphere': weight_and_frequency,
            'ackley': None            
            }
        params.num_trials = 3
        params.num_steps = 25
        return params
    
@register_parameters("campaignTestSize01")
class TestSize01ParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize01()
        params.agent_steplim = [0.001, 1.0]
        params.fn_type2opts = {
            'root': weight,
            'sin2sphere': weight_and_frequency,
            'ackley': None            
            }
        params.num_trials = 3
        params.num_steps = 25
        return params
    
@register_parameters("campaignTestSize02")
class TestSize02ParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.agent_steplim = [0.001, 1.0]
        params.fn_type2opts = {
            'root': weight,
            'sin2sphere': weight_and_frequency,
            'ackley': None            
            }
        params.num_trials = 3
        params.num_steps = 25
        return params
    
@register_parameters("campaignTestSpecific")
class TestSpecificParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.agent_steplim = [0.001, 0.01, 0.1, 1.0]
        params.fn_type2opts = {
            'sin2root': weight_and_frequency
            }
        params.num_trials = 3
        params.num_steps = 25
        return params
    
#%% Large Test (Rehearsal) Parameter Sets
        
@register_parameters("campaignRehearsalDefault")
class RehearsalDefaultParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = Parameters()
        params.num_trials = 2
        return params
    
@register_parameters("campaignRehearsalSize01")
class RehearsalSize01ParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize01()
        params.num_trials = 2
        return params
    
@register_parameters("campaignRehearsalSize02")
class RehearsalSize02ParametersFactory(ParametersFactory):
    """ Factory for campaign testing."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.num_trials = 2
        return params
    
#%% Large Test (Rehearsal) Parameter Sets with Dual Annealing

@register_parameters("campaignRehearsalSize01DualAnnealing")
class RehearsalSize01DualAnnealingParametersFactory(ParametersFactory):
    """Factory for teams of 1. Networks don't matter here so use complete."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize01()
        params.agent_optim_type = ['dual_annealing']
        params.num_trials = 2
        return params

@register_parameters("campaignRehearsalSize02DualAnnealing")
class RehearsalSize02DualAnnealingParametersFactory(ParametersFactory):
    """Factory for teams of 2. Networks don't matter here so use complete."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.agent_optim_type = ['dual_annealing']
        params.num_trials = 2
        return params

@register_parameters("campaignRehearsalDefaultDualAnnealing")
class RehearsalDefaultDualAnnealingParametersFactory(ParametersFactory):
    """Factory for teams of 4, 8, and 16."""
    def create_parameters(self) -> Parameters:
        params = Parameters()
        params.agent_optim_type = ['dual_annealing']
        params.num_trials = 2 
        return params

#%% Study 01 Default (Nelder-Mead) Parameter Sets

@register_parameters("campaignAITeams01Size01")
class Campaign001ParametersFactory(ParametersFactory):
    """ Factory for teams of 1. Networks don't matter here so use complete."""
    def create_parameters(self) -> Parameters:
        return ParametersSize01()

@register_parameters("campaignAITeams01Size02")
class Campaign002ParametersFactory(ParametersFactory):
    """ Factory for teams of 2. Networks don't matter here so use complete"""
    def create_parameters(self) -> Parameters:
        return ParametersSize02()

@register_parameters("campaignAITeams01Default")
class Campaign003ParametersFactory(ParametersFactory):
    """ Factory for teams of 4, 8, and 16."""
    def create_parameters(self) -> Parameters:
        return Parameters()

#%% Study 01 LBFGSB Parameter Sets

@register_parameters("campaignAITeams01Size01LBFGSB")
class Campaign004ParametersFactory(ParametersFactory):
    """ Factory for teams of 1. Networks don't matter here so use complete."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize01()
        params.agent_optim_type = ['lbfgsb']
        return params

@register_parameters("campaignAITeams01Size02LBFGSB")
class Campaign005ParametersFactory(ParametersFactory):
    """ Factory for teams of 2. Networks don't matter here so use complete"""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.agent_optim_type = ['lbfgsb']
        return params

@register_parameters("campaignAITeams01DefaultLBFGSB")
class Campaign006ParametersFactory(ParametersFactory):
    """ Factory for teams of 4, 8, and 16."""
    def create_parameters(self) -> Parameters:
        params = Parameters()
        params.agent_optim_type = ['lbfgsb']
        return params
    
#%% Study 01 Random Walk Parameter Sets

@register_parameters("campaignAITeams01Size01RandomWalk")
class Campaign007ParametersFactory(ParametersFactory):
    """ Factory for teams of 1. Networks don't matter here so use complete."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize01()
        params.agent_optim_type = ['random_walk']
        return params

@register_parameters("campaignAITeams01Size02RandomWalk")
class Campaign008ParametersFactory(ParametersFactory):
    """ Factory for teams of 2. Networks don't matter here so use complete"""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.agent_optim_type = ['random_walk']
        return params

@register_parameters("campaignAITeams01DefaultRandomWalk")
class Campaign009ParametersFactory(ParametersFactory):
    """ Factory for teams of 4, 8, and 16."""
    def create_parameters(self) -> Parameters:
        params = Parameters()
        params.agent_optim_type = ['random_walk']
        return params

#%% Study 01 Dual Annealing Parameter Sets

@register_parameters("campaignAITeams01Size01DualAnnealing")
class Campaign010ParametersFactory(ParametersFactory):
    """ Factory for teams of 1. Networks don't matter here so use complete."""
    def create_parameters(self) -> Parameters:
        params = ParametersSize01()
        params.agent_optim_type = ['dual_annealing']
        params.num_trials = 50
        return params

@register_parameters("campaignAITeams01Size02DualAnnealing")
class Campaign011ParametersFactory(ParametersFactory):
    """ Factory for teams of 2. Networks don't matter here so use complete"""
    def create_parameters(self) -> Parameters:
        params = ParametersSize02()
        params.agent_optim_type = ['dual_annealing']
        params.num_trials = 50
        return params

@register_parameters("campaignAITeams01DefaultDualAnnealing")
class Campaign012ParametersFactory(ParametersFactory):
    """ Factory for teams of 4, 8, and 16."""
    def create_parameters(self) -> Parameters:
        params = Parameters()
        params.agent_optim_type = ['dual_annealing']
        params.num_trials = 50
        return params


if __name__ == '__main__':
    
    # Test parameter generation
    from config import ParametersRegistry
    import numpy as np
    rng = np.random.default_rng()
    par_id = 2
    tot_par = 3
    
    # Main steps for creation
    factory = ParametersRegistry.get_factory("campaignTestDefault")
    params_object = factory.create_parameters()
    params = params_object.get_dict()
    combos = params_object.get_combinations()
    
    # Display outputs
    rand_combo = combos[rng.integers(len(combos))]
    print(f"Num. Combos: {len(combos)}")
    print(f"Ex. Combo: {rand_combo}")