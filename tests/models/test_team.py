# tests/models/test_team.py
"""
Test suite for Team class.

Tests verify Team functionality including:
- Network creation and properties
- Variable distribution
- Agent initialization and management 
- State management and stepping
- Metric calculations
- Monte Carlo sampling
"""

import pytest
import numpy as np
import networkx as nx
from models.team import Team, get_objective

@pytest.fixture
def base_team_params():
    """Return base parameters for Team initialization."""
    return {
        'team_size': 4,
        'agent_num_vars': 2,
        'team_graph_type': 'complete',
        'team_graph_opts': None,
        'agent_optim_type': 'random_walk',
        'agent_optim_opts': None,
        'agent_steplim': 0.1,
        'fn_type': 'sphere',
        'fn_opts': {'weight': 'node'}
    }

@pytest.fixture
def objective_combinations():
    """Valid objective function combinations."""
    return [
        ('average', {'weight': 'node'}),
        ('average', {'weight': 'degree'}),
        ('sphere', {'weight': 'node'}),
        ('sphere', {'weight': 'degree'}),
        ('root', {'weight': 'node'}),
        ('root', {'weight': 'degree'}),
        ('sin2', {'weight': 'node', 'frequency': 'uniform'}),
        ('sin2', {'weight': 'node', 'frequency': 'degree'}),
        ('sin2', {'weight': 'degree', 'frequency': 'uniform'}),
        ('sin2', {'weight': 'degree', 'frequency': 'degree'}),
        ('sin2sphere', {'weight': 'node', 'frequency': 'uniform'}),
        ('sin2sphere', {'weight': 'node', 'frequency': 'degree'}),
        ('sin2sphere', {'weight': 'degree', 'frequency': 'uniform'}),
        ('sin2sphere', {'weight': 'degree', 'frequency': 'degree'}),
        ('sin2root', {'weight': 'node', 'frequency': 'uniform'}),
        ('sin2root', {'weight': 'node', 'frequency': 'degree'}),
        ('sin2root', {'weight': 'degree', 'frequency': 'uniform'}),
        ('sin2root', {'weight': 'degree', 'frequency': 'degree'}),
        ('losqr_hiroot', {'weight': 'node', 'exponent': 'degree'}),
        ('losqr_hiroot', {'weight': 'degree', 'exponent': 'degree'}),
        ('hisqr_loroot', {'weight': 'node', 'exponent': 'degree'}),
        ('hisqr_loroot', {'weight': 'degree', 'exponent': 'degree'}),
        ('max', {}),
        ('min', {}),
        ('median', {}),
        ('kth_power', {'weight': 'node'}),
        ('kth_power', {'weight': 'degree'}),
        ('kth_root', {'weight': 'node'}),
        ('kth_root', {'weight': 'degree'}),
        ('ackley', {})
    ]

@pytest.fixture
def graph_combinations():
    """Valid graph type combinations."""
    return [
        ('complete', {}),
        ('empty', {}),
        ('power', {'m': 2, 'p': 0.3}),
        ('random', {'p': 0.3}),
        ('ring_cliques', {}),
        ('rook', {}),
        ('small_world', {'k': 2, 'p': 0.1}),
        ('star', {}),
        ('tree', {}),
        ('wheel', {}),
        ('windmill', {})
    ]

@pytest.fixture
def samples(base_team_params):
    """Generate Monte Carlo samples for testing distribution properties"""
    team = Team(**base_team_params)
    
    # Generate samples and get team objective values
    team.gen_mc_samples()
    return np.array([team.eval_team_fn(x) for x in team.mc_samples])

def test_team_initialization(base_team_params):
    """Test team initialization including:
    - Basic graph properties (network functionality)
    - Variable distribution (task division behavior)
    - Agent initialization (optimization capability)
    - State initialization (readiness to optimize)
    - Objective function setup (ability to evaluate solutions)
    - Network metrics (analysis capability)
    """
    team = Team(**base_team_params)
    
    # 1. Test basic graph behavior
    assert isinstance(team, nx.Graph)  # Maintains graph functionality
    assert len(team) == base_team_params['team_size']  # Correct team size
    assert team.agent_num_vars == base_team_params['agent_num_vars'] # Correct vars per agent
    
    # 2. Test agent behavior
    for ag in team:
        # Check agent exists and is properly initialized
        assert 'agent' in team.nodes[ag]
        agent = team.nodes[ag]['agent']
        
        # Check state initialization and bounds
        x = agent.get_x()
        assert isinstance(x, np.ndarray)  # Proper state type
        assert len(x) == team.agent_num_vars  # Correct number of variables
        assert np.all((x >= 0) & (x <= 1))  # Valid state bounds
        
        # Check optimization capability
        assert agent.get_fx() is not None  # Has objective value
        assert 0 <= agent.get_fx() <= 1  # Valid objective bounds
        
        # Test optimizer is properly configured
        assert agent.optimizer is not None
        assert agent.step_limit == base_team_params['agent_steplim']
        
        # Test proper vector handling
        nbhd_state = team.get_nbhd_states()[ag]
        new_x = agent.step(nbhd_state)
        assert isinstance(new_x, np.ndarray)  # Valid return type
        assert len(new_x) == team.agent_num_vars  # Maintains dimensionality
        assert np.all((new_x >= 0) & (new_x <= 1))  # Valid bounds
        
        # Test step limit enforcement
        component_changes = np.abs(new_x - x)
        assert np.all(component_changes <= agent.step_limit + 1e-10)
    
    # 3. Test team state initialization
    assert not(np.isnan(team.get_team_fx())) # This value has been initialized
    assert np.isnan(team.get_team_dfdt())  # Ready but not yet evaluated
    
    # 4. Test objective function behavior
    test_x = np.random.random(team.agent_num_vars * team.team_size)
    result = team.eval_team_fn(test_x)
    assert isinstance(result, (float, np.floating))  # Proper output type
    assert 0 <= result <= 1  # Valid objective bounds
    
    # 5. Test metric calculation capability
    assert hasattr(team, 'network_metrics')
    assert len(team.network_metrics) > 0  # Has some metrics
    
    # Basic network analysis capabilities
    assert any('assortativity' in key for key in team.network_metrics)  # Has assortativity measure
    assert any('clustering' in key for key in team.network_metrics)     # Has clustering measure
    assert any('density' in key for key in team.network_metrics)        # Has density measure
    assert any('path' in key for key in team.network_metrics)          # Has path length measure
    assert any('diameter' in key for key in team.network_metrics)      # Has diameter measure
    
    # Verify metric behaviors (these are the important parts)
    # Get the actual keys that contain our metrics of interest
    assortativity_key = next(k for k in team.network_metrics if 'assortativity' in k)
    clustering_key = next(k for k in team.network_metrics if 'clustering' in k)
    density_key = next(k for k in team.network_metrics if 'density' in k)
    path_key = next(k for k in team.network_metrics if 'path' in k)
    diameter_key = next(k for k in team.network_metrics if 'diameter' in k)
    
    # Test metric behaviors
    assert -1 <= team.network_metrics[assortativity_key] <= 1  # Valid correlation
    assert 0 <= team.network_metrics[clustering_key] <= 1      # Valid probability
    assert 0 <= team.network_metrics[density_key] <= 1         # Valid ratio
    
    # Path metrics may be None for disconnected graphs
    if team.network_metrics[path_key] is not None:
        assert team.network_metrics[path_key] >= 1
    if team.network_metrics[diameter_key] is not None:
        assert team.network_metrics[diameter_key] >= 1

def test_network_metrics(base_team_params):
    """Test network metric calculations."""
    team = Team(**base_team_params)
    
    # Test metric existence
    required_metrics = [
        'team_graph_centrality_degree_mean',
        'team_graph_centrality_degree_stdev',
        'team_graph_centrality_betweenness_mean',
        'team_graph_centrality_betweenness_stdev',
        'team_graph_centrality_eigenvector_mean',
        'team_graph_centrality_eigenvector_stdev',
        'team_graph_nearest_neighbor_degree_mean',
        'team_graph_nearest_neighbor_degree_stdev',
        'team_graph_clustering',
        'team_graph_density',
        'team_graph_assortativity',
        'team_graph_pathlength',
        'team_graph_diameter'
    ]
    for metric in required_metrics:
        assert metric in team.network_metrics

def test_monte_carlo_sampling():
    """Test Monte Carlo sampling behaviors:
    1. Proper sampling of decision space
    2. Valid objective evaluation on samples
    3. Neighborhood consistency
    4. Proper handling of different objective landscapes
    """
    
    # Test with small team/var count for clearer validation
    base_params = {
        'team_size': 2,  # Minimal team size
        'agent_num_vars': 2,  # 2 vars per agent
        'team_graph_type': 'complete',
        'team_graph_opts': {},
        'agent_steplim': 0.1,
        'agent_optim_type': 'random_walk',
    }

    # Test three different landscapes: linear, periodic, and radial
    landscapes = [
        ('average', {'weight': 'node'}),  # Linear landscape
        ('sin2', {'weight': 'node', 'frequency': 'uniform'}),  # Periodic
        ('sphere', {'weight': 'node'})  # Radial
    ]
    
    for fn_type, fn_opts in landscapes:
        params = base_params.copy()
        params.update({
            'fn_type': fn_type,
            'fn_opts': fn_opts
        })
        team = Team(**params)
        
        # 1. Test sampling behavior
        team.gen_mc_samples()
        assert isinstance(team.mc_samples, np.ndarray)  # Proper return type
        assert team.mc_samples.shape == (team.mc_trials, team.team_size * team.agent_num_vars)  # Correct dimensionality
        assert np.all((team.mc_samples >= 0) & (team.mc_samples <= 1))  # Valid bounds
        
        # 2. Test sample distribution properties
        for dim in range(team.mc_samples.shape[1]):
            dim_samples = team.mc_samples[:, dim]
            # Reasonable coverage of space
            assert 0.1 < np.mean(dim_samples) < 0.9
            assert np.std(dim_samples) > 0.2
            
        # 3. Test neighborhood consistency
        for ag in team:
            nbhd_fx = team.get_nbhd_mc_fx(ag)
            assert isinstance(nbhd_fx, np.ndarray)  # Proper return type
            assert len(nbhd_fx) == team.mc_trials  # Correct number of samples
            assert np.all((nbhd_fx >= 0) & (nbhd_fx <= 1))  # Valid bounds
            
            # Verify neighborhood evaluations match direct calculation
            nbhd_indices = team.get_nbhd_var_indices(ag)
            agent = team.nodes[ag]['agent']
            for i in range(min(5, team.mc_trials)):  # Check first few samples
                direct_eval = agent.objective_fn(team.mc_samples[i, nbhd_indices])
                assert np.isclose(direct_eval, nbhd_fx[i])
        
        # 4. Test landscape-specific behaviors
        team_fx = np.array([team.eval_team_fn(x) for x in team.mc_samples])
        assert np.all((team_fx >= 0) & (team_fx <= 1))  # Valid bounds
        
        if fn_type == 'average':
            # For average function with n uniform random variables:
            # - Mean should be ~0.5 (center of [0,1] interval) 
            # - Std dev should be ~1/(√12*√n) ≈ 0.144 for n=4 variables
            assert 0.4 < np.mean(team_fx) < 0.6, "Mean should be close to center"
            assert 0.12 < np.std(team_fx) < 0.17, \
                f"Std dev {np.std(team_fx)} outside expected range"
            
        elif fn_type == 'sin2':
            # Periodic landscape should have multiple peaks
            peaks = []
            for i in range(1, len(team_fx)-1):
                if team_fx[i] > team_fx[i-1] and team_fx[i] > team_fx[i+1]:
                    peaks.append(team_fx[i])
            assert len(peaks) > 1  # Multiple local maxima
            
        elif fn_type == 'sphere':
            # Radial landscape should increase with distance from origin
            distances = np.linalg.norm(team.mc_samples, axis=1)  # Distance from origin
            # Positive correlation between distance and objective
            corr = np.corrcoef(distances, team_fx)[0,1]
            assert corr > 0.5

def test_uniformity(samples, bins=5, max_ratio=3.5, min_bin_coverage=0.7):
    """Test uniformity of samples using ratio of maximum to mean bin counts."""
    assert samples is not None and len(samples) > 0
        
    hist, _ = np.histogram(samples, bins=bins)
    
    # Filter out empty bins
    nonempty_bins = hist[hist > 0]
    
    # Check bin coverage
    assert len(nonempty_bins) >= bins * min_bin_coverage, \
        f"Only {len(nonempty_bins)} bins have samples (need {bins * min_bin_coverage})"
    
    # Check ratio of maximum to mean of nonempty bins
    max_count = np.max(nonempty_bins)
    mean_count = np.mean(nonempty_bins)
    
    assert (max_count / mean_count) < max_ratio, \
        f"Max/mean ratio {max_count/mean_count} exceeds limit {max_ratio}"

def test_team_stepping(base_team_params):
    """Test team stepping mechanism including state changes, bound enforcement,
    and metric updates."""
    team = Team(**base_team_params)
    
    # 1. Test initial conditions
    assert not(np.isnan(team.get_team_fx()))
    assert np.isnan(team.get_team_dfdt())
    
    # Store initial states
    initial_states = {
        ag: team.nodes[ag]['agent'].get_x().copy()
        for ag in team
    }
    
    # 2. Execute multiple steps
    n_steps = 5
    performance_history = []
    productivity_history = []
    print(productivity_history)
    any_agent_changed = False  # Track if any agent changed in any step
    
    for step in range(n_steps + 1):
        # Store pre-step states
        old_states = {
            ag: team.nodes[ag]['agent'].get_x().copy()
            for ag in team
        }
        
        # Execute step
        if step > 0:
            team.step()
        
        # Store metrics
        performance_history.append(team.get_team_fx())
        productivity_history.append(team.get_team_dfdt())
        print(team.get_team_dfdt())
        
        # Verify all agents maintain valid states
        for ag in team:
            agent = team.nodes[ag]['agent']
            current_state = agent.get_x()
            
            # Check bounds
            assert np.all((current_state >= 0) & (current_state <= 1))
            
            # Verify step limit enforcement - check component-wise changes
            component_changes = np.abs(current_state - old_states[ag])
            assert np.all(component_changes <= agent.step_limit + 1e-10)
            
            # Track if this agent has changed state
            if not np.array_equal(current_state, initial_states[ag]):
                any_agent_changed = True
    
    # Verify at least one agent changed state during optimization
    assert any_agent_changed, "No agents changed state during optimization"
    
    # 3. Test metric updates
    assert not np.any(np.isnan(performance_history))  # All performance values set
    assert not np.any(np.isnan(productivity_history[1:]))  # All productivity values after first step are set
    print(productivity_history)
    assert np.isnan(productivity_history[0])  # First productivity should be NaN
    
    # Verify productivity calculation
    for i in range(1, len(performance_history)):
        expected_dfdt = (performance_history[i] - performance_history[i-1]) / team.team_size
        assert np.abs(productivity_history[i] - expected_dfdt) < 1e-10
    
    # 4. Test state consistency
    # Get all current states
    final_states = {
        ag: team.nodes[ag]['agent'].get_x().copy() 
        for ag in team
    }
    
    # Reconstruct full state vector
    state_vector = np.concatenate([
        final_states[ag] 
        for ag in sorted(final_states.keys())
    ])
    
    # Verify team objective matches state
    direct_eval = team.eval_team_fn(state_vector)
    assert np.abs(direct_eval - team.get_team_fx()) < 1e-10
    
    # 5. Test neighborhood consistency
    for ag in team:
        agent = team.nodes[ag]['agent']
        nbhd_indices = team.get_nbhd_var_indices(ag)
        nbhd_state = state_vector[nbhd_indices]
        
        # Verify neighborhood objective matches agent's stored value
        nbhd_eval = agent.objective_fn(nbhd_state)
        assert np.abs(nbhd_eval - agent.get_fx()) < 1e-10

def test_objective_functions(objective_combinations, base_team_params):
    """Test integration of objective functions within Team context, focusing on:
    - Proper function creation and distribution to agents
    - Consistency between team and neighborhood objectives
    - Proper handling in optimization context
    """
    for fn_type, fn_opts in objective_combinations:
        params = base_team_params.copy()
        params.update({
            'fn_type': fn_type,
            'fn_opts': fn_opts
        })
        
        team = Team(**params)
        
        # 1. Test proper objective distribution
        for ag in team:
            agent = team.nodes[ag]['agent']
            assert agent.objective_fn is not None
            
            # Neighborhood objective should use same type and options
            nbhd_degrees = team.get_nbhd_ks(ag)
            test_fn = get_objective(fn_type, fn_opts, team.agent_num_vars, nbhd_degrees)
            
            # Test they handle same input identically
            test_x = np.random.random(len(nbhd_degrees) * team.agent_num_vars)
            assert np.isclose(agent.objective_fn(test_x), test_fn(test_x))
        
        # 2. Test consistency between team and neighborhood objectives
        test_x = np.random.random(team.agent_num_vars * team.team_size)
        team_result = team.eval_team_fn(test_x)
        
        for ag in team:
            # Get neighborhood variables and evaluate
            nbhd_indices = team.get_nbhd_var_indices(ag)
            nbhd_x = test_x[nbhd_indices]
            nbhd_result = team.nodes[ag]['agent'].objective_fn(nbhd_x)
            
            # Both should be valid outputs
            assert 0 <= nbhd_result <= 1
            assert 0 <= team_result <= 1
        
        # 3. Test optimization context
        initial_states = {
            ag: team.nodes[ag]['agent'].get_x().copy() 
            for ag in team
        }
        
        # Run a few optimization steps
        for _ in range(3):
            team.step()
            
            # Verify states remain valid
            for ag in team:
                agent = team.nodes[ag]['agent']
                current_state = agent.get_x()
                
                # States should be valid
                assert np.all((current_state >= 0) & (current_state <= 1))
                
                # Objectives should be valid
                assert 0 <= agent.get_fx() <= 1
        
            # Team objective should be valid
            assert 0 <= team.get_team_fx() <= 1

def test_neighborhood_consistency(base_team_params):
    """Test consistency of neighborhood calculations including variable indexing,
    state management, and objective evaluation across different network structures."""
    
    # Test with different network types
    network_cases = [
        ('complete', None),  # Fully connected
        ('star', None),      # Central hub with spokes
        ('ring_cliques', None)  # Local clusters
    ]
    
    for graph_type, graph_opts in network_cases:
        params = base_team_params.copy()
        params.update({
            'team_graph_type': graph_type,
            'team_graph_opts': graph_opts
        })
        
        team = Team(**params)
        
        # 1. Test Variable Index Consistency
        for ag in team:
            nbhd_indices = team.get_nbhd_var_indices(ag)
            
            # Get expected variable ranges for agent and neighbors
            expected_indices = []
            
            # Add agent's own variables
            var_start = sum(team.agent_num_vars for i in range(ag))
            var_end = var_start + team.agent_num_vars
            expected_indices.extend(range(var_start, var_end))
            
            # Add neighbors' variables
            for nbr in team.neighbors(ag):
                nbr_start = sum(team.agent_num_vars for i in range(nbr))
                nbr_end = nbr_start + team.agent_num_vars
                expected_indices.extend(range(nbr_start, nbr_end))
            
            # Verify all expected indices are present
            assert set(nbhd_indices) == set(expected_indices), \
                f"Incorrect neighborhood indices for agent {ag} in {graph_type} network"
            
            # Verify indices are within bounds
            assert max(nbhd_indices) < team.agent_num_vars * team.team_size, \
                f"Index out of bounds for agent {ag}"
            assert min(nbhd_indices) >= 0, \
                f"Negative index for agent {ag}"
        
        # 2. Test State Consistency
        # Create test state vector
        test_state = np.random.random(team.agent_num_vars * team.team_size)
        
        # Update all agent states
        for ag in team:
            agent = team.nodes[ag]['agent']
            var_start = sum(team.agent_num_vars for i in range(ag))
            var_end = var_start + team.agent_num_vars
            agent._x = test_state[var_start:var_end].copy()
        
        # Test neighborhood state reconstruction
        for ag in team:
            nbhd_indices = team.get_nbhd_var_indices(ag)
            
            # Get actual states from neighbors
            nbhd_state = []
            # Agent's own state
            var_start = sum(team.agent_num_vars for i in range(ag))
            var_end = var_start + team.agent_num_vars
            nbhd_state.extend(test_state[var_start:var_end])
            
            # Neighbors' states
            for nbr in team.neighbors(ag):
                nbr_start = sum(team.agent_num_vars for i in range(nbr))
                nbr_end = nbr_start + team.agent_num_vars
                nbhd_state.extend(test_state[nbr_start:nbr_end])
            
            # Verify neighborhood state matches expected
            assert np.array_equal(nbhd_state, test_state[nbhd_indices]), \
                f"Incorrect neighborhood state for agent {ag}"
        
        # 3. Test Objective Evaluation Consistency
        for ag in team:
            agent = team.nodes[ag]['agent']
            nbhd_indices = team.get_nbhd_var_indices(ag)
            
            # Get full neighborhood state
            nbhd_state = test_state[nbhd_indices]
            
            # Test direct evaluation
            direct_result = agent.objective_fn(nbhd_state)
            
            # Test through agent interface
            agent_states = []
            # Add agent's state
            var_start = sum(team.agent_num_vars for i in range(ag))
            var_end = var_start + team.agent_num_vars
            agent_states.append(test_state[var_start:var_end])
            
            # Add neighbors' states
            for nbr in team.neighbors(ag):
                nbr_start = sum(team.agent_num_vars for i in range(nbr))
                nbr_end = nbr_start + team.agent_num_vars
                agent_states.append(test_state[nbr_start:nbr_end])
            
            # Update agent's objective value
            agent.set_fx(agent_states)
            interface_result = agent.get_fx()
            
            # Results should match
            assert np.isclose(direct_result, interface_result), \
                f"Inconsistent objective evaluation for agent {ag}"
            
            # Both should be valid outputs
            assert 0 <= direct_result <= 1
            assert 0 <= interface_result <= 1

def test_step_limits(base_team_params):
    """Test step limit enforcement in vector-based optimization.
    Tests:
    - Component-wise step size enforcement
    - Proper handling of different step limits
    - Boundary condition handling
    """
    
    # Test different step limit values
    for step_limit in [0.1, 0.5, 1.0]:
        params = base_team_params.copy()
        params['agent_steplim'] = step_limit
        team = Team(**params)
        
        # Test multiple steps for consistency
        for _ in range(5):
            # Store old states
            old_states = {
                ag: team.nodes[ag]['agent'].get_x().copy() 
                for ag in team
            }
            
            team.step()
            
            # Check step sizes for each agent
            for ag in team:
                agent = team.nodes[ag]['agent']
                new_state = agent.get_x()
                old_state = old_states[ag]
                
                # Test key step limit properties
                if step_limit < 1:
                    # Component-wise changes should not exceed limit
                    component_changes = np.abs(new_state - old_state)
                    assert np.all(component_changes <= step_limit + 1e-10), \
                        f"Component change {component_changes.max()} exceeds limit {step_limit}"
                
                # Test boundary handling
                assert np.all((new_state >= 0) & (new_state <= 1)), \
                    f"State {new_state} outside [0,1] bounds"
                
                # Special test for boundary behavior
                if np.any(old_state < step_limit) or np.any(old_state > 1 - step_limit):
                    # States near boundaries should still respect bounds
                    assert np.all((new_state >= 0) & (new_state <= 1)), \
                        f"Boundary violation: {new_state}"

    # Test extreme cases
    extreme_cases = [
        ('small_limit', 0.001),   # Very small steps
        ('no_limit', 2.0),       # Effectively no limit
        ('zero_limit', 0.0)      # Zero step size
    ]
    
    for case_name, step_limit in extreme_cases:
        params = base_team_params.copy()
        params['agent_steplim'] = step_limit
        team = Team(**params)
        
        # Store initial states
        initial_states = {
            ag: team.nodes[ag]['agent'].get_x().copy() 
            for ag in team
        }
        
        team.step()
        
        for ag in team:
            agent = team.nodes[ag]['agent']
            new_state = agent.get_x()
            
            if case_name == 'small_limit':
                # Should allow very small changes
                changes = np.abs(new_state - initial_states[ag])
                assert np.all(changes <= step_limit + 1e-10)
                
            elif case_name == 'no_limit':
                # Should still maintain bounds
                assert np.all((new_state >= 0) & (new_state <= 1))
                
            elif case_name == 'zero_limit':
                # Should prevent any changes
                np.testing.assert_array_equal(new_state, initial_states[ag])

if __name__ == '__main__':
    pytest.main([__file__])
