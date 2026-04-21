# tests/models/test_agent.py
import pytest
import numpy as np
from numpy.random import default_rng
from models.agent import Agent

@pytest.fixture
def rng():
    """Provide consistent random number generator for tests."""
    return default_rng(seed=42)

@pytest.fixture
def base_agent_params():
    """Basic parameters for creating test agents."""
    return {
        'num_vars': 3,
        'optimizer_type': 'random_walk',
        'optimizer_opts': None,
        'step_limit': 0.1
    }

@pytest.fixture
def simple_objective():
    """Simple quadratic objective function for testing."""
    def objective(nbhd_state):
        x = nbhd_state[0]  # Agent's state is first in neighborhood
        return -np.sum((x - 0.5)**2)  # Maximum at x = 0.5
    return objective

class TestAgentInitialization:
    """Test agent initialization and basic properties."""
    
    def test_basic_initialization(self, base_agent_params, simple_objective, rng):
        """Test that agent initializes with basic parameters."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        
        assert len(agent.x) == base_agent_params['num_vars']
        assert agent.step_limit == base_agent_params['step_limit']
        assert agent.fx is None
        assert agent.track_trajectories is False
        assert agent.trajectories is None
    
    def test_initial_x_random(self, base_agent_params, simple_objective, rng):
        """Test that agent generates random initial position."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        
        assert agent.x.shape == (base_agent_params['num_vars'],)
        assert np.all(agent.x >= 0) and np.all(agent.x <= 1)
    
    def test_different_optimizers(self, simple_objective, rng):
        """Test initialization with different optimizer types."""
        optimizer_types = ['random_walk', 'dual_annealing', 'nelder_mead', 'lbfgs_b']
        
        for opt_type in optimizer_types:
            agent = Agent(
                num_vars=2,
                optimizer_type=opt_type,
                objective_fn=simple_objective,
                rng=rng
            )
            assert agent.optimizer is not None
    
    def test_invalid_optimizer(self, base_agent_params, simple_objective, rng):
        """Test that invalid optimizer type raises error."""
        params = base_agent_params.copy()
        params['optimizer_type'] = 'invalid_optimizer'
        with pytest.raises(ValueError):
            Agent(**params, objective_fn=simple_objective, rng=rng)

class TestAgentState:
    """Test state management."""
    
    def test_get_set_x(self, base_agent_params, simple_objective, rng):
        """Test x getter and setter."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        new_x = np.array([0.1, 0.2, 0.3])
        
        agent.set_x(new_x)
        np.testing.assert_array_equal(agent.get_x(), new_x)
    
    def test_get_set_fx(self, base_agent_params, simple_objective, rng):
        """Test fx getter and setter."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        nbhd_state = [agent.x, np.array([0.5, 0.5, 0.5])]  # Example neighborhood state
        
        agent.set_fx(nbhd_state)
        assert agent.get_fx() is not None
        assert isinstance(agent.get_fx(), (float, np.floating))
    
    def test_state_property(self, base_agent_params, simple_objective, rng):
        """Test state property returns correct information."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        nbhd_state = [agent.x, np.array([0.5, 0.5, 0.5])]
        agent.set_fx(nbhd_state)
        
        state = agent.state
        assert 'x' in state
        assert 'fx' in state
        assert 'trajectories' in state
        np.testing.assert_array_equal(state['x'], agent.x)
        assert state['fx'] == agent.fx
        assert state['trajectories'] is None

class TestAgentStep:
    """Test agent stepping behavior."""
    
    def test_step_shape(self, base_agent_params, simple_objective, rng):
        """Test that step returns correct shape."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        nbhd_state = [agent.x, np.array([0.5, 0.5, 0.5])]
        
        new_x = agent.step(nbhd_state)
        assert new_x.shape == (base_agent_params['num_vars'],)
    
    def test_step_bounds(self, base_agent_params, simple_objective, rng):
        """Test that step respects [0,1] bounds."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        nbhd_state = [agent.x, np.array([0.5, 0.5, 0.5])]
        
        for _ in range(100):  # Test multiple steps
            new_x = agent.step(nbhd_state)
            assert np.all(new_x >= 0) and np.all(new_x <= 1)
    
    def test_step_limit(self, base_agent_params, simple_objective, rng):
        """Test that step respects step limit."""
        agent = Agent(**base_agent_params, objective_fn=simple_objective, rng=rng)
        nbhd_state = [agent.x, np.array([0.5, 0.5, 0.5])]
        
        for _ in range(100):
            old_x = agent.x.copy()
            new_x = agent.step(nbhd_state)
            
            # Check component-wise step limit
            component_changes = np.abs(new_x - old_x)
            assert np.all(component_changes <= agent.step_limit + 1e-10)

    def test_trajectory_tracking(self, base_agent_params, simple_objective, rng):
        """Test trajectory tracking when enabled."""
        agent = Agent(
            **base_agent_params,
            objective_fn=simple_objective,
            rng=rng,
            track_trajectories=True
        )
        nbhd_state = [agent.x, np.array([0.5, 0.5, 0.5])]
        
        # Take several steps
        n_steps = 5
        for _ in range(n_steps):
            agent.step(nbhd_state)
        
        assert len(agent.trajectories) == n_steps
        for traj in agent.trajectories:
            assert 'x' in traj
            assert 'fx' in traj
            assert traj['x'].shape == (base_agent_params['num_vars'],)
            assert isinstance(traj['fx'], (float, np.floating))

if __name__ == '__main__':
    pytest.main([__file__])
