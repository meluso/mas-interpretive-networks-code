# tests/models/optimizers/test_optimizers.py
"""Tests for optimizer implementations."""

import pytest
import numpy as np
from numpy.random import default_rng
from models.optimizers import create_optimizer, OPTIMIZERS
from models.optimizers.base import BaseOptimizer
from models.optimizers.random_walk import RandomWalkOptimizer
from models.optimizers.scipy_wrappers import (
    DualAnnealingOptimizer,
    NelderMeadOptimizer,
    LBFGSBOptimizer
)
from models.objectives.objective import Average, Sphere

# ============= Fixtures ================

@pytest.fixture
def rng():
    """Provide consistent random number generator for tests."""
    return default_rng(seed=42)

@pytest.fixture
def dimensions():
    """Test dimensionalities for vector optimization."""
    return [1, 2, 5, 10]

@pytest.fixture
def neighborhood_degrees():
    """Test neighborhood degrees for objective functions."""
    return np.array([2, 3])  # Agent degree 2, neighbor degree 3

@pytest.fixture
def average_objective(neighborhood_degrees):
    """Create Average objective function for testing."""
    return Average({'weight': 'node'}, 16, neighborhood_degrees)

@pytest.fixture
def sphere_objective(neighborhood_degrees):
    """Create Sphere objective function for testing."""
    return Sphere({'weight': 'node'}, 16, neighborhood_degrees)

# ============= Base Optimizer Tests ================

class TestBaseOptimizer:
    """Tests for the abstract base optimizer class."""
    
    def test_cannot_instantiate_abstract(self):
        """Verify BaseOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseOptimizer()
            
    def test_requires_step_method(self):
        """Verify subclasses must implement step method."""
        class IncompleteOptimizer(BaseOptimizer):
            pass
            
        with pytest.raises(TypeError):
            IncompleteOptimizer()
            
    def test_stores_objective_function(self, average_objective):
        """Test objective function is properly stored."""
        class SimpleOptimizer(BaseOptimizer):
            def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
                return current_x, current_fx
                
        optimizer = SimpleOptimizer(objective_fn=average_objective)
        assert optimizer.objective_fn is average_objective

    def test_evaluate_helper(self, average_objective):
        """Test _evaluate helper method."""
        class SimpleOptimizer(BaseOptimizer):
            def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
                return current_x, current_fx
        
        optimizer = SimpleOptimizer(objective_fn=average_objective)
        x = np.array([0.5, 0.5])
        neighbor_x = np.array([0.3, 0.7])
        
        # Test without neighborhood
        fx = optimizer._evaluate(x)
        assert isinstance(fx, float)
        assert 0 <= fx <= 1
        
        # Test with neighborhood
        fx_with_nbr = optimizer._evaluate(x, [x, neighbor_x])
        assert isinstance(fx_with_nbr, float)
        assert 0 <= fx_with_nbr <= 1

    def test_get_bounds_helper(self):
        """Test _get_bounds helper method."""
        class SimpleOptimizer(BaseOptimizer):
            def step(self, current_x, current_fx, step_limit, neighborhood_state=None):
                return current_x, current_fx
        
        optimizer = SimpleOptimizer()
        current_x = np.array([0.5, 0.5])
        
        # Test with step_limit >= 1
        lower, upper = optimizer._get_bounds(current_x, 1.0)
        np.testing.assert_array_equal(lower, np.zeros_like(current_x))
        np.testing.assert_array_equal(upper, np.ones_like(current_x))
        
        # Test with small step_limit
        step_limit = 0.1
        lower, upper = optimizer._get_bounds(current_x, step_limit)
        np.testing.assert_array_equal(lower, np.maximum(0, current_x - step_limit))
        np.testing.assert_array_equal(upper, np.minimum(1, current_x + step_limit))
        
# ============= Specific Optimizer Tests ================

class TestRandomWalkOptimizer:
    """Tests for the random walk optimization strategy."""
    
    @pytest.fixture
    def optimizer(self, rng, average_objective):
        return RandomWalkOptimizer(objective_fn=average_objective, rng=rng)
    
    def test_bounds_respected(self, optimizer):
        """Test component-wise bounds are respected."""
        current_x = np.array([0.5, 0.5])
        step_limit = 0.1
        current_fx = optimizer._evaluate(current_x)
        
        # Generate many steps to test constraints
        n_samples = 1000
        for _ in range(n_samples):
            new_x, new_fx = optimizer.step(current_x, current_fx, step_limit)
            
            # Check bounds
            assert np.all(new_x >= 0)
            assert np.all(new_x <= 1)
            
            # Check step size
            assert np.all(np.abs(new_x - current_x) <= step_limit + 1e-10)
            
            # Check objective value
            assert isinstance(new_fx, float)
            assert 0 <= new_fx <= 1
    
    def test_hill_climbing(self, optimizer):
        """Test hill climbing behavior."""
        current_x = np.zeros(2)  # Start at suboptimal point
        current_fx = optimizer._evaluate(current_x)
        step_limit = 1.0  # Allow full movement
        
        # Try multiple steps
        for _ in range(10):
            new_x, new_fx = optimizer.step(current_x, current_fx, step_limit)
            # Should never decrease objective
            assert new_fx >= current_fx
            current_x, current_fx = new_x, new_fx
    
    def test_reproducibility(self, average_objective):
        """Test reproducible behavior with same seed."""
        optimizer1 = RandomWalkOptimizer(objective_fn=average_objective, rng=default_rng(42))
        optimizer2 = RandomWalkOptimizer(objective_fn=average_objective, rng=default_rng(42))
        
        current_x = np.array([0.5, 0.5])
        current_fx = optimizer1._evaluate(current_x)
        step_limit = 0.1
        
        # Should produce identical sequences
        for _ in range(10):
            new_x1, new_fx1 = optimizer1.step(current_x, current_fx, step_limit)
            new_x2, new_fx2 = optimizer2.step(current_x, current_fx, step_limit)
            np.testing.assert_array_equal(new_x1, new_x2)
            assert new_fx1 == new_fx2

# ============= General Optimizer Tests ================

@pytest.mark.parametrize("optimizer_class", OPTIMIZERS.values())
class TestOptimizers:
    """Tests for all optimizer implementations."""
    
    @pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
    def test_common_properties(self, optimizer_class, rng, average_objective):
        """Run common tests for all optimizers."""
        optimizer = optimizer_class(objective_fn=average_objective, rng=rng)
        
        # Test initialization
        assert isinstance(optimizer, BaseOptimizer)
        assert optimizer.objective_fn is average_objective
        assert optimizer.rng is rng
        
        # Test optimization step
        current_x = np.array([0.5, 0.5])
        current_fx = optimizer._evaluate(current_x)
        neighbor_x = np.array([0.3, 0.7])
        step_limit = 0.1
        
        new_x, new_fx = optimizer.step(
            current_x, 
            current_fx,
            step_limit, 
            [current_x, neighbor_x]
        )
        
        # Check output properties
        assert isinstance(new_x, np.ndarray)
        assert new_x.shape == current_x.shape
        assert np.all((new_x >= 0) & (new_x <= 1))
        assert isinstance(new_fx, float)
        assert 0 <= new_fx <= 1
        
        # Check component-wise step limits
        if step_limit < 1:
            component_changes = np.abs(new_x - current_x)
            assert np.all(component_changes <= step_limit + 1e-10)
    
    def test_optimization_progress(self, optimizer_class, average_objective):
        """Test optimizer makes progress toward better solutions."""
        optimizer = optimizer_class(objective_fn=average_objective)
        
        # Start from suboptimal point
        current_x = np.zeros(2)  # [0, 0] is suboptimal for average function
        current_fx = optimizer._evaluate(current_x)
        neighbor_x = np.array([0.5, 0.5])
        step_limit = 1.0  # Allow full movement
        
        # Try multiple steps
        initial_fx = current_fx
        for _ in range(5):
            new_x, new_fx = optimizer.step(
                current_x,
                current_fx,
                step_limit,
                [current_x, neighbor_x]
            )
            
            # Should never decrease objective
            assert new_fx >= current_fx
            
            current_x, current_fx = new_x, new_fx
        
        # Should improve overall
        assert current_fx > initial_fx
    
    @pytest.mark.filterwarnings("ignore:invalid value encountered in subtract")
    def test_reproducibility(self, optimizer_class, average_objective):
        """Test reproducible behavior with same seed."""
        rng1 = default_rng(42)
        rng2 = default_rng(42)
        
        optimizer1 = optimizer_class(objective_fn=average_objective, rng=rng1)
        optimizer2 = optimizer_class(objective_fn=average_objective, rng=rng2)
        
        current_x = np.array([0.5, 0.5])
        current_fx = optimizer1._evaluate(current_x)
        neighbor_x = np.array([0.3, 0.7])
        step_limit = 0.1
        
        for _ in range(5):
            new_x1, new_fx1 = optimizer1.step(
                current_x,
                current_fx,
                step_limit,
                [current_x, neighbor_x]
            )
            new_x2, new_fx2 = optimizer2.step(
                current_x,
                current_fx,
                step_limit,
                [current_x, neighbor_x]
            )
            
            # Should get identical results with same seed
            np.testing.assert_allclose(new_x1, new_x2, rtol=1e-10)
            assert new_fx1 == new_fx2

# ============= Factory Tests ================

class TestOptimizerFactory:
    """Tests for optimizer creation factory function."""
    
    @pytest.mark.parametrize("optimizer_type,expected_class", [
        ('random_walk', RandomWalkOptimizer),
        ('dual_annealing', DualAnnealingOptimizer),
        ('nelder_mead', NelderMeadOptimizer),
        ('lbfgsb', LBFGSBOptimizer)
    ])
    def test_create_known_optimizer(self, optimizer_type, expected_class):
        """Test creation of valid optimizer types."""
        optimizer = create_optimizer(optimizer_type)
        assert isinstance(optimizer, expected_class)
    
    def test_create_with_custom_rng(self, rng):
        """Test creation with custom random number generator."""
        for optimizer_type in OPTIMIZERS:
            optimizer = create_optimizer(optimizer_type, rng=rng)
            assert optimizer.rng is rng
    
    def test_reject_unknown_optimizer(self):
        """Test error on unknown optimizer type."""
        with pytest.raises(ValueError):
            create_optimizer('not_an_optimizer')
    
    def test_registry_contains_all_optimizers(self):
        """Verify all implemented optimizers are in registry."""
        expected = {
            'random_walk': RandomWalkOptimizer,
            'dual_annealing': DualAnnealingOptimizer,
            'nelder_mead': NelderMeadOptimizer,
            'lbfgsb': LBFGSBOptimizer
        }
        assert OPTIMIZERS == expected

if __name__ == '__main__':
    pytest.main([__file__])
