# tests/models/objective/test_objective.py
import pytest
import numpy as np
from numpy.testing import assert_allclose
import itertools as it
from models.objectives.objective import (
    Average, Sphere, Root, Sin2, Sin2sphere, Sin2root,
    Losqr_hiroot, Hisqr_loroot, Max, Min, Median,
    Kth_power, Kth_root, Ackley
)

# Test fixtures for different degree scenarios
@pytest.fixture
def test_degrees_basic():
    """Test case with highly different degrees."""
    return [1, 5]

@pytest.fixture
def test_degrees_zero():
    """Test case including a disconnected node (degree 0)."""
    return [0, 3]

@pytest.fixture
def test_degrees_range():
    """Test case with full range of degrees 0-5."""
    return [0, 1, 2, 3, 4, 5]

@pytest.fixture
def weight_variants():
    """All possible weight options."""
    return ['node', 'degree']

@pytest.fixture
def frequency_variants():
    """All possible frequency options."""
    return ['uniform', 'degree']

@pytest.fixture
def exponent_variants():
    """All possible exponent options."""
    return ['uniform', 'degree']

# Group 1: Functions with only weight variants
WEIGHT_ONLY_FUNCTIONS = [
    (Average, "average"),
    (Sphere, "sphere"),
    (Root, "root"),
    (Kth_power, "kth_power"),
    (Kth_root, "kth_root")
]

@pytest.mark.parametrize("FunctionClass,name", WEIGHT_ONLY_FUNCTIONS)
@pytest.mark.parametrize("weight", ['node', 'degree'])
@pytest.mark.parametrize("degrees", ["test_degrees_basic", "test_degrees_zero", "test_degrees_range"])
def test_weight_only_functions(FunctionClass, name, weight, degrees, request):
    """Test functions that only have weight variants with different degree scenarios."""
    degrees = request.getfixturevalue(degrees)
    opts = {'weight': weight}
    num_vars = 2  # Number of variables per node
    fn = FunctionClass(opts, num_vars, degrees)
    
    # Generate test input with proper dimensionality
    x_test = np.linspace(0.3, 0.7, len(degrees) * num_vars)
    
    # Test basic properties
    result = fn(x_test)
    assert 0 <= result <= 1, f"{name} with {weight} weight out of bounds"
    
    # Test reproducibility
    assert_allclose(fn(x_test), fn(x_test), 
                   err_msg=f"{name} with {weight} weight not reproducible")
    
    if weight == 'degree':
        # For degree weighting, verify self-edge contribution
        if 0 in degrees:
            # Calculate expected result manually for a simple case
            x = np.ones(len(degrees) * num_vars)
            result = fn(x)
            assert 0 < result <= 1, f"{name} not handling self-edges correctly"

# Group 2: Functions with weight and frequency variants
WEIGHT_FREQ_FUNCTIONS = [
    (Sin2, "sin2"),
    (Sin2sphere, "sin2sphere"),
    (Sin2root, "sin2root")
]

@pytest.mark.parametrize("FunctionClass,name", WEIGHT_FREQ_FUNCTIONS)
@pytest.mark.parametrize("weight", ['node', 'degree'])
@pytest.mark.parametrize("frequency", ['uniform', 'degree'])
@pytest.mark.parametrize("degrees", ["test_degrees_basic", "test_degrees_zero", "test_degrees_range"])
def test_weight_freq_functions(FunctionClass, name, weight, frequency, degrees, request):
    """Test functions that have both weight and frequency variants."""
    degrees = request.getfixturevalue(degrees)
    opts = {'weight': weight, 'frequency': frequency}
    num_vars = 2  # Number of variables per node
    fn = FunctionClass(opts, num_vars, degrees)
    
    # Generate test input with proper dimensionality
    x_test = np.linspace(0.3, 0.7, len(degrees) * num_vars)
    
    # Test basic properties
    result = fn(x_test)
    assert 0 <= result <= 1, f"{name} with {weight}/{frequency} out of bounds"
    
    if frequency == 'degree':
        # Test that frequency scales with degree
        if 0 in degrees:
            # Disconnected nodes should have minimal oscillation
            zero_idx = degrees.index(0)
            base_state = np.ones(len(degrees) * num_vars) * 0.5
            
            # Test frequency scaling by varying variables of disconnected node
            results = []
            for val in np.linspace(0, 1, 100):
                test_state = base_state.copy()
                test_state[zero_idx*num_vars:(zero_idx+1)*num_vars] = val
                results.append(fn(test_state))
                
            # Check for reduced oscillation at disconnected node
            oscillation = np.max(results) - np.min(results)
            assert oscillation < 1, \
                f"{name} not handling degree-based frequency correctly for k=0"

# Group 3: Functions with weight and exponent variants
WEIGHT_EXP_FUNCTIONS = [
    (Losqr_hiroot, "losqr_hiroot"),
    (Hisqr_loroot, "hisqr_loroot")
]

@pytest.mark.parametrize("FunctionClass,name", WEIGHT_EXP_FUNCTIONS)
@pytest.mark.parametrize("weight", ['node', 'degree'])
@pytest.mark.parametrize("exponent", ['uniform', 'degree'])
@pytest.mark.parametrize("degrees", ["test_degrees_basic", "test_degrees_zero", "test_degrees_range"])
def test_weight_exp_functions(FunctionClass, name, weight, exponent, degrees, request):
    """Test functions that have both weight and exponent variants."""
    degrees = request.getfixturevalue(degrees)
    opts = {'weight': weight, 'exponent': exponent}
    num_vars = 2  # Number of variables per node
    fn = FunctionClass(opts, num_vars, degrees)
    
    # Generate test input with proper dimensionality
    x_test = np.linspace(0.3, 0.7, len(degrees) * num_vars)
    
    # Test basic properties
    result = fn(x_test)
    assert 0 <= result <= 1, f"{name} with {weight}/{exponent} out of bounds"
    
    if exponent == 'degree':
        # Test that exponent scales appropriately with degree
        if 0 in degrees and 5 in degrees:
            zero_idx = degrees.index(0)
            max_idx = degrees.index(5)
            base_state = np.ones(len(degrees) * num_vars) * 0.5
            
            # Set test values for zero-degree and max-degree nodes
            test_state = base_state.copy()
            test_state[zero_idx*num_vars:(zero_idx+1)*num_vars] = 0.9
            test_state[max_idx*num_vars:(max_idx+1)*num_vars] = 0.9
            
            result = fn(test_state)
            assert 0 <= result <= 1, \
                f"{name} not handling degree-based exponents correctly"

# Group 4: Functions with no variants
NO_VARIANT_FUNCTIONS = [
    (Max, "max"),
    (Min, "min"),
    (Median, "median"),
    (Ackley, "ackley")
]

@pytest.mark.parametrize("FunctionClass,name", NO_VARIANT_FUNCTIONS)
@pytest.mark.parametrize("degrees", ["test_degrees_basic", "test_degrees_zero", "test_degrees_range"])
def test_no_variant_functions(FunctionClass, name, degrees, request):
    """Test functions that have no variants."""
    degrees = request.getfixturevalue(degrees)
    opts = {}  # No options needed
    num_vars = 2  # Number of variables per node
    fn = FunctionClass(opts, num_vars, degrees)
    
    # Generate test input with proper dimensionality
    x_test = np.linspace(0.3, 0.7, len(degrees) * num_vars)
    
    # Test basic properties
    result = fn(x_test)
    assert 0 <= result <= 1, f"{name} out of bounds"
    
    if name == "max":
        assert_allclose(fn(x_test), np.max(x_test), err_msg="Max function error")
    elif name == "min":
        assert_allclose(fn(x_test), np.min(x_test), err_msg="Min function error")
    elif name == "median":
        assert_allclose(fn(x_test), np.median(x_test), err_msg="Median function error")

def test_degree_dependent_behavior():
    """Test specific behaviors that depend on node degrees."""
    degrees = [0, 1, 5]  # Test extreme cases
    num_vars = 2  # Variables per node
    
    # Create test values repeated for each node's variables
    x_base = np.array([0.2, 0.5, 0.8])
    x_test = np.repeat(x_base, num_vars)
    
    # Test degree weighting
    for FunctionClass, name in WEIGHT_ONLY_FUNCTIONS:
        fn_node = FunctionClass({'weight': 'node'}, num_vars, degrees)
        fn_degree = FunctionClass({'weight': 'degree'}, num_vars, degrees)
        
        # Node weighting treats all variables equally
        result_node = fn_node(x_test)
        
        # Degree weighting weights by degree + 1
        result_degree = fn_degree(x_test)
        
        assert result_node != result_degree, \
               f"{name} not showing different behavior for node vs degree weighting"
        
        if name == "average":
            # For average function, we can test exact values
            expected_node = np.mean(x_test)
            assert_allclose(result_node, expected_node)
            
            # With degree weighting: weighted by [1,2,6] for each variable
            weights = np.repeat([1, 2, 6], num_vars)  # Repeat weights for each var
            expected_degree = np.sum(weights * x_test) / np.sum(weights)
            assert_allclose(result_degree, expected_degree)

def test_self_edge_contribution():
    """Test how functions handle self-edges in degree weighting."""
    degrees = [0, 3, 5]  # Including disconnected node
    num_vars = 2
    
    for FunctionClass, name in WEIGHT_ONLY_FUNCTIONS:
        opts = {'weight': 'degree'}
        fn = FunctionClass(opts, num_vars, degrees)
        
        # Create base state
        x_base = np.ones(len(degrees) * num_vars) * 0.5
        
        # Create test states varying disconnected node's variables
        x_test1 = x_base.copy()
        x_test2 = x_base.copy()
        x_test1[0:num_vars] = 0.0  # Set disconnected node's vars to 0
        x_test2[0:num_vars] = 1.0  # Set disconnected node's vars to 1
        
        result1 = fn(x_test1)
        result2 = fn(x_test2)
        
        # Results should differ due to self-edge contribution
        assert result1 != result2, \
               f"{name} not showing self-edge contribution for isolated node"
        
        if name == "average":
            # Calculate expected values manually
            weights = np.repeat([1, 4, 6], num_vars)  # [k+1] repeated for each var
            total_weight = np.sum(weights)
            
            # Calculate expected weighted averages
            vars1 = np.concatenate([[0.0]*num_vars, [0.5]*num_vars*2])
            vars2 = np.concatenate([[1.0]*num_vars, [0.5]*num_vars*2])
            expected1 = np.sum(weights * vars1) / total_weight
            expected2 = np.sum(weights * vars2) / total_weight
            
            assert_allclose(result1, expected1)
            assert_allclose(result2, expected2)

@pytest.mark.parametrize("degrees", ["test_degrees_basic", "test_degrees_zero", "test_degrees_range"])
def test_edge_cases(degrees, request):
    """Test edge cases across all function variants."""
    degrees = request.getfixturevalue(degrees)
    num_vars = 2
    n = len(degrees) * num_vars  # Total variables
    
    test_cases = [
        np.zeros(n),      # All zeros
        np.ones(n),       # All ones
        np.array([0]*int(n/2) + [1]*(n-int(n/2))),  # Mix of extremes
        np.ones(n) * 0.5  # Mid points
    ]
    
    # Test each function group
    for FunctionClass, name in (WEIGHT_ONLY_FUNCTIONS + WEIGHT_FREQ_FUNCTIONS + 
                              WEIGHT_EXP_FUNCTIONS + NO_VARIANT_FUNCTIONS):
        
        # Get appropriate options for function type
        if name in [f[1] for f in WEIGHT_FREQ_FUNCTIONS]:
            variants = [
                {'weight': w, 'frequency': f} 
                for w, f in it.product(['node', 'degree'], ['uniform', 'degree'])
            ]
        elif name in [f[1] for f in WEIGHT_EXP_FUNCTIONS]:
            variants = [
                {'weight': w, 'exponent': e}
                for w, e in it.product(['node', 'degree'], ['uniform', 'degree'])
            ]
        elif name in [f[1] for f in WEIGHT_ONLY_FUNCTIONS]:
            variants = [{'weight': w} for w in ['node', 'degree']]
        else:
            variants = [{}]
        
        # Test each variant
        for opts in variants:
            fn = FunctionClass(opts, num_vars, degrees)
            for x_vals in test_cases:
                result = fn(x_vals)
                assert 0 <= result <= 1, \
                       f"Edge case failed for {name} with options {opts}"

def test_sqrt_numerical_stability():
    """Test handling of numerical underflow in sqrt calculations."""
    # Test both basic sqrt functions and composite functions
    test_cases = [
        (Root, {'weight': 'node'}),
        (Sin2root, {'weight': 'node', 'frequency': 'uniform'})
    ]
    
    degrees = [1, 5]
    num_vars = 2
    
    for FunctionClass, opts in test_cases:
        fn = FunctionClass(opts, num_vars, degrees)
        
        # Test with small negative value
        x_test = np.ones(len(degrees) * num_vars) * 0.5
        x_test[0] = -1e-8  # Add tiny negative value
        
        # Should not raise warning and return valid result
        result = fn(x_test)
        assert 0 <= result <= 1, f"Out of bounds result for {FunctionClass.__name__}"

if __name__ == '__main__':
    pytest.main([__file__])
