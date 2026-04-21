# tests/storage/test_trial.py
"""Tests for trial.py storage implementation."""

import pytest
import numpy as np
from pathlib import Path
from storage.trial import TrialStorage

@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage with test data."""
    storage = TrialStorage(
        study_name="test_study",
        campaign_id="campaign_001",
        trial_id="trial_001",
        data_dir=tmp_path
    )
    return storage

@pytest.fixture
def sample_parameters():
    """Create sample parameters matching Team class initialization."""
    return {
        'team_size': 4,
        'agent_num_vars': 2,
        'team_graph_type': 'small_world',
        'team_graph_opts': {'k': 2, 'p': 0.1},
        'agent_optim_type': 'random_walk',
        'agent_optim_opts': None,
        'agent_steplim': 0.1,
        'fn_type': 'sin2',
        'fn_opts': {'weight': 'degree', 'frequency': 'uniform'}
    }

@pytest.fixture
def sample_network_metrics():
    """Create sample network metrics matching NetworkMetrics output."""
    return {
        'team_graph_centrality_degree_mean': 0.5,
        'team_graph_centrality_degree_stdev': 0.1,
        'team_graph_centrality_betweenness_mean': 0.3,
        'team_graph_centrality_betweenness_stdev': 0.2,
        'team_graph_centrality_eigenvector_mean': 0.4,
        'team_graph_centrality_eigenvector_stdev': 0.15,
        'team_graph_nearest_neighbor_degree_mean': 0.6,
        'team_graph_nearest_neighbor_degree_stdev': 0.12,
        'team_graph_clustering': 0.45,
        'team_graph_density': 0.33,
        'team_graph_assortativity': 0.2,
        'team_graph_pathlength': 1.8,
        'team_graph_diameter': 3
    }

@pytest.fixture
def sample_objective_metrics():
    """Create sample objective metrics matching Team class attributes."""
    return {
        'team_fn_diff_integral': 0.7,
        'team_fn_diff_peaks': 4,
        'team_fn_diff_alignment': 0.6,
        'team_fn_diff_interdep': 0.4
    }

def test_store_metadata(temp_storage, sample_parameters, 
                       sample_network_metrics, sample_objective_metrics):
    """Test storing and retrieving metadata."""
    # Store metadata
    temp_storage.store_parameters(sample_parameters)
    temp_storage.store_network_metrics(sample_network_metrics)
    temp_storage.store_objective_metrics(sample_objective_metrics)
    
    # Load and verify metadata
    loaded_meta = temp_storage.load_metadata()
    
    # Check parameters
    params = loaded_meta["parameters"]
    for key, value in sample_parameters.items():
        if isinstance(value, (dict, np.ndarray)):
            if isinstance(value, dict):
                assert params[key] == value
            else:
                assert np.array_equal(params[key], value)
        else:
            assert params[key] == value
    
    # Check network metrics
    net_metrics = loaded_meta["network_metrics"]
    for key, value in sample_network_metrics.items():
        assert np.isclose(net_metrics[key], value)
    
    # Check objective metrics - verify correct mapping of internal names
    obj_metrics = loaded_meta["objective_metrics"]
    assert np.isclose(obj_metrics["team_fn_diff_integral"], sample_objective_metrics["team_fn_diff_integral"])
    assert obj_metrics["team_fn_diff_peaks"] == sample_objective_metrics["team_fn_diff_peaks"]
    assert np.isclose(obj_metrics["team_fn_diff_alignment"], sample_objective_metrics["team_fn_diff_alignment"])
    assert np.isclose(obj_metrics["team_fn_diff_interdep"], sample_objective_metrics["team_fn_diff_interdep"])

def test_store_timesteps(temp_storage, sample_parameters):
    """Test storing and retrieving timestep data."""
    # Setup dimensions from parameters
    team_size = sample_parameters['team_size']
    agent_num_vars = sample_parameters['agent_num_vars']
    num_steps = 5
    
    # Create and store timestep data
    for t in range(num_steps):
        # Generate sample data matching Team class state structure
        states = np.random.uniform(0, 1, (team_size, agent_num_vars))
        performance = np.random.uniform(0, 1)
        productivity = np.random.uniform(-0.1, 0.1)  # Can be negative
        
        # Store timestep
        temp_storage.store_timestep(t, states, performance, productivity)
    
    # Load and verify all timeseries
    timeseries = temp_storage.load_timeseries()
    
    # Verify data shapes
    assert timeseries["agent_states"].shape == (num_steps, team_size, agent_num_vars)
    assert timeseries["team_performance"].shape == (num_steps,)
    assert timeseries["team_productivity"].shape == (num_steps,)
    
    # Load and verify specific timestep
    timestep = 2
    step_data = temp_storage.load_timestep(timestep)
    
    # Check shapes
    assert step_data["agent_states"].shape == (team_size, agent_num_vars)
    assert isinstance(step_data["team_performance"], (np.number, float))
    assert isinstance(step_data["team_productivity"], (np.number, float))
    
    # Verify data matches
    assert np.array_equal(step_data["agent_states"], timeseries["agent_states"][timestep])
    assert step_data["team_performance"] == timeseries["team_performance"][timestep]
    assert step_data["team_productivity"] == timeseries["team_productivity"][timestep]

def test_file_creation(temp_storage, sample_parameters):
    """Test file and directory creation."""
    # Store some data to trigger file creation
    temp_storage.store_parameters(sample_parameters)
    
    # Check file exists
    assert (Path(temp_storage.data_dir) / "test_study.h5").exists()

def test_multiple_trials(tmp_path):
    """Test storing multiple trials in same file."""
    # Create two trial storages
    storage1 = TrialStorage(
        study_name="test_study",
        campaign_id="campaign_001",
        trial_id="trial_001",
        data_dir=tmp_path
    )
    
    storage2 = TrialStorage(
        study_name="test_study",
        campaign_id="campaign_001",
        trial_id="trial_002",
        data_dir=tmp_path
    )
    
    # Create different data for each
    params1 = {'team_size': 4, 'agent_num_vars': 2}
    params2 = {'team_size': 8, 'agent_num_vars': 3}
    
    metrics1 = {'team_fn_diff_integral': 0.5, 'team_fn_diff_peaks': 2}
    metrics2 = {'team_fn_diff_integral': 0.7, 'team_fn_diff_peaks': 4}
    
    # Store data for each
    storage1.store_parameters(params1)
    storage1.store_objective_metrics(metrics1)
    
    storage2.store_parameters(params2)
    storage2.store_objective_metrics(metrics2)
    
    # Verify each trial has correct data
    loaded1 = storage1.load_metadata()
    loaded2 = storage2.load_metadata()
    
    # Check parameters
    assert loaded1["parameters"]["team_size"] == 4
    assert loaded2["parameters"]["team_size"] == 8
    assert loaded1["parameters"]["agent_num_vars"] == 2
    assert loaded2["parameters"]["agent_num_vars"] == 3
    
    # Check metrics
    assert loaded1["objective_metrics"]["team_fn_diff_integral"] == 0.5
    assert loaded2["objective_metrics"]["team_fn_diff_integral"] == 0.7
    assert loaded1["objective_metrics"]["team_fn_diff_peaks"] == 2
    assert loaded2["objective_metrics"]["team_fn_diff_peaks"] == 4

def test_error_handling(temp_storage, sample_parameters):
    """Test error handling for invalid operations."""
    # First create the file by storing some metadata
    temp_storage.store_parameters(sample_parameters)
    
    # Now test loading non-existent timestep
    with pytest.raises(KeyError):
        temp_storage.load_timestep(0)
    
    # Create a new storage object with non-existent file
    bad_storage = TrialStorage(
        study_name="nonexistent_study",
        campaign_id="campaign_001",
        trial_id="trial_001",
        data_dir=temp_storage.data_dir
    )
    
    # Test loading from non-existent file
    with pytest.raises(FileNotFoundError):
        bad_storage.load_metadata()

def test_state_bounds(temp_storage, sample_parameters):
    """Test handling of state bounds and constraints."""
    team_size = sample_parameters['team_size']
    agent_num_vars = sample_parameters['agent_num_vars']
    
    # Test bounds enforcement
    states = np.random.uniform(0, 1, (team_size, agent_num_vars))
    performance = 0.5
    productivity = 0.1
    
    # Store and load
    temp_storage.store_timestep(0, states, performance, productivity)
    loaded = temp_storage.load_timestep(0)
    
    # Verify bounds maintained
    assert np.all((loaded["agent_states"] >= 0) & (loaded["agent_states"] <= 1))
    assert 0 <= loaded["team_performance"] <= 1
    
if __name__ == '__main__':
    pytest.main([__file__])