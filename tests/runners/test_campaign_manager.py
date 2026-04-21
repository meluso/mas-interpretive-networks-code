# tests/pipeline/test_campaign_manager.py
"""Tests for campaign management functionality."""
# import pytest
# import numpy as np
# import time
# from pathlib import Path
# from typing import Dict

# from config.parameters import Parameters
from runners.campaign_manager import CampaignManager
# from storage.trial import TrialStorage
# from storage.campaign import CampaignStorage

        
# @pytest.fixture(autouse=True)
# def cleanup_hdf5(tmp_path):
#     yield
#     for file in tmp_path.glob("*.h5"):
#         file.unlink()

# @pytest.fixture
# def base_params() -> Dict:
#     """Basic parameter set for testing."""
#     return {
#         'team_size': 4,
#         'team_graph_type': 'complete',
#         'agent_steplim': 0.1,
#         'agent_optim_type': 'random_walk',
#         'fn_type': 'average',
#         'fn_opts': {'weight': 'node'},
#         'num_steps': 10
#     }

# @pytest.fixture
# def campaign_manager(tmp_path):
#     timestamp = int(time.time())
#     manager = CampaignManager(
#         study_name=f"test_study_{timestamp}",
#         campaign_id="test_campaign",
#         data_dir=tmp_path
#     )
#     return manager

# def test_initialization(campaign_manager):
#     """Test campaign manager initialization."""
#     assert isinstance(campaign_manager.parameters, Parameters)
#     assert campaign_manager.campaign_storage.campaign_id == "test_campaign"
#     assert isinstance(campaign_manager.rng, np.random.Generator)

# def test_storage_setup(tmp_path):
#     """Test storage hierarchy creation."""
#     manager = CampaignManager(
#         study_name="storage_test",
#         campaign_id="setup_test",
#         data_dir=tmp_path
#     )
#     h5_file = tmp_path / "storage_test.h5"
#     assert h5_file.exists()
#     assert manager.campaign_storage.file_path == h5_file
#     assert isinstance(manager.campaign_storage, CampaignStorage)

# def test_single_trial_execution(campaign_manager, base_params):
#     """Test basic execution of a single trial."""
#     trial_id = campaign_manager._execute_trial(base_params)
#     trial = campaign_manager.campaign_storage.get_trial(trial_id)
    
#     metadata = trial.load_metadata()
#     stored_params = metadata['parameters']
    
#     # Check all parameters including dependent ones
#     assert stored_params['team_size'] == base_params['team_size']
#     assert stored_params['agent_num_vars'] == 32 // base_params['team_size']
#     assert stored_params['team_graph_type'] == base_params['team_graph_type']
    
#     # Check timeseries data
#     timeseries = trial.load_timeseries()
#     assert len(timeseries['agent_states']) == base_params['num_steps']
#     assert timeseries['agent_states'].shape[1] == 32  # Total variables
#     assert len(timeseries['team_performance']) == base_params['num_steps']
#     assert len(timeseries['team_productivity']) == base_params['num_steps']

# def test_trial_metrics(campaign_manager, base_params):
#     """Test metrics storage in trial execution."""
#     trial_id = campaign_manager._execute_trial(base_params)
#     trial = campaign_manager.campaign_storage.get_trial(trial_id)
#     metadata = trial.load_metadata()
    
#     # Check network metrics
#     net_metrics = list(metadata['network_metrics'].keys())
#     required_metrics = {
#         'team_graph_centrality_degree_mean',
#         'team_graph_centrality_degree_stdev',
#         'team_graph_centrality_betweenness_mean',
#         'team_graph_centrality_betweenness_stdev',
#         'team_graph_centrality_eigenvector_mean',
#         'team_graph_centrality_eigenvector_stdev',
#         'team_graph_nearest_neighbor_degree_mean',
#         'team_graph_nearest_neighbor_degree_stdev',
#         'team_graph_clustering',
#         'team_graph_density',
#         'team_graph_assortativity',
#         'team_graph_pathlength',
#         'team_graph_diameter'
#     }    
#     assert all(metric in net_metrics for metric in required_metrics)
    
#     # Check objective metrics
#     obj_metrics = list(metadata['objective_metrics'].keys())
#     print(obj_metrics)
#     required_metrics = {
#         'team_fn_diff_integral',
#         'team_fn_diff_peaks',
#         'team_fn_diff_alignment',
#         'team_fn_diff_interdep'
#     }
#     assert all(metric in obj_metrics for metric in required_metrics)

# def test_parameter_combinations(campaign_manager):
#     """Test campaign execution with parameter combinations."""
#     param_subset = {
#         'team_size': [2, 4],
#         'team_graph_type': ['complete', 'star'],
#         'agent_steplim': [0.1],
#         'agent_optim_type': ['random_walk'],
#         'fn_type': ['average'],
#         'num_steps': 5
#     }
    
#     print(f"Input type: {type(param_subset['team_size'])}")
    
#     trial_ids = campaign_manager.execute_campaign(param_subset=param_subset)
    
#     # Count unique combinations
#     sizes = set()
#     graphs = set()
#     weights = set()
    
#     for trial_id in trial_ids:
#         trial = campaign_manager.campaign_storage.get_trial(trial_id)
#         params = trial.load_metadata()['parameters']
        
#         print(trial.__dict__)
#         print(params)
#         print(params['team_size'])
#         print(f"Input type: {type(param_subset['team_size'])}")
        
#         sizes.add(params['team_size'])
#         graphs.add(params['team_graph_type'])
#         weights.add(params['fn_opts']['weight'])
    
#     assert sizes == {2, 4}
#     assert graphs == {'complete', 'star'}
#     assert weights == {'node', 'degree'}

# def test_campaign_partitioning(campaign_manager):
#     """Test partitioned campaign execution."""
#     param_subset = {
#         'team_size': [4],
#         'team_graph_type': ['complete', 'star'],
#         'agent_steplim': [0.1],
#         'agent_optim_type': ['random_walk'],
#         'fn_type': ['average'],
#         'num_steps': 5
#     }
    
#     # Execute partitions
#     part0_ids = campaign_manager.execute_campaign(
#         param_subset=param_subset,
#         partition_id=0,
#         total_partitions=2,
#         allow_resume=True  # Added for partition execution
#     )
    
#     part1_ids = campaign_manager.execute_campaign(
#         param_subset=param_subset,
#         partition_id=1,
#         total_partitions=2,
#         allow_resume=True  # Added for partition execution
#     )
    
#     assert len(part0_ids) + len(part1_ids) == 4  # 2 graphs × 2 weights
#     assert not set(part0_ids) & set(part1_ids)  # No overlap

# def test_campaign_resume(campaign_manager):
#     """Test campaign resumption."""
#     param_subset = {
#         'team_size': [4],
#         'team_graph_type': ['complete', 'star'],
#         'agent_steplim': [0.1],
#         'agent_optim_type': ['random_walk'],
#         'fn_type': ['average'],
#         'num_steps': 5
#     }
    
#     # Execute partial campaign
#     campaign_manager.execute_campaign(
#         param_subset=param_subset,
#         partition_id=0,
#         total_partitions=2,
#         allow_resume=True  # Added for partial execution
#     )
    
#     # Resume with remaining parameters
#     remaining = campaign_manager.campaign_storage.get_remaining_parameters()
#     _ = campaign_manager.execute_campaign(
#         param_subset=remaining,
#         allow_resume=True  # Added for resumption
#     )
    
#     # Resume with remaining parameters
#     remaining = campaign_manager.campaign_storage.get_remaining_parameters()
#     _ = campaign_manager.execute_campaign(param_subset=remaining)
    
#     completed = campaign_manager.campaign_storage.get_completed_parameters()
#     assert len(completed) == 4  # All combinations completed

# def test_error_handling(campaign_manager, base_params):
#     """Test error handling in trial execution."""
#     invalid_params = base_params.copy()
#     invalid_params['team_size'] = 3  # Invalid team size
    
#     with pytest.raises(AssertionError, match="Invalid team size"):
#         campaign_manager._execute_trial(invalid_params)
    
#     # No trial should be created for invalid parameters
#     trial_ids = campaign_manager.campaign_storage.list_trials()
#     assert len(trial_ids) == 0


    
#     # Check error storage
#     trial_ids = campaign_manager.campaign_storage.list_trials()
#     if trial_ids:
#         last_trial = campaign_manager.campaign_storage.get_trial(trial_ids[-1])
#         metadata = last_trial.load_metadata()
#         assert 'error' in metadata.get('objective_metrics', {})

if __name__ == '__main__':
    
    # Create test campaign
    camp_man = CampaignManager(
        param_set='test'
        )
    
    # Execute test campaign
    trial_ids = camp_man.execute_campaign()
    print(trial_ids)
    test_trial = camp_man.campaign_storage.get_trial('trial000000247')