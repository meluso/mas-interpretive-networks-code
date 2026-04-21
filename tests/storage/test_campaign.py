# tests/storage/test_campaign.py

import pytest
import numpy as np
from pathlib import Path
import shutil
import h5py

from config import ParametersRegistry
from storage.campaign import CampaignStorage

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory for test data."""
    yield tmp_path
    # Cleanup after tests
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

@pytest.fixture
def campaign(test_dir):
    """Create a test campaign instance."""
    return CampaignStorage(
        study_name="test_study",
        campaign_id="test_campaign",
        param_set='test',
        data_dir=test_dir
    )

@pytest.fixture
def basic_params():
    """Create a basic parameter space for testing."""
    factory = ParametersRegistry.get_factory("test")
    params_object = factory.create_parameters()
    return params_object.get_dict()

def test_campaign_initialization(campaign):
    """Test campaign initialization and file structure."""
    # Check file creation
    assert campaign.file_path.exists()
    
    # Verify initial trial counter
    with h5py.File(campaign.file_path, "r") as f:
        assert f[campaign.metadata_group].attrs["trial_counter"] == 0

def test_parameter_space_storage(campaign, basic_params):
    """Test storing and retrieving parameter space."""
    
    # Retrieve parameters
    retrieved_params = campaign.get_parameter_space()
    
    # Verify all parameters match
    for key in basic_params:
        print(f"{key}: \n\t{retrieved_params[key]}\n\t{basic_params[key]}")
        assert key in retrieved_params
        assert retrieved_params[key] == basic_params[key]

def test_trial_creation_and_retrieval(campaign, basic_params):
    """Test creating and retrieving trials."""
    trial = campaign.create_trial(basic_params)
    trial.store_parameters(basic_params)
    
    # Get campaign and trial attributes
    campaign_attr = vars(campaign)
    trial_attr = vars(trial)
    
    # Check campaign and trial attributes match
    check = ['study_name', 'campaign_id', 'mode', 'file_path']
    for key in check:
        assert trial_attr[key] == campaign_attr[key]
    
    # Test trial retrieval
    retrieved_trial = campaign.get_trial("trial000000001")
    retrieved_metadata = retrieved_trial.load_metadata()
    
    for key in basic_params.keys():
        aa = basic_params[key]
        bb = retrieved_metadata['parameters'][key]
        try:
            __ = iter(aa)
            assert all(aa == bb)
        except:
            assert aa == bb

def test_invalid_trial_access(campaign):
    """Test error handling for invalid trial access."""
    with pytest.raises(KeyError):
        campaign.get_trial("nonexistent_trial")

def test_empty_campaign_listing(campaign):
    """Test listing trials in empty campaign."""
    assert campaign.list_trials() == []

def test_multiple_trial_listing(campaign):
    """Test listing multiple trials."""
    # Create several trials with minimal parameters
    base_params = {"team_size": 4, "team_graph_type": "complete"}
    for _ in range(3):
        campaign.create_trial(base_params)
    
    trials = campaign.list_trials()
    assert len(trials) == 3
    assert set(trials) == {"trial000000001", "trial000000002", "trial000000003"}
    
if __name__ == '__main__':
    pytest.main([__file__])