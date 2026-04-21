# config/studies.py
"""Study definitions for AI teams experiments.

Each study is mapped to a list of campaign numbers that define its experimental 
structure. Campaign numbers correspond to parameter sets in factory_implementations.py.
"""

from typing import Dict, List

STUDY_CAMPAIGNS: Dict[str, List[str]] = {
    'test': [
        'TestSize01',
        'TestSize02',
        'TestDefault'
        ],
    'test_specific': [
        'TestSpecific'
        ],
    'rehearsal': [
        'RehearsalSize01',
        'RehearsalSize02',
        'RehearsalDefault'
        ],
    'rehearsal_da': [
        'RehearsalSize01DualAnnealing',
        'RehearsalSize02DualAnnealing',
        'RehearsalDefaultDualAnnealing'
    ],
    'aiteams01nm': [
        'AITeams01Size01',
        'AITeams01Size02',
        'AITeams01Default'
        ],
    'aiteams01lb': [
        'AITeams01Size01LBFGSB',
        'AITeams01Size02LBFGSB',
        'AITeams01DefaultLBFGSB'
        ],
    'aiteams01rw': [
        'AITeams01Size01RandomWalk',
        'AITeams01Size02RandomWalk',
        'AITeams01DefaultRandomWalk'
        ],
    'aiteams01da': [
        'AITeams01Size01DualAnnealing',
        'AITeams01Size02DualAnnealing',
        'AITeams01DefaultDualAnnealing'
        ],
}

def validate_study(study_name: str) -> bool:
    """Validate that a study name exists in configuration.
    
    Args:
        study_name: Name of study to validate
        
    Returns:
        bool: True if study exists
        
    Raises:
        ValueError: If study does not exist
    """
    if study_name not in STUDY_CAMPAIGNS:
        raise ValueError(f"Study {study_name} not found in configuration")
    return True

def get_campaigns(study_name: str) -> List[str]:
    """Get list of campaign numbers for a study.
    
    Args:
        study_name: Name of study
        
    Returns:
        List of campaign numbers
    """
    validate_study(study_name)
    return STUDY_CAMPAIGNS[study_name]