# runners/study_manager.py
"""Study management for AI teams simulation."""

import logging
import numpy as np
from datetime import datetime
from multiprocessing import Pool, current_process
from pathlib import Path
from time import time, sleep, perf_counter
from typing import Dict, List, Union
import psutil

from models.team import Team
from runners.trial_storage import TrialWriter
from config import ParametersRegistry, get_campaigns, validate_study
from config.defaults import DEFAULT_DATA_DIR
from config.logging import worker_setup_logging

# Create module-level logger
logger = logging.getLogger(__name__)

class StudyManager:
    """Manages execution of simulation studies."""
    
    def __init__(self,
                 study_name: str,
                 data_dir: Union[str, Path] = DEFAULT_DATA_DIR):
        """Initialize study manager.
        
        Args:
            study_name: Name of study to execute (must exist in studies.py)
            data_dir: Optional data storage directory
        """
        # Validate study exists
        validate_study(study_name)
        
        # Create timestamped study directory
        self.study_name = study_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(data_dir) / f"{study_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer registry for parallel execution
        self.writers = {}
        self.logger = logger
        
    def execute_study(self, processes: int = 1) -> Dict[str, List[str]]:
        """Execute all campaigns in study."""
        # Get campaigns for this study
        campaigns = get_campaigns(self.study_name)
        
        # Execute each campaign sequentially
        results = {}
        for i, campaign in enumerate(campaigns):
            # Add sleep between campaigns (not before first one)
            if i > 0:
                sleep(2)
                
            self.logger.info(f"\nExecuting campaign{campaign}")
            
            # Initialize progress tracking
            self.start_time = time()
            self.last_percent = -1
            
            # Execute campaign
            trial_ids = self._execute_campaign(campaign, processes)
            
            # Store results
            results[f"campaign{campaign}"] = trial_ids
            
        # Sleep before final message to ensure all campaign output is done
        sleep(2)
        
        return results
    
    def _execute_campaign(self, 
                         campaign: str,
                         processes: int) -> List[str]:
        """Execute a single campaign.
        
        Args:
            campaign: Campaign identifier
            processes: Number of processes to use
            
        Returns:
            List of executed trial IDs
        """
        if processes > 1:
            return self._execute_parallel_campaign(campaign, processes)
        else:
            return self._execute_serial_campaign(campaign)
            
    def _execute_serial_campaign(self, campaign: str) -> List[str]:
        """Execute campaign serially.
        
        Args:
            campaign: Campaign identifier
            
        Returns:
            List of executed trial IDs
        """
        # Create factory and get combinations
        factory = ParametersRegistry.get_factory(f"campaign{campaign}")
        params = factory.create_parameters()
        combinations = params.get_combinations()
        total_trials = len(combinations)
        
        self.logger.info(f"  Total trials: {total_trials}")
        
        # Create writer for single process
        writer = TrialWriter(self.output_dir, 0)
        
        # Execute trials sequentially
        trial_ids = []
        for i, params in enumerate(combinations, 1):
            results = self._compute_trial(params, campaign)
            trial_id = results['trial_id']
            writer.add_trial(results)
            trial_ids.append(trial_id)
            self.logger.debug(f"Completed trial {i}/{total_trials}")
            self._report_progress(campaign, i, total_trials)
        
        writer.close()
        return trial_ids
    
    
    def _execute_parallel_campaign(self, campaign: str, processes: int) -> List[str]:
        """Execute campaign using parallel processing."""
        
        # Create factory and get combinations
        factory = ParametersRegistry.get_factory(f"campaign{campaign}")
        params = factory.create_parameters()
        combinations = params.get_combinations()
        total_trials = len(combinations)
        
        self.logger.info(f"  Total trials: {total_trials}")
        
        # Execute trials in parallel
        trial_ids = []
        
        # Initialize Pool with logging setup
        from functools import partial
        setup_worker = partial(worker_setup_logging, Path('logs'))
        
        with Pool(processes, initializer=setup_worker) as pool:
            
            # Create writers for each process
            self.writers = {
                p: TrialWriter(self.output_dir)
                for p in range(processes)
            }
            
            # Prepare argument tuples for each combination
            args = [(params, campaign) for params in combinations]
            
            for i, results in enumerate(
                pool.imap_unordered(self._compute_trial_parallel, args), 1):
                pid = current_process().pid % processes
                self.writers[pid].add_trial(results)
                trial_ids.append(results['trial_id'])
                self.logger.debug(f"Completed trial {i}/{total_trials}")
                self._report_progress(campaign, i, total_trials)
            
            # Close all writers
            for writer in self.writers.values():
                writer.close()
                
        return trial_ids
    
    def _compute_trial_parallel(self, args):
        """Wrapper for parallel execution of compute_trial.
        
        Args:
            args: Tuple of (params, campaign)
        """
        params, campaign = args
        return self._compute_trial(params, campaign)
    
    def _compute_trial(self, params: Dict, campaign: str) -> Dict:
        """Compute a single trial with given parameters.
        
        Args:
            params: Parameter dictionary for trial
            campaign: Campaign identifier
            
        Returns:
            Dictionary containing flattened trial results
        """
        # Add memory check before we start
        process = psutil.Process()
        start_mem = process.memory_info()
        total_start = perf_counter()
        trial_num = params['num_trial']
        
        # Save old completion message for compatibility
        self.logger.debug(f"Started trial {trial_num}")
        
        # Calculate dependent parameters
        param_start = perf_counter()
        params = self._calculate_dependent_parameters(params)
        param_time = perf_counter() - param_start
    
        # Time team creation and simulation
        compute_start = perf_counter()
        team = Team(
            team_size=params['team_size'],
            team_graph_type=params['team_graph_type'],
            agent_num_vars=params['agent_num_vars'],
            agent_steplim=params['agent_steplim'],
            agent_optim_type=params['agent_optim_type'],
            fn_type=params['fn_type'],
            team_graph_opts=params.get('team_graph_opts'),
            agent_optim_opts=params.get('agent_optim_opts'),
            fn_opts=params.get('fn_opts')
        )
    
        # Initialize timeseries data
        timeseries = {
            'states': [],
            'performance': [],
            'productivity': []
        }
    
        # Run simulation steps
        num_steps = params.get('num_steps', 25)
        performance_history = []
        convergence_step = num_steps  # Default if no convergence
    
        # Execute steps and check convergence
        for step in range(num_steps + 1):
            if step > 0:
                team.step()
    
            # Record state
            timeseries['states'].append(team.get_team_xs())
            current_performance = team.get_team_fx()
            timeseries['performance'].append(current_performance)
            timeseries['productivity'].append(team.get_team_dfdt())
    
            # Check convergence after collecting at least 4 performance values
            performance_history.append(current_performance)
            if step >= 4 and convergence_step == num_steps:
                if self._check_convergence(performance_history):
                    convergence_step = step
    
        compute_time = perf_counter() - compute_start
    
        # Time the dictionary creation
        prep_start = perf_counter()
        results = {
            # Study organization
            'study_id': self.study_name,
            'campaign_id': f"campaign{campaign}",
            'trial_id': int(params['num_trial']),
            
            # Core parameters
            'team_size': params['team_size'],
            'agent_num_vars': params['agent_num_vars'],
            'agent_steplim': params['agent_steplim'],
            'agent_optim_type': params['agent_optim_type'],
            'graph_slug': params['team_graph_slug'],
            'fn_slug': params['fn_slug'],
            
            # Network metrics
            'team_graph_centrality_degree_mean': team.network_metrics['team_graph_centrality_degree_mean'],
            'team_graph_centrality_degree_stdev': team.network_metrics['team_graph_centrality_degree_stdev'],
            'team_graph_centrality_betweenness_mean': team.network_metrics['team_graph_centrality_betweenness_mean'],
            'team_graph_centrality_betweenness_stdev': team.network_metrics['team_graph_centrality_betweenness_stdev'],
            'team_graph_centrality_eigenvector_mean': team.network_metrics['team_graph_centrality_eigenvector_mean'],
            'team_graph_centrality_eigenvector_stdev': team.network_metrics['team_graph_centrality_eigenvector_stdev'],
            'team_graph_nearest_neighbor_degree_mean': team.network_metrics['team_graph_nearest_neighbor_degree_mean'],
            'team_graph_nearest_neighbor_degree_stdev': team.network_metrics['team_graph_nearest_neighbor_degree_stdev'],
            'team_graph_clustering': team.network_metrics['team_graph_clustering'],
            'team_graph_density': team.network_metrics['team_graph_density'],
            'team_graph_assortativity': team.network_metrics['team_graph_assortativity'],
            'team_graph_pathlength': team.network_metrics['team_graph_pathlength'],
            'team_graph_diameter': team.network_metrics['team_graph_diameter'],
            
            # Objective metrics
            'team_fn_diff_integral': team.team_fn_diff_integral,
            'team_fn_diff_peaks': team.team_fn_diff_peaks,
            'team_fn_diff_alignment': team.team_fn_diff_alignment,
            'team_fn_diff_interdep': team.team_fn_diff_interdep,
            
            # Output metrics
            'convergence_step': convergence_step,
            'convergence_performance': performance_history[convergence_step],
            'final_performance': performance_history[-1],
            
            # Timeseries data
            'agent_states': np.array(timeseries['states']),
            'team_performance': np.array(timeseries['performance']),
            'team_productivity': np.array(timeseries['productivity'])
        }
    
        prep_time = perf_counter() - prep_start
    
        # Get final memory stats
        end_mem = process.memory_info()
        total_time = perf_counter() - total_start
    
        # Log detailed timing and memory info
        self.logger.debug(
            f"Trial {trial_num} performance: "
            f"param={param_time:.3f}s, "
            f"compute={compute_time:.3f}s, "
            f"prep={prep_time:.3f}s, "
            f"total={total_time:.3f}s, "
            f"memory={((end_mem.rss - start_mem.rss) / 1024**2):.1f}MB"
        )
        
        # Keep old completion message for compatibility
        self.logger.debug(f"Completed trial {trial_num}")
    
        return results

    def _check_convergence(self, performance_history: List[float]) -> bool:
        """Check if team performance has converged.
        
        Convergence is defined as three consecutive performance changes < 0.0001.
        
        Args:
            performance_history: List of performance values
            
        Returns:
            True if converged, False otherwise
        """
        if len(performance_history) < 4:
            return False
            
        # Calculate three consecutive differences
        diffs = np.diff(performance_history[-4:])
        
        # Check if all three differences are small
        return all(abs(diff) < 0.0001 for diff in diffs)
    
    def _calculate_dependent_parameters(self, params: Dict) -> Dict:
        """Calculate dependent parameters based on provided parameters.
        
        Args:
            params: Base parameter dictionary
            
        Returns:
            Updated parameter dictionary with calculated values
        """
        def split_dict(input_dict):
            """Helper function for splitting dictionaries"""
            type_val, opts = next(iter(input_dict.items()))
            return type_val, opts
        
        # Calculate agent_num_vars based on team_size
        total_vars = 32
        vars_per_agent = total_vars / params['team_size']
        params['agent_num_vars'] = int(vars_per_agent)
        
        # Split out graph and function options
        params['team_graph_type'], params['team_graph_opts'] = \
            split_dict(params['team_graph2opts'])
        params['fn_type'], params['fn_opts'] = \
            split_dict(params['fn_type2opts'])
            
        # Create slugs
        from config.util import create_graph_slug, create_fn_slug
        params['team_graph_slug'] = create_graph_slug(
            params['team_graph_type'],
            params['team_graph_opts']
        )
        params['fn_slug'] = create_fn_slug(
            params['fn_type'],
            params['fn_opts']
        )
        
        return params
    
    def _report_progress(self, campaign_id: str, completed: int, total: int):
        """Report progress percentage for current campaign."""
        current_percent = int((completed / total) * 100)
        
        # Update progress whenever the percentage changes
        if current_percent > self.last_percent:
            elapsed = time() - self.start_time
            
            # For intermediate updates, use carriage return to overwrite
            if current_percent < 100:
                print(f"\r  {current_percent}% complete ({elapsed:.1f}s elapsed)", end="", flush=True)
            else:
                # For 100%, print on new line
                print(f"\r  100% complete ({elapsed:.1f}s elapsed)")
            
        self.last_percent = current_percent