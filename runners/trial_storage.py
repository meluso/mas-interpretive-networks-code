# storage/parquet_storage.py
"""Parquet-based storage implementation for AI teams simulation.

Provides schema definition and writer class for storing trial data in Parquet format.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Define schema for all columns
SCHEMA = pa.schema([
    # Study organization
    ('study_id', pa.string()),
    ('campaign_id', pa.string()),
    ('trial_id', pa.uint32()),
    
    # Core parameters
    ('team_size', pa.uint8()),
    ('agent_num_vars', pa.uint8()),
    ('agent_steplim', pa.float32()),
    ('agent_optim_type', pa.string()),
    ('graph_slug', pa.string()),
    ('fn_slug', pa.string()),
    
    # Network metrics
    ('team_graph_centrality_degree_mean', pa.float32()),
    ('team_graph_centrality_degree_stdev', pa.float32()),
    ('team_graph_centrality_betweenness_mean', pa.float32()),
    ('team_graph_centrality_betweenness_stdev', pa.float32()),
    ('team_graph_centrality_eigenvector_mean', pa.float32()),
    ('team_graph_centrality_eigenvector_stdev', pa.float32()),
    ('team_graph_nearest_neighbor_degree_mean', pa.float32()),
    ('team_graph_nearest_neighbor_degree_stdev', pa.float32()),
    ('team_graph_clustering', pa.float32()),
    ('team_graph_density', pa.float32()),
    ('team_graph_assortativity', pa.float32()),
    ('team_graph_pathlength', pa.float32()),
    ('team_graph_diameter', pa.float32()),
    
    # Objective metrics
    ('team_fn_diff_integral', pa.float32()),
    ('team_fn_diff_peaks', pa.float64()),  # Needs higher precision
    ('team_fn_diff_alignment', pa.float32()),
    ('team_fn_diff_interdep', pa.float32()),
    
    # Output metrics
    ('convergence_step', pa.uint8()),
    ('convergence_performance', pa.float32()),
    ('final_performance', pa.float32()),
    
    # Timeseries data as nested arrays
    ('agent_states', pa.list_(pa.list_(pa.float32()))),
    ('team_performance', pa.list_(pa.float32())),
    ('team_productivity', pa.list_(pa.float32()))
])

class TrialWriter:
    """Manages batch writing of trials during simulation."""
    
    def __init__(self, output_dir: Path):
        """Initialize writer for batch storage.
        
        Args:
            output_dir: Base directory for study output
        """
        # Create single trials directory for all processes
        self.output_dir = output_dir / "trials"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = 250
        self.current_batch = []
        
    def add_trial(self, trial_data: Dict):
        """Add trial data to current batch, writing if batch is full.
        
        Args:
            trial_data: Dictionary containing trial data matching schema
        """
        # Convert numpy arrays to lists for storage
        processed_data = trial_data.copy()
        for key in ['agent_states', 'team_performance', 'team_productivity']:
            if isinstance(processed_data[key], np.ndarray):
                processed_data[key] = processed_data[key].tolist()
        
        self.current_batch.append(processed_data)
        if len(self.current_batch) >= self.batch_size:
            self._write_batch()
            
    def _write_batch(self):
        """Write current batch to Parquet file with compression."""
        batch_num = len(list(self.output_dir.glob("*.parquet")))
        output_file = self.output_dir / f"batch_{batch_num:05d}.parquet"
        
        # Convert to pandas then arrow for schema validation
        df = pd.DataFrame(self.current_batch)
        table = pa.Table.from_pandas(df, schema=SCHEMA)
        
        # Write with high compression
        pq.write_table(
            table,
            output_file,
            compression='zstd',
            compression_level=3
        )
        
        self.current_batch = []
        
    def close(self):
        """Write any remaining trials in final batch."""
        if self.current_batch:
            self._write_batch()
    
if __name__ == '__main__':

    from pathlib import Path
    
    # Resolve path relative to project root
    project_root = Path(__file__).parent.parent
    test_file = project_root / "data/raw/rehearsal_20250128_160853"
    print(f"\nReading {test_file}")
    
    # Read and show what we got
    df = pd.read_parquet(test_file)
    print(f"\nFound {len(df)} trials")
    print("\nColumns:", df.columns.tolist())
    
    # Look at first trial
    if len(df) > 0:
        trial = df.iloc[-1]
        print("\nFirst Trial:")
        print(f"Trial ID: {trial['trial_id']}")
        print(f"Team Size: {trial['team_size']}")
        print(f"Final Performance: {trial['final_performance']:.4f}")
        print(f"Agent States Shape: {len(trial['agent_states'])} timesteps")
