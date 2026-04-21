# analysis.py
"""Run all analyses for a specified raw dataset."""

import logging
import time
from pathlib import Path

# Import analysis files
import analysis.calculate_mean_differences as cmd
import analysis.create_dataset as cd
import analysis.network_properties_by_graph as npbg
import analysis.network_property_regression_effects as npre
import analysis.ols_regression as ols


def setup_logging():
    """Configure logging for the analysis script."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )


def run_analyses(study_name):
    """Run all analyses for a specified study name."""
    start_time = time.time()
    logging.info(f"Starting analysis pipeline for study: {study_name}")
    
    logging.info("Step 1/4: Creating analysis datasets...")
    cd.create_all_datasets(study_name)
    
    logging.info("Step 2/4: Calculating descriptive statistics...")
    cmd.calculate_relative_descriptive_statistics(study_name)
    
    logging.info("Step 3/4: Calculating network properties by graph...")
    npbg.calculate_network_properties(study_name)
    
    logging.info("Step 4/4: Running linear regression analyses...")
    ols.run_all_ols_regressions(study_name)
    npre.run_all_network_property_analyses(study_name)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Analysis pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    setup_logging()
    studies = [
        'aiteams01nm_20250128_223001',
        'aiteams01lb_20250130_210957',
        'aiteams01rw_20250321_215818',
        'aiteams01da_20251202_192855'
    ]
    for study_name in studies:
        run_analyses(study_name)
