#!/usr/bin/env python3
# simulation.py
"""
Main entry point for AI teams simulation.

This script provides the command-line interface for running AI team simulations.
Core simulation functionality is delegated to StudyManager.

# Execution with 4 processes
Bash: python simulation.py --study_name test --processes 4
IPython: %run simulation.py --study_name test --processes 4
"""

import argparse
from datetime import datetime
import logging
from multiprocessing import cpu_count
from pathlib import Path
import sys

from runners.study_manager import StudyManager
from config.logging import setup_logging

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description='Run AI teams simulation study'
    )
    
    # Required arguments
    parser.add_argument(
        '--study_name',
        type=str,
        default='test',
        help='Name of study to execute (e.g. "rehearsal")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--processes',
        type=int,
        default=cpu_count(),
        help='Number of worker processes per campaign (default: cpu_count)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Log directory (default: logs)'
    )
    
    args = parser.parse_args()
    
    # Validate number of processes
    if args.processes < 1:
        parser.error("Number of processes must be at least 1")
    
    return args

def simulation() -> None:
    """Main simulation execution function."""
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    
    # Set timestamp for this simulation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parse arguments and setup logging
    args = parse_args()
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    try:
        # Log start of study
        logger.info(f"Starting study: {args.study_name}")
        if args.processes > 1:
            logger.info(f"Using {args.processes} processes per campaign")
        
        # Initialize and run study
        study_manager = StudyManager(study_name=args.study_name)
        results = study_manager.execute_study(processes=args.processes)
        
        # Log completion
        logger.info("\nStudy execution completed successfully")
        
    except Exception as e:
        logger.error(f"Study execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    simulation()