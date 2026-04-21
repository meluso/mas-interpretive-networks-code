# plots.py
"""Generate all figures for the paper and supplement from saved analysis results."""

import logging
import time
from pathlib import Path

# Import figure modules
import figures.plot_joint_task_effects as plot_te
import figures.plot_performance_means as plot_pm


def setup_logging():
    """Configure logging for the plotting script."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/plots_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )


def plot_main_paper(study_name):
    """Generate main paper figures from a primary study."""
    logging.info(f"Generating main paper figures for: {study_name}")
    plot_te.plot_joint_task_effects(study_name)
    plot_pm.plot_convergence_performance(study_name)


def plot_supplement(study_names):
    """Generate supplement figures across additional optimizer studies."""
    for study_name in study_names:
        logging.info(f"Generating supplement figures for: {study_name}")
        plot_te.plot_joint_task_effects(study_name)
        plot_pm.plot_convergence_performance(study_name)


def generate_all_figures(main_study, supplement_studies):
    """Generate all paper and supplement figures."""
    start_time = time.time()
    logging.info("Starting figure generation pipeline")
    plot_main_paper(main_study)
    plot_supplement(supplement_studies)
    elapsed_time = time.time() - start_time
    logging.info(f"Figure generation completed in {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    setup_logging()
    main_study = 'aiteams01nm_20250128_223001'
    supplement_studies = [
        'aiteams01lb_20250130_210957',
        'aiteams01rw_20250321_215818',
        'aiteams01da_20251202_192855'
    ]
    generate_all_figures(main_study, supplement_studies)
