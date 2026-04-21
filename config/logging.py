# config/logging.py
from datetime import datetime
import logging
from pathlib import Path
import atexit

# Global variable to store the log file path
_log_file_path = None

def setup_logging(log_dir: Path) -> None:
    """Configure logging for the simulation.
    
    Args:
        log_dir: Directory to store log files
    """
    global _log_file_path
    
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Only create new log file if one doesn't exist
    if _log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file_path = log_dir / f"study_{timestamp}.log"
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])
    
    # File handler with DEBUG level and appropriate formatting
    file_handler = logging.FileHandler(_log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(message)s'  # Simplified format for console
    ))
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Make sure handlers are closed properly
    atexit.register(lambda: [h.close() for h in root_logger.handlers])

def worker_setup_logging(log_dir: Path):
    """Set up logging for worker processes using main process log file."""
    setup_logging(log_dir)