# cluster/monitor.py
"""Monitor script to check simulation completion and mark status."""

import os
import time
import glob
import json
from datetime import datetime
from pathlib import Path
import psutil

def check_completion(pid_file: str = "cluster/run/simulation.pid", 
                    log_dir: str = "logs") -> bool:
    """Check if simulation has completed successfully."""
    # Read PID
    with open(pid_file) as f:
        pid = int(f.read().strip())
    
    # Check if process is still running
    if psutil.pid_exists(pid):
        return False
        
    # Process has ended, check logs for successful completion
    log_files = sorted(glob.glob(f"{log_dir}/study_*.log"))
    if not log_files:
        raise FileNotFoundError("No log files found")
        
    latest_log = log_files[-1]
    with open(latest_log) as f:
        last_lines = f.readlines()[-20:]  # Check last 20 lines
        return any("Study execution completed successfully" in line 
                  for line in last_lines)

def write_status(run_dir: str, success: bool):
    """Write completion status to a JSON file.
    
    Args:
        run_dir: Directory for run files
        success: Whether simulation completed successfully
    """
    # Create status info
    status = {
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if success else 'failed',
        'pid_file': 'cluster/run/simulation.pid'
    }
    
    # Write status file
    status_file = Path(run_dir) / 'completion_status.json'
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    # Write a simple flag file for easy checking
    flag_file = Path(run_dir) / ('COMPLETE' if success else 'FAILED')
    flag_file.touch()
    
    print(f"Status written to {status_file}")

def main(run_dir: str = 'cluster/run', check_interval: int = 300):
    """Monitor simulation and write status on completion.
    
    Args:
        run_dir: Directory for run files
        check_interval: Seconds between checks
    """
    print("Starting simulation monitor...")
    
    while True:
        try:
            # Check completion
            if check_completion():
                print("Simulation completed successfully")
                write_status(run_dir, True)
                break
            time.sleep(check_interval)
            
        except FileNotFoundError:
            print("Error: Required files not found")
            write_status(run_dir, False)
            break
        except Exception as e:
            print(f"Error checking completion: {e}")
            write_status(run_dir, False)
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='cluster/run',
                       help='Directory for run files')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')
    args = parser.parse_args()
    
    main(args.run_dir, args.interval)