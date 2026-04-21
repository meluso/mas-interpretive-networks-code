#!/bin/bash
# run_study.sh - Run and monitor AI Teams simulation study
# 
# Usage: ./cluster/run_study.sh STUDY_NAME
#
# Example: ./cluster/run_study.sh aiteams01

# Exit on any error
set -e

# Check arguments
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 STUDY_NAME"
    echo "Example: $0 aiteams01"
    exit 1
fi

STUDY_NAME=$1

# Create required directories
mkdir -p logs
mkdir -p cluster/run

# Remove old monitor files
rm cluster/run/*

# Start simulation
echo "Starting simulation for study ${STUDY_NAME}..."
nohup python simulation.py --study_name ${STUDY_NAME} > cluster/run/simulation.out 2>&1 &

# Save process ID
echo $! > cluster/run/simulation.pid
echo "Simulation process ID: $!"

# Start monitor
echo "Starting monitor script..."
nohup python cluster/monitor.py > cluster/run/monitor.out 2>&1 &
echo "Monitor process ID: $!"

echo "Study is now running. You can:"
echo ""
echo "1. Check recent progress: tail -f logs/study_*.log"
echo "2. Check if running: ps -p \$(cat cluster/run/simulation.pid)"
echo "3. Monitor output: tail -f cluster/run/simulation.out"
echo "4. Monitor script output: tail -f cluster/run/monitor.out"
echo "5. Check completion: ls cluster/run/COMPLETE"
echo ""
echo "When complete, status will be available in cluster/run/completion_status.json"
