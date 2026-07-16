#!/usr/bin/env bash
# Run the complete demo: simulation, analysis, and figure generation.
set -euo pipefail

# Ensure dependencies are installed (in a fresh Codespace, the automatic
# install may still be running; this catches that case)
if ! python -c "import scipy, dask, seaborn, pyarrow" > /dev/null 2>&1; then
    echo "Dependencies not found; installing from requirements.txt..."
    pip install -r requirements.txt
fi

python simulation.py --study_name demo "$@"

# Find the most recent demo dataset that contains trial results
STUDY=""
for d in $(ls -td data/raw/demo_*/ 2>/dev/null); do
    if compgen -G "${d}trials/*.parquet" > /dev/null; then
        STUDY=$(basename "$d")
        break
    fi
done
if [ -z "$STUDY" ]; then
    echo "Error: no demo dataset with trial results found in data/raw/" >&2
    exit 1
fi
echo "Running analysis and figures for: $STUDY"

python analysis.py --study_name "$STUDY"
python plots.py --study_name "$STUDY"

echo ""
echo "Demo complete."
echo "  Results: data/results/$STUDY/"
echo "  Figures: figures/publication/png/ and figures/publication/pdf/"
echo "  Compare figures against demo/expected_output/"
