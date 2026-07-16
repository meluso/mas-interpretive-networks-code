# Code: Artificial collectives of specialists and generalists excel at different tasks

This repository contains code in support of the following paper:
Meluso, J., Hébert-Dufresne, L., Riedl, C., & Gao, H. O. (2026). Artificial collectives of specialists and generalists excel at different tasks. arXiv. https://arxiv.org/abs/2606.20877

The accompanying data can be found in the following archive:
Meluso, J., Hébert-Dufresne, L., Riedl, C., & Gao, H. O. (2026). Dataset: Artificial collectives of specialists and generalists excel at different tasks [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19682737

## Overview

The code simulates multi-agent systems in which agents are modeled abstractly as optimizers that iteratively search constrained state spaces. Tasks are represented by objective functions that map agents' collective states to performance, varying along four difficulty qualities (generate, choose, coordinate, and negotiate). Ties in an interpretive network describe who can effectively interpret whom. Each multi-agent system seeks to maximize performance on one of 30 collective tasks, and convergence performance is measured across repeated trials. A study systematically varies group size (2, 4, 8, and 16 agents, with equal computational resources of 32 decision variables across group sizes), interpretive network topology (18 topologies), and agent rationality bounds (searchable fraction of the decision variable domain of ±0.1%, 1%, 10%, or 100% per time step). The analyses use multivariate ordinary least squares regressions with robust standard errors to isolate the performance effects of each network property while controlling for task qualities and agent rationality bounds.

Three entry points run the full pipeline:

1. `simulation.py` runs a study and writes raw trial results.
2. `analysis.py` builds analysis datasets from raw results and runs all statistical analyses.
3. `plots.py` generates the figures from saved analysis results.

Supporting packages: `config/` (study definitions and parameter factories), `models/` (agents, teams, network generators, objective functions, optimizers), `runners/` (study execution and storage), `analysis/` (dataset creation and statistics), `figures/` (plotting), `cluster/` (HPC helper scripts), `tests/` (pytest suite and validation notebooks).

## 1. System requirements

- **Operating systems:** macOS or Linux. Developed and tested on macOS (Apple Silicon); the demo was also verified on Ubuntu 22.04.
- **Python:** 3.12 (the demo was also verified on 3.10 with the same pinned dependencies).
- **Dependencies** (pinned in `environment.yml` and `requirements.txt`): numpy 1.26.4, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.6.1, networkx 3.3, matplotlib 3.9.2, seaborn 0.13.2, h5py 3.12.1, pyarrow 17.0.0, dask 2024.8.2, dask-ml 2024.4.4, psutil 5.9.0, pytest 7.4.4.
- **Hardware:** no non-standard hardware is required. The demo runs on a normal desktop or laptop. The full-scale studies reported in the paper (~1.7 million trials per optimizer) were run on a computing cluster using `cluster/run_study.sh`; the demo does not require a cluster.
- Figures use the Times New Roman font when it is installed (as in the paper); otherwise matplotlib's default font is used.

## 2. Installation guide

**No local setup (GitHub Codespaces).** Open the repository in a GitHub Codespace (Code → Codespaces → Create codespace, or https://codespaces.new/meluso/mas-interpretive-networks-code). The included `.devcontainer` installs all pinned dependencies automatically during the container build; wait for setup to finish before using the terminal. Then run the demo with `bash run_demo.sh`. (If you ever see a `ModuleNotFoundError` on a fresh Codespace, the automatic install had not finished; `run_demo.sh` will install the requirements itself and continue.)

**Local install.** With conda (recommended, reproduces the environment used for the paper):

```bash
conda env create -f environment.yml
conda activate aitt
```

Or with pip (Python 3.10–3.12):

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Typical install time on a normal desktop computer: about 1 minute with pip (measured), a few minutes with conda.

## 3. Demo

The demo simulates its own small example dataset and runs the complete pipeline: simulation, analysis, and figure generation. With the environment active, run everything with one command from the repository root:

```bash
bash run_demo.sh
```

The script runs the three pipeline steps in order and prints where the outputs land. To run the steps individually instead:

```bash
# Step 1: simulate the demo study (writes data/raw/demo_<timestamp>/trials/)
python simulation.py --study_name demo

# Step 2: run all analyses (use the directory name created by step 1;
#         you can find it with: ls data/raw)
python analysis.py --study_name demo_<timestamp>

# Step 3: generate the figures
python plots.py --study_name demo_<timestamp>
```

The demo study runs 9,120 trials with the random-walk optimizer and 2 replications per cell: groups of 8 and 16 agents across the full grid of 18 network topologies, 30 collective tasks, and all four rationality bounds, plus 2-agent baseline groups on their two admissible topologies (complete and empty).

**Expected output:**

- `data/raw/demo_<timestamp>/trials/` — raw trial results (39 parquet files, 9,120 rows total).
- `data/preprocessed/demo_<timestamp>/` — `dataset1.parquet`, `dataset2.parquet`.
- `data/results/demo_<timestamp>/` — `mean_differences.csv` and `network_properties/`, `network_property_effects/`, `ols_regressions/` directories.
- `figures/publication/png/` and `figures/publication/pdf/` — `joint_task_effects_demo` and `performance_means_demo_team8`.

Reference copies of the two demo figures are provided in `demo/expected_output/`. Your figures should match their structure (panel layout, axes, populated effect estimates). Exact values vary between runs because trials are stochastic, and demo values differ from the paper's figures because the demo uses 2 replications and the random-walk optimizer rather than 250 replications across four optimizers.

**Expected runtime** (measured in GitHub Codespaces): about 16 minutes for the simulation on a 4-core machine, or about 36 minutes on the default 2-core machine; the analysis takes under a minute and the figures about 10 seconds.

## 4. Instructions for use

**Running your own simulations.** Studies are defined in `config/studies.py`, which maps a study name to a list of campaigns. Each campaign is a parameter factory in `config/factory_implementations.py` that sets the parameter grid: team sizes, network topologies (`config/defaults.py: graph2opts`), objective functions (`fn2opts`), rationality bounds (`agent_steplim`), optimizer (`agent_optim_type`), replications (`num_trials`), and steps per trial (`num_steps`). To create a new study, add a factory and register the study name, then run:

```bash
python simulation.py --study_name <your_study> --processes <n>
```

`--processes` defaults to all available cores. Results are written to `data/raw/<your_study>_<timestamp>/trials/`.

**Important note for multi-run studies.** Every invocation of `simulation.py` creates a new timestamped output directory, but `analysis.py` reads exactly one directory. If you split a study across multiple invocations (for example, on a cluster), merge all `trials/*.parquet` files into a single `data/raw/<study>_<timestamp>/trials/` directory before running the analysis, renaming files as needed to avoid `batch_*.parquet` name collisions.

**Analyzing and plotting.** `analysis.py --study_name <raw_directory_name>` runs the full statistical pipeline; `plots.py --study_name <raw_directory_name>` generates the figures. Without arguments, both default to the paper's four studies (see section 5).

**Cluster execution.** `cluster/run_study.sh <study_name>` launches a study with monitoring on an HPC system.

## 5. Reproducing the paper's results

To reproduce all quantitative results and figures in the paper, download the archived dataset from Zenodo (https://doi.org/10.5281/zenodo.19682737), and unzip it into a directory called `data` in the repository root. Then run:

```bash
python analysis.py   # defaults to the four paper studies
python plots.py      # defaults to main + supplement figures
```

To reproduce the raw data itself, run `simulation.py` with study names `aiteams01nm`, `aiteams01lb`, `aiteams01rw`, and `aiteams01da` (about 1.7 million trials each for the first three and 342,000 for the fourth; this requires cluster-scale resources, and results land in new timestamped directories per the note in section 4).

## License

This code is released under the MIT License (see `LICENSE`).
