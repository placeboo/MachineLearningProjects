# Markov Decision Processes Analysis

This project implements and analyzes various Markov Decision Process (MDP) algorithms, including Value Iteration, Policy Iteration, Q-Learning, and SARSA, on two distinct environments: Blackjack and CartPole.

## Project Structure

```
├── results/               # Stores experiment results
├── figs/                 # Generated figures and plots
├── environment.yml       # Conda environment specification
├── notebooks/           # Jupyter notebooks for experiments
│   ├── blackjack.ipynb  # Blackjack environment experiments
│   ├── cartpole.ipynb   # CartPole environment experiments
├── src/                 # Source code
    ├── algorithms.py    # Core algorithm implementations
    ├── experiments/     # Experiment runners
    │   ├── blackjack_exp.py
    │   └── cartpole_exp.py
    └── utils/           # Utility functions
        ├── logging.py   # Logging setup
        └── plotting.py  # Visualization utilities
```

## Setup Instructions

1. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate mdp
```

## Running Experiments

The experiments are organized in Jupyter notebooks under the `notebooks/` directory.

### Blackjack Environment

Run the experiments in `notebooks/blackjack.ipynb`:

1. Value Iteration and Policy Iteration grid search
2. Q-Learning grid search
3. SARSA grid search
4. Results analysis and visualization

### CartPole Environment

Run the experiments in `notebooks/cartpole.ipynb`:

1. Value Iteration and Policy Iteration grid search with different state discretizations
2. Q-Learning grid search with varying parameters
3. SARSA grid search with varying parameters
4. Results analysis and visualization

## Code Organization

- `src/algorithms.py`: Implements the core MDP algorithms (Value Iteration, Policy Iteration, Q-Learning, SARSA)
- `src/experiments/`: Contains experiment runner classes for each environment
- `src/utils/`: Helper functions for logging, plotting, and testing


## Results

Results are saved in two locations:
- `results/`: Contains pickle files with raw experiment data
- `figs/`: Contains generated plots and visualizations


## Notes

- The CartPole environment is discretized using bins to enable value/policy iteration
- Experiments include parameter grid searches for optimal performance
- Results include runtime analysis, reward comparisons, and convergence plots