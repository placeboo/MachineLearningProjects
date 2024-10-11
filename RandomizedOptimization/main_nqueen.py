import mlrose_ky as mlrose
import numpy as np
from src.optimization import run_queues_experiment
from src.utils import save_results, plot_performance_vs_size, plot_iterations_vs_time, plot_fitness_vs_iteration
import os
import pandas as pd
import itertools


def run_size_experiment(problem_sizes, hyperparameters, results_dir, fig_dir):
    results_df = run_queues_experiment(problem_sizes, hyperparameters)

    # Save the results
    save_results(results_df, 'nqueens_size', results_dir)
    print(f'Size experiment results saved to {results_dir}/nqueens_size_results.csv')

    # Create plots
    plot_performance_vs_size(results_df, 'best_fitness', fig_dir)
    plot_performance_vs_size(results_df, 'execution_time', fig_dir)
    plot_iterations_vs_time(results_df, fig_dir)
    plot_fitness_vs_iteration(results_df, fig_dir)


def run_hyperparameter_experiment(size, hyperparameters, num_runs, results_dir, fig_dir):
    all_results = []

    for alg, params in hyperparameters.items():
        param_names = list(params.keys())
        param_values = list(params.values())

        # Generate all combinations of parameter values
        param_combinations = list(itertools.product(*param_values))

        for combination in param_combinations:
            current_params = hyperparameters[alg].copy()
            param_dict = dict(zip(param_names, combination))
            current_params.update(param_dict)

            param_str = ", ".join([f"{name}={value}" for name, value in param_dict.items()])
            print(f"Running {alg} with {param_str}")

            for run in range(num_runs):
                results_df = run_queues_experiment([size], {alg: current_params})
                results_df['run'] = run
                for name, value in param_dict.items():
                    results_df[f'param_{name}'] = value
                all_results.append(results_df)

    combined_results = pd.concat(all_results, ignore_index=True)

    # Save the results
    save_results(combined_results, f'nqueens_hyperparameter_size_{size}', results_dir)
    print(f'Hyperparameter experiment results saved to {results_dir}/nqueens_hyperparameter_size_{size}_results.csv')

    # Create plots (you might want to create custom plots for hyperparameter analysis)
    plot_hyperparameter_performance(combined_results, fig_dir)


def plot_hyperparameter_performance(results, fig_dir):
    # Implement custom plotting for hyperparameter analysis
    # This is a placeholder - you should implement this based on your specific needs
    pass


def main():
    max_attempts = 100
    max_iters = 1000
    random_state = 42

    # Define problem sizes for the size experiment
    problem_sizes = np.arange(8, 128, 8)

    # Define hyperparameters for the size experiment
    size_experiment_hyperparameters = {
        'RHC': {'restarts': 10, 'max_iters': max_iters, 'max_attempts': max_attempts, 'random_state': random_state},
        'SA': {'schedule': mlrose.ExpDecay(), 'max_attempts': max_attempts, 'max_iters': max_iters,
               'random_state': random_state},
        'GA': {'pop_size': 200, 'mutation_prob': 0.1, 'max_attempts': max_attempts, 'max_iters': max_iters,
               'random_state': random_state},
    }

    # Run size experiment
    results_dir = 'results/nqueens2/size_experiment'
    fig_dir = 'figures/nqueens2/size_experiment'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    #run_size_experiment(problem_sizes, size_experiment_hyperparameters, results_dir, fig_dir)

    # Define hyperparameters for the hyperparameter experiment
    fixed_size = 56  # You can change this to your preferred size
    hyperparameter_experiment_hyperparameters = {
        'RHC': {
            'restarts': [5, 10, 20],
            'max_attempts': [max_attempts],
            'max_iters': [max_iters],
            'random_state': [random_state]
        },
        'SA': {
            'schedule': [mlrose.ArithDecay(), mlrose.GeomDecay(), mlrose.ExpDecay()],
            'max_attempts': [10, 20, 50, 100],
            'max_iters': [max_iters],
            'random_state': [random_state]
        },
        'GA': {
            'pop_size': [100, 200, 400],
            'mutation_prob': [0.1, 0.2, 0.3, 0.4],
            'max_iters': [max_iters],
            'random_state': [random_state],
            'max_iters': [max_iters],
        },
    }

    # Run hyperparameter experiment
    results_dir = f'results/nqueens/hyperparameter_experiment_size_{fixed_size}'
    fig_dir = f'figures/nqueens/hyperparameter_experiment_size_{fixed_size}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    num_runs = 1  # Number of runs for each hyperparameter configuration
    run_hyperparameter_experiment(fixed_size, hyperparameter_experiment_hyperparameters, num_runs, results_dir, fig_dir)


if __name__ == '__main__':
    main()