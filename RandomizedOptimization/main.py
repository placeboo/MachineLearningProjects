import mlrose_ky as mlrose
from src.optimization import run_4peak_experiment
from src.utils import save_results, plot_performance_vs_size, plot_convergence_vs_size
import numpy as np


def main(problem_sizes, t_pct, problem_name: str, results_dir: str, fig_dir: str):

    hyperparameters = {
        'RHC': {'max_attempts': 100, 'max_iters': 1000},
        'SA': {'schedule': mlrose.ExpDecay(), 'max_attempts': 100, 'max_iters': 1000},
        'GA': {'pop_size': 200, 'mutation_prob': 0.1, 'max_attempts': 100, 'max_iters': 1000},
        'MIMIC': {'pop_size': 200, 'keep_pct': 0.2, 'max_attempts': 100, 'max_iters': 1000}
    }

    # Run the experiment and get results as a DataFrame
    results_df = run_4peak_experiment(problem_sizes, t_pct, hyperparameters)

    # Save the results
    save_results(results_df, problem_name, results_dir)
    print(f'Experiment results saved to {results_dir}/{problem_name}_results.csv')

    # Create plots
    plot_performance_vs_size(results_df, 'best_fitness', fig_dir)
    plot_performance_vs_size(results_df, 'execution_time', fig_dir)
    plot_convergence_vs_size(results_df, fig_dir)


if __name__ == '__main__':
    problem_sizes = np.arange(10, 110, 10)
    t_pcts = [0.1, 0.2, 0.3, 0.4, 0.5]

    for t in t_pcts:
        print(f'Running experiments for t_pct = {t}')
        results_dir = f'results/four_peaks_t_{t}'
        fig_dir = f'figures/four_peaks_t_{t}'
        main(problem_sizes, t, 'four_peaks', results_dir, fig_dir)
