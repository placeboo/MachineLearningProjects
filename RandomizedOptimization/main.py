import mlrose_ky as mlrose
from src.optimization import run_4peak_experiment, run_queues_experiment
from src.utils import save_results, plot_performance_vs_size, plot_iterations_vs_time, plot_fitness_vs_iteration
import numpy as np

def main(name, problem_name: str, results_dir: str, fig_dir: str, hyperparameters: dict, **kwargs):

    if name == 'four_peaks':
        problem_sizes = kwargs['problem_sizes']
        t_pct = kwargs['t_pct']
    # Run the experiment and get results as a DataFrame
        results_df = run_4peak_experiment(problem_sizes, t_pct, hyperparameters)
    elif name == 'queens':
        problem_sizes = kwargs['problem_sizes']
        results_df = run_queues_experiment(problem_sizes, hyperparameters)
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")
    # Save the results
    save_results(results_df, problem_name, results_dir)
    print(f'Experiment results saved to {results_dir}/{problem_name}_results.csv')

    # Create plots
    plot_performance_vs_size(results_df, 'best_fitness', fig_dir)
    plot_performance_vs_size(results_df, 'execution_time', fig_dir)
    plot_iterations_vs_time(results_df, fig_dir)
    plot_fitness_vs_iteration(results_df, fig_dir)


if __name__ == '__main__':
    max_exp = 10

    # case 1: regular hyperparameters
    # hyperparameters = {
    #     'RHC': {'restarts': 10, 'max_iters': 1000, 'random_state': 17},
    #     'SA': {'schedule': mlrose.ExpDecay(), 'max_attempts': max_exp, 'max_iters': 1000, 'random_state': 17},
    #     'GA': {'pop_size': 200, 'mutation_prob': 0.1, 'max_attempts': max_exp, 'max_iters': 1000, 'random_state': 17},
    #     'MIMIC': {'pop_size': 200, 'keep_pct': 0.2, 'max_attempts': max_exp, 'max_iters': 1000, 'random_state': 17}
    # }
    # problem_names = ['four_peaks', 'queens']
    # problem_sizes = np.arange(10, 110, 5)
    # t_pcts = [0.2]
    # queen_sizes = np.arange(10, 110, 5)
    # suffix = ''
    # # case 2: SA with different schedule
    # hyperparameters = {
    #     'SA': {'schedule': mlrose.GeomDecay(), 'max_attempts': max_exp, 'max_iters': 1000, 'random_state': 17},
    # }
    # problem_names = ['four_peaks', 'queens']
    # problem_sizes = np.arange(10, 110, 5)
    # t_pcts = [0.2]
    # queen_sizes = np.arange(10, 110, 5)
    # suffix = '_geom_decay'

    # case 3: GA with different pop_size
    hyperparameters = {
        'GA': {'pop_size': 400, 'mutation_prob': 0.1, 'max_attempts': max_exp, 'max_iters': 1000, 'random_state': 17},
    }
    problem_names = ['four_peaks', 'queens']
    problem_sizes = np.arange(10, 110, 5)
    t_pcts = [0.2]
    queen_sizes = np.arange(10, 110, 5)
    suffix = '_pop_400'

    for name in problem_names:
        if name == 'four_peaks':
            for t in t_pcts:
                print(f'Running experiments for t_pct = {t}')
                results_dir = f'results/{name}_t_{t}_exp_{max_exp}{suffix}'
                fig_dir = f'figures/{name}_t_{t}_exp_{max_exp}{suffix}'
                main(name, name, results_dir, fig_dir, hyperparameters, problem_sizes=problem_sizes, t_pct=t)
        elif name == 'queens':
            results_dir = f'results/{name}_exp_{max_exp}{suffix}'
            fig_dir = f'figures/{name}_exp_{max_exp}{suffix}'
            main(name, name, results_dir, fig_dir, hyperparameters, problem_sizes=queen_sizes)
        else:
            raise ValueError(f"Unknown problem name: {name}")