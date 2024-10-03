import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from time import time


def create_four_peak_problem(size: int, t_pct: float) -> mlrose.DiscreteOpt:
    fitness = mlrose.FourPeaks(t_pct=t_pct)
    problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)
    return problem


def run_algorithm(problem: mlrose.DiscreteOpt, algorithm: str, **kwargs) -> Dict[str, Any]:
    start_time = time()

    if algorithm == 'RHC':
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, random_state=17, curve=True,
                                                                           **kwargs)
    elif algorithm == 'SA':
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, random_state=17, curve=True,
                                                                             **kwargs)
    elif algorithm == 'GA':
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, random_state=17, curve=True, **kwargs)
    elif algorithm == 'MIMIC':
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem, random_state=17, curve=True, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    end_time = time()
    execution_time = end_time - start_time

    return {
        'best_state': best_state,
        'best_fitness': best_fitness,
        'fitness_curve': fitness_curve,
        'execution_time': execution_time
    }


def run_4peak_experiment(problem_sizes: List[int], t_pct: float,
                         hyperparameters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    results = []
    algorithms = ['RHC', 'SA', 'GA', 'MIMIC']

    for size in problem_sizes:
        print(f'Running experiments for problem size {size}')
        problem = create_four_peak_problem(size, t_pct)

        for alg_name in algorithms:
            alg_results = run_algorithm(problem, alg_name, **hyperparameters[alg_name])
            results.append({
                'problem_size': size,
                'algorithm': alg_name,
                'best_state': alg_results['best_state'].tolist(),
                'best_fitness': alg_results['best_fitness'],
                'fitness_curve': alg_results['fitness_curve'].tolist(),
                'execution_time': alg_results['execution_time']
            })

    return pd.DataFrame(results)