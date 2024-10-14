import mlrose_ky as mlrose
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import itertools

def create_four_peak_problem(size: int, t_pct: float) -> mlrose.DiscreteOpt:
    fitness = mlrose.FourPeaks(t_pct=t_pct)
    problem = mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2)
    return problem


def create_queens_problem(size: int) -> mlrose.DiscreteOpt:
    def queens_max(state):

        # Initialize counter
        fitness_cnt = 0

        # For all pairs of queens
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):
                # Check for horizontal, diagonal-up and diagonal-down attacks
                if (state[j] != state[i]) \
                        and (state[j] != state[i] + (j - i)) \
                        and (state[j] != state[i] - (j - i)):
                    # If no attacks, then increment counter
                    fitness_cnt += 1
        return fitness_cnt

    # def queens_max(state):
    #     queens_fitness = mlrose.Queens()
    #     return -queens_fitness.evaluate(state)
    fitness =  mlrose.CustomFitness(queens_max)
    problem =  mlrose.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=size)
    return problem


def run_algorithm(problem: mlrose.DiscreteOpt, algorithm: str, **kwargs) -> Dict[str, Any]:
    start_time = time()

    if algorithm == 'RHC':
        best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, curve=True, **kwargs)
    elif algorithm == 'SA':
        best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, curve=True, **kwargs)
    elif algorithm == 'GA':
        best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, curve=True, **kwargs)
    elif algorithm == 'MIMIC':
        best_state, best_fitness, fitness_curve = mlrose.mimic(problem, curve=True, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    end_time = time()
    execution_time = end_time - start_time

    return {
        'best_state': best_state,
        'best_fitness': best_fitness,
        'fitness_curve': fitness_curve,
        'execution_time': execution_time,
        'iterations': len(fitness_curve)
    }

def run_4peak_experiment(problem_sizes: List[int], t_pct: float,
                         hyperparameters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    results = []
    algorithms = ['RHC', 'SA', 'GA', 'MIMIC']
    required_algorithms = list(hyperparameters.keys())
    for size in problem_sizes:
        print(f'Running experiments for problem size {size}')
        problem = create_four_peak_problem(size, t_pct)

        for required_algorithm in required_algorithms:
            if required_algorithm not in algorithms:
                raise ValueError(f"Unknown algorithm: {required_algorithm}")
            else:
                alg_results = run_algorithm(problem, required_algorithm, **hyperparameters[required_algorithm])
                results.append({
                    'problem_size': size,
                    'algorithm': required_algorithm,
                    'best_fitness': alg_results['best_fitness'],
                    'execution_time': alg_results['execution_time'],
                    'fitness_curve': alg_results['fitness_curve'].tolist(),
                    'iterations': alg_results['iterations']
                })
        # for alg_name in algorithms:
        #     alg_results = run_algorithm(problem, alg_name, **hyperparameters[alg_name])
        #     results.append({
        #         'problem_size': size,
        #         'algorithm': alg_name,
        #         'best_fitness': alg_results['best_fitness'],
        #         'execution_time': alg_results['execution_time'],
        #         'fitness_curve': alg_results['fitness_curve'].tolist(),
        #         'iterations': alg_results['iterations']
        #     })

    return pd.DataFrame(results)

def run_queues_experiment(problem_sizes: List[int],
                         hyperparameters: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    results = []
    algorithms = ['RHC', 'SA', 'GA', 'MIMIC']
    required_algorithms = list(hyperparameters.keys())
    for size in problem_sizes:
        print(f'Running experiments for problem size {size}')
        problem = create_queens_problem(size)

        for required_algorithm in required_algorithms:
            if required_algorithm not in algorithms:
                raise ValueError(f"Unknown algorithm: {required_algorithm}")
            else:
                alg_results = run_algorithm(problem, required_algorithm, **hyperparameters[required_algorithm])
                results.append({
                    'problem_size': size,
                    'algorithm': required_algorithm,
                    'best_fitness': alg_results['best_fitness'],
                    'execution_time': alg_results['execution_time'],
                    'fitness_curve': alg_results['fitness_curve'].tolist(),
                    'iterations': alg_results['iterations']
                })

    return pd.DataFrame(results)


def run_nn_experiment(X_train, y_train, X_test, y_test, hidden_nodes: List[int],
                      algorithms: List[str], hyperparameters: Dict[str, Dict[str, Any]],
                      val_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    results = []
    best_models = {}

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    for alg in algorithms:
        print(f'Running experiments for algorithm {alg}')
        alg_params = hyperparameters[alg]
        best_val_score = -np.inf
        best_model = None

        for param_combination in get_param_combinations(alg_params):
            start_time = time()
            kwargs = param_combination.copy()
            kwargs['random_state'] = random_state

            nn_model = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu',
                                            algorithm=alg, bias=True, is_classifier=True,
                                            learning_rate=0.06, early_stopping=True,
                                            max_iters=1000,
                                            clip_max=5, curve=True, **kwargs)

            nn_model.fit(X_train, y_train)

            end_time = time()
            execution_time = end_time - start_time

            # Predictions
            train_pred = nn_model.predict(X_train)
            val_pred = nn_model.predict(X_val)
            test_pred = nn_model.predict(X_test)

            # Calculate metrics
            train_metrics = calculate_metrics(y_train, train_pred)
            val_metrics = calculate_metrics(y_val, val_pred)
            test_metrics = calculate_metrics(y_test, test_pred)

            # fitness curve, only the first element
            fitness_values = nn_model.fitness_curve.tolist()

            results.append({
                'algorithm': alg,
                'params': str(param_combination),
                'train_accuracy': train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_f1': train_metrics['f1'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1': test_metrics['f1'],
                'fitness_curve': fitness_values,
                'execution_time': execution_time,
                'iterations': len(nn_model.fitness_curve)
            })

            # Update best model if validation score is higher
            if val_metrics['accuracy'] > best_val_score:
                print(f'New best model found for {alg} with params {param_combination}')
                best_val_score = val_metrics['accuracy']
                best_model = (nn_model, param_combination, train_metrics, val_metrics, test_metrics, fitness_values, execution_time)

        # Store best model for the algorithm
        best_models[alg] = best_model

    return pd.DataFrame(results), best_models


def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def get_param_combinations(params):
    keys = params.keys()
    values = params.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))