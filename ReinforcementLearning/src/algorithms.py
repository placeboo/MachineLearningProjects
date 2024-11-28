import random

import numpy as np
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.test_env import TestEnv
from src.utils.test_env import test_env
from typing import Dict, Tuple, List
from time import time
import itertools


def run_value_iteration(env,
                        gamma: float=1.0,
                        theta: float=1e-10,
                        n_iters: int=1000,
                        test_iters: int=200,
                        random_seed:int = 42) -> Dict:
    start_time = time()
    V, V_track, pi = Planner(env.P).value_iteration(
        gamma=gamma,
        n_iters=n_iters,
        theta=theta
    )
    runtime = time() - start_time

    # calculate coverage metrics
    mean_values = np.mean(V_track, axis=1)
    # trim all the zeros in mean values (if converged, V are all zeros after certain iterations)
    mean_values = np.trim_zeros(mean_values, trim='b')
    max_values = np.max(V_track, axis=1)
    max_values = np.trim_zeros(max_values, trim='b')
    delta_values = np.abs(np.diff(mean_values))

    # test policy
    episode_rewards = test_env(env, n_iters=test_iters, pi=pi, seed=random_seed)

    return {
        'runtime': runtime,
        'converged_iter': len(mean_values),
        'final_delta': delta_values[-1] if len(delta_values) > 0 else None,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'V': V,
        'V_track': V_track,
        'pi': pi,
        'mean_values': mean_values,
        'max_values': max_values,
        'delta_values': delta_values,
        'episode_rewards': episode_rewards,
        'gamma': gamma,
        'theta': theta,
        'n_iters': n_iters
    }

def vi_grid_search(env,
                   params: Dict={},
                   verbose=True,
                   test_iters: int = 200,
                   random_seed: int = 42) -> Tuple[Dict, float, List[Dict]]:
    """
    Hyperparameter tunning for value iteration
    Args:
        env:
        params: {'gamma': np.ndarray,
                'theta': np.ndarray,
                'n_iters': np.ndarray}
    Returns:
        best params, highest_avg_reward, the return from run_value_iterations for every combination
    """
    gamma = params.get('gamma', [1.0])
    theta = params.get('theta', [1e-10])
    n_iters = params.get('n_iters', [50])

    highest_avg_reward = -np.inf
    best_params = None
    iteration_results = []

    for i in itertools.product(gamma, n_iters, theta):
        param_i = {
            'gamma': i[0],
            'n_iters': i[1],
            'theta': i[2]
        }
        if verbose:
            print(f"running VI with gamma: {i[0]}; n_iters: {i[1]}; theta: {i[2]}")

        result_i = run_value_iteration(env, **param_i, test_iters=test_iters, random_seed=random_seed)
        iteration_results.append(result_i)

        if result_i['mean_reward'] > highest_avg_reward:
            highest_avg_reward = result_i['mean_reward']
            best_params = param_i

        if verbose:
            print("Average. episode reward: ", result_i['mean_reward'])
            print('-' * 50)

    return best_params, highest_avg_reward, iteration_results

def run_policy_iteration(env,
                         gamma=1.0,
                         theta=1e-10,
                         n_iters=50,
                         test_iters: int=200,
                         random_seed:int = 42) -> Dict:
    start_time = time()
    np.random.seed(random_seed)
    random.seed(random_seed)
    V, V_track, pi = Planner(env.P).policy_iteration(
        gamma=gamma,
        n_iters=n_iters,
        theta=theta
    )
    runtime = time() - start_time

    # calculate coverage metrics
    mean_values = np.mean(V_track, axis=1)
    # trim all the zeros in mean values (if converged, V are all zeros after certain iterations)
    mean_values = np.trim_zeros(mean_values, trim='b')
    max_values = np.max(V_track, axis=1)
    max_values = np.trim_zeros(max_values, trim='b')
    delta_values = np.abs(np.diff(mean_values))

    # test policy
    # env.reset(seed=random_seed)
    # episode_rewards = TestEnv.test_env(env, n_iters=test_iters, pi=pi)
    episode_rewards = test_env(env, n_iters=test_iters, pi=pi, seed=random_seed)

    return {
        'runtime': runtime,
        'converged_iter': len(mean_values),
        'final_delta': delta_values[-1] if len(delta_values) > 0 else None,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'V': V,
        'V_track': V_track,
        'pi': pi,
        'mean_values': mean_values,
        'max_values': max_values,
        'delta_values': delta_values,
        'episode_rewards': episode_rewards,
        'gamma': gamma,
        'theta': theta,
        'n_iters': n_iters
    }

def pi_grid_search(env,
                   params: Dict={},
                   test_iters: int=200,
                   verbose=True,
                   random_seed: int=42) -> Tuple[Dict, float, List[Dict]]:
    """
    Hyperparameter tunning for policy iteration
    Args:
        env:
        params: {'gamma': np.ndarray,
                'theta': np.ndarray,
                'n_iters': np.ndarray}
    Returns:
        best params, highest_avg_reward, the return from run_value_iterations for every combination
    """
    gamma = params.get('gamma', [1.0])
    theta = params.get('theta', [1e-10])
    n_iters = params.get('n_iters', [50])

    highest_avg_reward = -np.inf
    best_params = None
    iteration_results = []

    for i in itertools.product(gamma, n_iters, theta):
        param_i = {
            'gamma': i[0],
            'n_iters': i[1],
            'theta': i[2]
        }
        if verbose:
            print(f"running PI with gamma: {i[0]}; n_iters: {i[1]}; theta: {i[2]}")

        result_i = run_policy_iteration(env, **param_i, test_iters=test_iters, random_seed=random_seed)
        iteration_results.append(result_i)

        if result_i['mean_reward'] > highest_avg_reward:
            highest_avg_reward = result_i['mean_reward']
            best_params = param_i

        if verbose:
            print("Average. episode reward: ", result_i['mean_reward'])
            print('-' * 50)

    return best_params, highest_avg_reward, iteration_results

def run_q_learning(env,
                   gamma: float=0.99,
                   init_alpha: float=0.5,
                   min_alpha: float=0.01,
                   alpha_decay_ratio: float=0.5,
                   init_epsilon: float=1.0,
                   min_epsilon: float=0.1,
                   epsilon_decay_ratio: float=0.9,
                   n_episodes: int=10000,
                   test_iters: int=200,
                   random_seed: int=42) -> Dict:
    start_time = time()
    env.reset(seed=random_seed)
    Q, V, pi, Q_track, pi_track = RL(env).q_learning(
        gamma=gamma,
        init_alpha=init_alpha,
        min_alpha=min_alpha,
        alpha_decay_ratio=alpha_decay_ratio,
        init_epsilon=init_epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay_ratio=epsilon_decay_ratio,
        n_episodes=n_episodes
    )
    runtime = time() - start_time

    # calculate coverage metrics
    mean_values = np.max(Q_track, axis=2)
    mean_values = np.mean(mean_values, axis=1)
    max_values = np.max(Q_track, axis=(1, 2))
    delta_values = np.abs(np.diff(mean_values))


    # test policy
    episode_rewards = test_env(env, n_iters=test_iters, pi=pi, seed=random_seed)

    return {
        'runtime': runtime,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'Q': Q,
        'V': V,
        'pi': pi,
        #'Q_track': Q_track,
        'mean_values': mean_values,
        'max_values': max_values,
        'delta_values': delta_values,
        #'pi_track': pi_track,
        #'episode_rewards': episode_rewards,
        'gamma': gamma,
        'init_alpha': init_alpha,
        'min_alpha': min_alpha,
        'alpha_decay_ratio': alpha_decay_ratio,
        'init_epsilon': init_epsilon,
        'min_epsilon': min_epsilon,
        'epsilon_decay_ratio': epsilon_decay_ratio,
        'n_episodes': n_episodes
    }

def q_learning_grid_search(env,
                           params: Dict={},
                           verbose=True,
                           test_iters: int=200,
                           random_seed: int= 42) -> Tuple[Dict, float, List[Dict]]:
    """
    Hyperparameter tunning for q-learning
    Args:
        env:
        params: {'gamma': np.ndarray,
                'init_alpha': np.ndarray,
                'min_alpha': np.ndarray,
                'alpha_decay_ratio': np.ndarray,
                'init_epsilon': np.ndarray,
                'min_epsilon': np.ndarray,
                'epsilon_decay_ratio': np.ndarray,
                'n_episodes': np.ndarray}
    Returns:
        best params, highest_avg_reward, the return from run_value_iterations for every combination
    """
    gamma = params.get('gamma', [0.99])
    init_alpha = params.get('init_alpha', [0.5])
    min_alpha = params.get('min_alpha', [0.01])
    alpha_decay_ratio = params.get('alpha_decay_ratio', [0.5])
    init_epsilon = params.get('init_epsilon', [1.0])
    min_epsilon = params.get('min_epsilon', [0.1])
    epsilon_decay_ratio = params.get('epsilon_decay_ratio', [0.9])
    n_episodes = params.get('n_episodes', [10000])

    highest_avg_reward = -np.inf
    best_params = None
    iteration_results = []

    for i in itertools.product(gamma, init_alpha, min_alpha, alpha_decay_ratio, init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes):
        param_i = {
            'gamma': i[0],
            'init_alpha': i[1],
            'min_alpha': i[2],
            'alpha_decay_ratio': i[3],
            'init_epsilon': i[4],
            'min_epsilon': i[5],
            'epsilon_decay_ratio': i[6],
            'n_episodes': i[7]
        }
        if verbose:
            print(f"running q-learning with gamma: {i[0]}; init_alpha: {i[1]}; min_alpha: {i[2]}; alpha_decay_ratio: {i[3]}; init_epsilon: {i[4]}; min_epsilon: {i[5]}; epsilon_decay_ratio: {i[6]}; n_episodes: {i[7]}")

        result_i = run_q_learning(env, **param_i, test_iters=test_iters, random_seed=random_seed)
        iteration_results.append(result_i)

        if result_i['mean_reward'] > highest_avg_reward:
            highest_avg_reward = result_i['mean_reward']
            best_params = param_i

        if verbose:
            print("Average. episode reward: ", result_i['mean_reward'])
            print('-' * 50)

    return best_params, highest_avg_reward, iteration_results


