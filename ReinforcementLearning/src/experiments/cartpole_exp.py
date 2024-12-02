import itertools
import gymnasium as gym
import pandas as pd
import typing as Dict
import os
import numpy as np
import pickle

from jsonschema.exceptions import best_match
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count

plt.ioff()

from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from src.algorithms import vi_grid_search, pi_grid_search, q_learning_grid_search, sarsa_learning_grid_search
from src.utils.plotting import create_param_heatmap, save_plot, create_v_iters_plot
from src.utils.logging import setup_logging


class CartPoleExperiment:
    def __init__(self,
                 env_name: str,
                 result_dir: str,
                 fig_dir: str,
                 random_seed: int):
        self.env_name = env_name
        self.result_dir = os.path.join(result_dir, env_name)
        self.fig_dir = os.path.join(fig_dir, env_name)
        self.random_seed = random_seed

        # create directories
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def run_vi_pi_grid_search(self,
                              params: Dict = None,
                              test_iters: int = 200,
                              verbose: bool = True,
                              log_name: str = 'log'):
        # Set up logging
        logger = setup_logging(result_dir=self.result_dir,
                               name=log_name)

        if params is None:
            params = {}

        # log the input parameters
        logger.info(f'Running {self.env_name} Experiment')
        logger.info(f'Params: {params}')
        logger.info(f'Test iters: {test_iters}')


        # set up different bins
        position_bins = params.get('position_bins', [10])
        velocity_bins = params.get('velocity_bins', [10])
        angular_velocity_bins = params.get('angular_velocity_bins', [10])

        vi_results = []
        pi_results = []

        for position_bin, velocity_bin, angular_velocity_bin in itertools.product(position_bins, velocity_bins, angular_velocity_bins):
            logger.info('#'*50)
            logger.info(f'Running experiment with position_bin={position_bin}, velocity_bin={velocity_bin}, angular_velocity_bin={angular_velocity_bin}')
            # create env
            env = gym.make('CartPole-v1', render_mode=None)
            env = CartpoleWrapper(env,
                                  position_bins=position_bin,
                                  velocity_bins=velocity_bin,
                                  angular_velocity_bins=angular_velocity_bin)

            # ************** run VI **************
            logger.info('-' * 50)
            logger.info('Running Value Iteration')
            best_params, highest_avg_reward, iteration_results = \
                vi_grid_search(env=env,
                               params = params,
                               test_iters=test_iters,
                               verbose=verbose,
                               random_seed=self.random_seed)

            logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")
            vi_result_i = {
                'position_bin': position_bin,
                'velocity_bin': velocity_bin,
                'angular_velocity_bin': angular_velocity_bin,
                'best_params': best_params,
                'highest_avg_reward': highest_avg_reward,
                'iteration_results': iteration_results
            }

            vi_results.append(vi_result_i)
            with open(f'{self.result_dir}/vi_grid_search_results_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.pkl', 'wb') as f:
                pickle.dump(vi_result_i, f)
            logger.info(f'Value interation results saved to {self.result_dir}/vi_grid_search_results_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.pkl')

            iteration_results_df = pd.DataFrame(iteration_results)
            logger.info("Creating Value Iteration Heatmap")
            for metric in ['converged_iter', 'mean_reward', 'runtime', 'std_reward']:
                for n_iter in iteration_results_df['n_iters'].unique():
                    tmp_iteration_results_df = iteration_results_df[iteration_results_df['n_iters'] == n_iter]
                    fig, _ = create_param_heatmap(
                        data=tmp_iteration_results_df,
                        x_param='gamma',
                        y_param='theta',
                        metric=metric,
                        cmap='Greens',
                        multiplier=2,
                        annot=False
                    )
                    save_plot(fig, self.fig_dir, f'heatmap_vi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}_n_iter{n_iter}')
                    logger.info(f'Value Iteration Heatmap saved to {self.fig_dir}/heatmap_vi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}_n_iter{n_iter}.png')
                    plt.close(fig)

            # draw convergence plot
            logger.info("Creating Value Iteration Convergence Plot")
            # find the resutls with best to draw the convergence plot
            best_results = iteration_results_df[
                (iteration_results_df['gamma'] == best_params['gamma']) &
                (iteration_results_df['theta'] == best_params['theta']) &
                (iteration_results_df['n_iters'] == best_params['n_iters'])
            ]
            for metric in ['mean_values', 'max_values', 'delta_values']:
                fig, _ = create_v_iters_plot(
                    v_arr = best_results[metric].values[0],
                    y_label = metric
                )
                save_plot(fig, self.fig_dir, f'convergence_vi_{metric}_p{position_bin}_v{velocity_bin}_a{angular_velocity_bin}')
                logger.info(f'Value Iteration Convergence Plot saved to {self.fig_dir}/convergence_vi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.png')
                plt.close(fig)


            # ************** run PI **************
            logger.info('-'*50)
            logger.info('Running Policy Iteration')
            best_params, highest_avg_reward, iteration_results = \
                pi_grid_search(env=env,
                               params = params,
                               test_iters=test_iters,
                               verbose=verbose,
                               random_seed=self.random_seed)
            logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")
            pi_result_i = {
                'position_bin': position_bin,
                'velocity_bin': velocity_bin,
                'angular_velocity_bin': angular_velocity_bin,
                'best_params': best_params,
                'highest_avg_reward': highest_avg_reward,
                'iteration_results': iteration_results
            }
            pi_results.append(pi_result_i)
            with open(f'{self.result_dir}/pi_grid_search_results_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.pkl', 'wb') as f:
                pickle.dump(pi_result_i, f)
            logger.info(f'Policy interation results saved to {self.result_dir}/pi_grid_search_results_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.pkl')

            iteration_results_df = pd.DataFrame(iteration_results)
            logger.info("Creating Policy Iteration Heatmap")
            for metric in ['converged_iter', 'mean_reward', 'runtime', 'std_reward']:
                for n_iter in iteration_results_df['n_iters'].unique():
                    tmp_iteration_results_df = iteration_results_df[iteration_results_df['n_iters'] == n_iter]
                    fig, _ = create_param_heatmap(
                        data=tmp_iteration_results_df,
                        x_param='gamma',
                        y_param='theta',
                        metric=metric,
                        cmap='Greens',
                        multiplier=2,
                        annot=False
                    )
                    save_plot(fig, self.fig_dir, f'heatmap_pi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}_n_iter{n_iter}')
                    logger.info(f'Policy Iteration Heatmap saved to {self.fig_dir}/heatmap_pi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}_n_iter{n_iter}.png')
                    plt.close(fig)

            # draw convergence plot
            logger.info("Creating Policy Iteration Convergence Plot")
            # find the resutls with best to draw the convergence plot
            best_results = iteration_results_df[
                (iteration_results_df['gamma'] == best_params['gamma']) &
                (iteration_results_df['theta'] == best_params['theta']) &
                (iteration_results_df['n_iters'] == best_params['n_iters'])
            ]
            for metric in ['mean_values', 'max_values', 'delta_values']:
                fig, _ = create_v_iters_plot(
                    v_arr = best_results[metric].values[0],
                    y_label = metric
                )
                save_plot(fig, self.fig_dir, f'convergence_pi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}')
                logger.info(f'Policy Iteration Convergence Plot saved to {self.fig_dir}/convergence_pi_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.png')
                plt.close(fig)
            logger.info('#' * 50)
        #save the results
        with open(f'{self.result_dir}/vi_results.pkl', 'wb') as f:
            pickle.dump(vi_results, f)
        logger.info(f'Value Iteration results saved to {self.result_dir}/vi_results.pkl')
        with open(f'{self.result_dir}/pi_results.pkl', 'wb') as f:
            pickle.dump(pi_results, f)
        logger.info(f'Policy Iteration results saved to {self.result_dir}/pi_results.pkl')
        logger.info('Finished experiment')

    def run_q_learning_grid_search(self,
                                   state_params: Dict = None,
                                   params: Dict = None,
                                   test_iters: int = 200,
                                   verbose: bool = True,
                                   log_name: str = 'log',
                                   n_processes: int = None):
        # Set up logging
        logger = setup_logging(result_dir=self.result_dir, name=log_name)

        if state_params is None:
            state_params = {}
        if params is None:
            params = {}

        # Log input parameters
        logger.info(f'Running {self.env_name} Experiment')
        logger.info(f'Params: {params}')
        logger.info(f'Test iters: {test_iters}')

        # Set up different bins
        position_bins = state_params.get('position_bins', [10])
        velocity_bins = state_params.get('velocity_bins', [10])
        angular_velocity_bins = state_params.get('angular_velocity_bins', [10])

        # Create parameter combinations for parallel processing
        param_combinations = [
            (pos, vel, ang, params, test_iters, verbose, self.random_seed, self.result_dir, self.fig_dir, 'q')
            for pos, vel, ang in itertools.product(position_bins, velocity_bins, angular_velocity_bins)
        ]

        # Set number of processes
        if n_processes is None:
            n_processes = min(cpu_count(), len(param_combinations))

        # Run parallel processing
        logger.info(f'Starting parallel processing with {n_processes} processes')
        with Pool(processes=n_processes) as pool:
            results = pool.map(worker_function, param_combinations)

        return results

    def run_sarsa_leanring_grid_search(self,
                                       state_params: Dict = None,
                                       params: Dict = None,
                                       test_iters: int = 200,
                                       verbose: bool = True,
                                       log_name: str = 'log',
                                       n_processes: int = None):
        # Set up logging
        logger = setup_logging(result_dir=self.result_dir, name=log_name)

        if state_params is None:
            state_params = {}
        if params is None:
            params = {}

            # Log input parameters
        logger.info(f'Running {self.env_name} Experiment')
        logger.info(f'Params: {params}')
        logger.info(f'Test iters: {test_iters}')

        # Set up different bins
        position_bins = state_params.get('position_bins', [10])
        velocity_bins = state_params.get('velocity_bins', [10])
        angular_velocity_bins = state_params.get('angular_velocity_bins', [10])

        # Create parameter combinations for parallel processing
        param_combinations = [
            (pos, vel, ang, params, test_iters, verbose, self.random_seed, self.result_dir, self.fig_dir, 'sarsa')
            for pos, vel, ang in itertools.product(position_bins, velocity_bins, angular_velocity_bins)
        ]

        # Set number of processes
        if n_processes is None:
            n_processes = min(cpu_count(), len(param_combinations))

            # Run parallel processing
        logger.info(f'Starting parallel processing with {n_processes} processes')
        with Pool(processes=n_processes) as pool:
            results = pool.map(worker_function, param_combinations)

        return results


def worker_function(args):
    """
    Worker function to handle a single parameter combination for q learning
    """

    position_bin, velocity_bin, angular_velocity_bin, params, test_iters, verbose, random_seed, result_dir, fig_dir, method = args

    if method not in ['q', 'sarsa']:
        raise ValueError(f"Method {method} not supported")

    # Create logger for this worker
    logger = setup_logging(
        result_dir=result_dir,
        name=f'{method}_worker_pos{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}'
    )
    logger.info('#' * 50)
    logger.info(
        f'Running experiment with position_bin={position_bin}, velocity_bin={velocity_bin}, angular_velocity_bin={angular_velocity_bin}')

    # Create environment
    env = gym.make('CartPole-v1', render_mode=None)
    env = CartpoleWrapper(env,
                          position_bins=position_bin,
                          velocity_bins=velocity_bin,
                          angular_velocity_bins=angular_velocity_bin)

    # Run Q-Learning
    best_params = None
    highest_avg_reward = -np.inf
    iteration_results = []
    if method == 'q':
        best_params, highest_avg_reward, iteration_results = q_learning_grid_search(
            env=env,
            params=params,
            test_iters=test_iters,
            verbose=verbose,
            random_seed=random_seed
        )
    elif method == 'sarsa':
        best_params, highest_avg_reward, iteration_results = sarsa_learning_grid_search(
            env=env,
            params=params,
            test_iters=test_iters,
            verbose=verbose,
            random_seed=random_seed
        )
    else:
        raise ValueError(f"Method {method} not supported")

    logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")

    # Save results
    result = {
        'position_bin': position_bin,
        'velocity_bin': velocity_bin,
        'angular_velocity_bin': angular_velocity_bin,
        'best_params': best_params,
        'highest_avg_reward': highest_avg_reward,
        'iteration_results': iteration_results
    }

    # Save to pickle
    result_path = f'{result_dir}/{method}_grid_search_results_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Results saved to {result_path}")

    # Create visualizations if params exist
    if params:
        param_keys = list(params.keys())
        iteration_results_df = pd.DataFrame(iteration_results)

        if len(param_keys) >= 2:
            for pair in itertools.combinations(param_keys, 2):
                logger.info(f'Creating heatmap for {pair[0]} and {pair[1]}')
                rest_params = param_keys.copy()
                rest_params.remove(pair[0])
                rest_params.remove(pair[1])

                tmp_result_df = iteration_results_df.copy()
                for param in rest_params:
                    tmp_result_df = tmp_result_df[tmp_result_df[param] == best_params[param]]

                # Create heatmaps
                for metric in ['mean_reward', 'runtime', 'std_reward']:
                    fig, _ = create_param_heatmap(
                        data=tmp_result_df,
                        x_param=pair[0],
                        y_param=pair[1],
                        metric=metric,
                        cmap='Greens',
                        multiplier=1.5,
                        annot=False
                    )
                    save_plot(fig, fig_dir,
                              f'heatmap_{method}_{metric}_{pair[0]}_{pair[1]}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}')
                    logger.info(f'Heatmap for {pair[0]} and {pair[1]} saved in {fig_dir}/heatmap_{method}_{metric}_{pair[0]}_{pair[1]}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}')
                    plt.close(fig)

        # Fixed: Create convergence plots with proper DataFrame filtering
        logger.info('Creating convergence plots')
        best_results = iteration_results_df.copy()
        for param, value in best_params.items():
            best_results = best_results[best_results[param] == value]

        if not best_results.empty:
            for metric in ['mean_values', 'max_values', 'delta_values']:
                if metric in best_results.columns:
                    fig, _ = create_v_iters_plot(
                        v_arr=best_results[metric].iloc[0],
                        y_label=metric
                    )
                    save_plot(fig, fig_dir,
                              f'convergence_{method}_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}')
                    logger.info(f'Convergence plot for {metric} saved in {fig_dir}/convergence_{method}_{metric}_posi{position_bin}_vel{velocity_bin}_ang{angular_velocity_bin}')
                    plt.close(fig)

    logger.info('Finished')
    return result


# def main():
#     params = {
#         'gamma': [0.2, 0.4, 0.6, 0.8, 0.9, 0.99],
#         'init_alpha': np.round(np.linspace(0.2, 1.0, 5), 1),
#         'init_epsilon': np.round(np.linspace(0.2, 1.0, 5), 1),
#         'n_episodes': [100, 500, 1000, 2000],
#     }
#     state_params = {
#         'position_bins': [2, 5, 10, 20, 50]
#     }
#
#     env_name = 'cartpole_sarsa_200'
#
#     cartpole_exp = CartPoleExperiment(env_name=env_name,
#                                       result_dir='results',
#                                       fig_dir='figs',
#                                       random_seed=17)
#     results = cartpole_exp.run_sarsa_leanring_grid_search(
#         params=params,
#         state_params=state_params,
#         test_iters=200,
#         verbose=False,
#         log_name='sarsa_learning_grid_search',
#         n_processes=None  # Will use cpu_count() by default
#     )
#
#     # save
#     with open(f'results/{env_name}/sarsa_results.pkl', 'wb') as f:
#         pickle.dump(results, f)

def main():
    params = {
        'gamma': [0.2, 0.4, 0.6, 0.8, 0.9, 0.99],
        'init_alpha': np.round(np.linspace(0.2, 1.0, 5), 1),
        'init_epsilon': np.round(np.linspace(0.2, 1.0, 5), 1),
        'n_episodes': [10000],
    }
    state_params = {
        'position_bins': [2, 5, 10, 20, 50]
    }
    env_name = 'cartpole_sarsa_n10000'

    cartpole_exp = CartPoleExperiment(env_name=env_name,
                                      result_dir='results',
                                      fig_dir='figs',
                                      random_seed=17)
    results = cartpole_exp.run_sarsa_leanring_grid_search(
        params=params,
        state_params=state_params,
        test_iters=200,
        verbose=False,
        log_name='sarsa_learning_grid_search',
        n_processes=None  # Will use cpu_count() by default
    )

if __name__ == '__main__':
    main()










