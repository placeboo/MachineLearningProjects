import itertools

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict
import os
import pickle

from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.plots import Plots
from src.algorithms import vi_grid_search, pi_grid_search, q_learning_grid_search, sarsa_learning_grid_search
from src.utils.plotting import create_param_heatmap, save_plot, create_v_iters_plot, modified_plot_policy
from src.utils.logging import setup_logging

class BlackjackExperiment:
    def __init__(self,
                 env_name: str = 'blackjack',
                 result_dir: str = 'results',
                 fig_dir: str = 'figs',
                 random_seed: int = 42):
        self.env = gym.make('Blackjack-v1', render_mode=None)
        self.env = BlackjackWrapper(self.env)
        self.result_dir = os.path.join(result_dir, env_name)
        self.fig_dir = os.path.join(fig_dir, env_name)
        self.env_name = env_name
        self.random_seed = random_seed

        # create directories
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def run_vi_pi_grid_search(self,
                              params: Dict=None,
                              test_iters: int = 200,
                              verbose: bool = True,
                              log_name: str='log'):
        # set up logging
        logger = setup_logging(result_dir=self.result_dir,
                               name=log_name)
        if params is None:
            params = {}

        # log the input params
        logger.info(f"Running {self.env_name} Experiment")
        logger.info(f"Params: {params}")
        logger.info(f"Test Iters: {test_iters}")

        # run vi grid search
        logger.info("Running Value Iteration Grid Search")
        best_params, highest_avg_reward, iteration_results = \
            vi_grid_search(
                env=self.env,
                params=params,
                test_iters=test_iters,
                verbose=verbose,
                random_seed=self.random_seed
            )
        logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")
        # save results
        vi_results = {
            'best_params': best_params,
            'highest_avg_reward': highest_avg_reward,
            'iteration_results': iteration_results
        }
        with open(f'{self.result_dir}/vi_grid_search_results.pkl', 'wb') as f:
            pickle.dump(vi_results, f)
        logger.info(f'Value Iteration Grid Search results saved to {self.result_dir}/vi_grid_search_results.pkl')

        iteration_results_df = pd.DataFrame(iteration_results)
        # draw heatmap
        logger.info('#' * 50)
        logger.info("Creating Value Iteration Heatmap")
        for metric in ['converged_iter', 'mean_reward', 'runtime', 'std_reward']:
            fig, _ = create_param_heatmap(
                data = iteration_results_df,
                x_param = 'gamma',
                y_param = 'theta',
                metric = metric,
                cmap='Greens',
                multiplier=2,
                annot=False
            )
            save_plot(fig, self.fig_dir, f'heatmap_vi_{metric}')
            logger.info(f'Value Iteration Heatmap saved to {self.fig_dir}/heatmap_vi_{metric}.png')

        # draw convergence plot
        logger.info("Creating Value Iteration Convergence Plot")
        # find the resutls with best to draw the convergence plot
        best_results = iteration_results_df[
            (iteration_results_df['gamma'] == best_params['gamma']) &
            (iteration_results_df['theta'] == best_params['theta'])
        ]

        for metric in ['mean_values', 'max_values', 'delta_values']:
            fig, _ = create_v_iters_plot(
                v_arr = best_results[metric].values[0],
                y_label = metric
            )
            save_plot(fig, self.fig_dir, f'convergence_vi_{metric}')
            logger.info(f'Value Iteration Convergence Plot saved to {self.fig_dir}/convergence_vi_{metric}.png')

        # draw policy plot
        logger.info("Creating Value Iteration Policy Plot")
        pi = best_results['pi'].values[0]
        V = best_results['V'].values[0]
        blackjack_actions = {0: "S", 1: "H"}
        blackjack_map_size = (29, 10)
        # get formatted state values and policy map
        val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)
        fig, _ = modified_plot_policy(val_max, policy_map, None, multiplier=1.5)
        save_plot(fig, self.fig_dir, f'policy_vi')
        logger.info(f'Value Iteration Policy Plot saved to {self.fig_dir}/policy_vi.png')

        logger.info('#' * 50)

        # run pi grid search
        logger.info("Running Policy Iteration Grid Search")
        best_params, highest_avg_reward, iteration_results = \
            pi_grid_search(
                env=self.env,
                params=params,
                test_iters=test_iters,
                verbose=verbose,
                random_seed=self.random_seed
            )
        logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")
        # save results
        pi_results = {
            'best_params': best_params,
            'highest_avg_reward': highest_avg_reward,
            'iteration_results': iteration_results
        }
        with open(f'{self.result_dir}/pi_grid_search_results.pkl', 'wb') as f:
            pickle.dump(pi_results, f)
        logger.info(f'Policy Iteration Grid Search results saved to {self.result_dir}/pi_grid_search_results.pkl')

        iteration_results_df = pd.DataFrame(iteration_results)
        # draw heatmap
        logger.info('#' * 50)
        logger.info("Creating Policy Iteration Heatmap")
        for metric in ['converged_iter', 'mean_reward', 'runtime', 'std_reward']:
            fig, _ = create_param_heatmap(
                data = iteration_results_df,
                x_param = 'gamma',
                y_param = 'theta',
                metric = metric,
                cmap='Greens',
                multiplier=2,
                annot=False
            )
            save_plot(fig, self.fig_dir, f'heatmap_pi_{metric}')
            logger.info(f'Policy Iteration Heatmap saved to {self.fig_dir}/heatmap_pi_{metric}.png')

        # draw convergence plot
        logger.info("Creating Policy Iteration Convergence Plot")
        # find the resutls with best to draw the convergence plot
        best_results = iteration_results_df[
            (iteration_results_df['gamma'] == best_params['gamma']) &
            (iteration_results_df['theta'] == best_params['theta'])
        ]

        for metric in ['mean_values', 'max_values', 'delta_values']:
            fig, _ = create_v_iters_plot(
                v_arr = best_results[metric].values[0],
                y_label = metric
            )
            save_plot(fig, self.fig_dir, f'convergence_pi_{metric}')
            logger.info(f'Policy Iteration Convergence Plot saved to {self.fig_dir}/convergence_pi_{metric}.png')

        # draw policy plot
        logger.info("Creating Policy Iteration Policy Plot")
        pi = best_results['pi'].values[0]
        V = best_results['V'].values[0]
        blackjack_actions = {0: "S", 1: "H"}
        blackjack_map_size = (29, 10)
        # get formatted state values and policy map
        val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)
        fig, _ = modified_plot_policy(val_max, policy_map, None, multiplier=2)
        save_plot(fig, self.fig_dir, f'policy_pi')
        logger.info(f'Policy Iteration Policy Plot saved to {self.fig_dir}/policy_pi.png')

    def run_q_learning_grid_search(self,
                                   params: Dict = None,
                                   test_iters: int = 200,
                                   verbose: bool = True,
                                   log_name: str = 'log'):
        # set up logging
        logger = setup_logging(result_dir=self.result_dir,
                               name=log_name)

        if params is None:
            params = {}
            param_keys = None
        else:
            param_keys = list(params.keys())

        # log the input params
        logger.info(f"Running {self.env_name} Experiment")
        logger.info(f"Params: {params}")
        logger.info(f"Test Iters: {test_iters}")

        # run q-learning grid search
        logger.info("Running Q-Learning Grid Search")
        best_params, highest_avg_reward, iteration_results = \
            q_learning_grid_search(
                env=self.env,
                params=params,
                test_iters=test_iters,
                verbose=verbose,
                random_seed=self.random_seed
            )
        logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")
        # save results
        q_learning_results = {
            'best_params': best_params,
            'highest_avg_reward': highest_avg_reward,
            'iteration_results': iteration_results
        }
        with open(f'{self.result_dir}/q_learning_grid_search_results.pkl', 'wb') as f:
            pickle.dump(q_learning_results, f)
        logger.info(f'Q-Learning Grid Search results saved to {self.result_dir}/q_learning_grid_search_results.pkl')

        iteration_results_df = pd.DataFrame(iteration_results)

        if param_keys is not None and len(param_keys) >= 2:
            for pair in itertools.combinations(param_keys, 2):
                # copy the params_keys
                rest_params = param_keys.copy()
                rest_params.remove(pair[0])
                rest_params.remove(pair[1])
                tmp_result_df = iteration_results_df.copy()
                for param in rest_params:
                    tmp_result_df = tmp_result_df[tmp_result_df[param] == best_params[param]]

                # draw heatmap
                logger.info('#' * 50)
                logger.info(f"Creating Q-Learning Heatmap for {pair[0]} and {pair[1]}")
                for metric in ['mean_reward', 'runtime', 'std_reward']:
                    fig, _ = create_param_heatmap(
                        data = tmp_result_df,
                        x_param = pair[0],
                        y_param = pair[1],
                        metric = metric,
                        cmap='Greens',
                        multiplier=2,
                        annot=False
                    )
                    save_plot(fig, self.fig_dir, f'heatmap_q_learning_{pair[0]}_{pair[1]}_{metric}')
                    logger.info(f'Q-Learning Heatmap saved to {self.fig_dir}/heatmap_q_learning_{pair[0]}_{pair[1]}_{metric}.png')

        # draw convergence plot
        logger.info("Creating Q-Learning Convergence Plot")
        # find the resutls with best to draw the convergence plot
        best_results = iteration_results_df[
            (iteration_results_df['gamma'] == best_params['gamma']) &
            (iteration_results_df['init_alpha'] == best_params['init_alpha']) &
            (iteration_results_df['min_alpha'] == best_params['min_alpha']) &
            (iteration_results_df['alpha_decay_ratio'] == best_params['alpha_decay_ratio']) &
            (iteration_results_df['init_epsilon'] == best_params['init_epsilon']) &
            (iteration_results_df['min_epsilon'] == best_params['min_epsilon']) &
            (iteration_results_df['epsilon_decay_ratio'] == best_params['epsilon_decay_ratio']) &
            (iteration_results_df['n_episodes'] == best_params['n_episodes'])
            ]

        for metric in ['mean_values', 'max_values', 'delta_values']:
            fig, _ = create_v_iters_plot(
                v_arr = best_results[metric].values[0],
                y_label = metric
            )
            save_plot(fig, self.fig_dir, f'convergence_q_learning_{metric}')
            logger.info(f'Q-Learning Convergence Plot saved to {self.fig_dir}/convergence_q_learning_{metric}.png')

        # draw policy plot
        logger.info("Creating Q-Learning Policy Plot")
        pi = best_results['pi'].values[0]
        V = best_results['V'].values[0]
        blackjack_actions = {0: "S", 1: "H"}
        blackjack_map_size = (29, 10)
        # get formatted state values and policy map
        val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)
        fig, _ = modified_plot_policy(val_max, policy_map, None, multiplier=2)
        save_plot(fig, self.fig_dir, f'policy_q_learning')
        logger.info(f'Q-Learning Policy Plot saved to {self.fig_dir}/policy_q_learning.png')
        logger.info("Experiment Completed")

    def run_sarsa_learning_grid_search(self,
                                   params: Dict = None,
                                   test_iters: int = 200,
                                   verbose: bool = True,
                                   log_name: str = 'log'):
        # set up logging
        logger = setup_logging(result_dir=self.result_dir,
                               name=log_name)

        if params is None:
            params = {}
            param_keys = None
        else:
            param_keys = list(params.keys())

        # log the input params
        logger.info(f"Running {self.env_name} Experiment")
        logger.info(f"Params: {params}")
        logger.info(f"Test Iters: {test_iters}")

        # run q-learning grid search
        logger.info("Running Sarsa-Learning Grid Search")
        best_params, highest_avg_reward, iteration_results = \
            sarsa_learning_grid_search(
                env=self.env,
                params=params,
                test_iters=test_iters,
                verbose=verbose,
                random_seed=self.random_seed
            )
        logger.info(f"Best params: {best_params}, Highest Avg. Reward: {highest_avg_reward}")
        # save results
        sarsa_learning_results = {
            'best_params': best_params,
            'highest_avg_reward': highest_avg_reward,
            'iteration_results': iteration_results
        }
        with open(f'{self.result_dir}/sarsa_learning_grid_search_results.pkl', 'wb') as f:
            pickle.dump(sarsa_learning_results, f)
        logger.info(f'Sarsa-Learning Grid Search results saved to {self.result_dir}/Sarsa_learning_grid_search_results.pkl')

        iteration_results_df = pd.DataFrame(iteration_results)

        if param_keys is not None and len(param_keys) >= 2:
            for pair in itertools.combinations(param_keys, 2):
                # copy the params_keys
                rest_params = param_keys.copy()
                rest_params.remove(pair[0])
                rest_params.remove(pair[1])
                tmp_result_df = iteration_results_df.copy()
                for param in rest_params:
                    tmp_result_df = tmp_result_df[tmp_result_df[param] == best_params[param]]

                # draw heatmap
                logger.info('#' * 50)
                logger.info(f"Creating Sarsa-Learning Heatmap for {pair[0]} and {pair[1]}")
                for metric in ['mean_reward', 'runtime', 'std_reward']:
                    fig, _ = create_param_heatmap(
                        data = tmp_result_df,
                        x_param = pair[0],
                        y_param = pair[1],
                        metric = metric,
                        cmap='Greens',
                        multiplier=2,
                        annot=False
                    )
                    save_plot(fig, self.fig_dir, f'heatmap_sarsa_learning_{pair[0]}_{pair[1]}_{metric}')
                    logger.info(f'Sarsa-Learning Heatmap saved to {self.fig_dir}/heatmap_sarsa_learning_{pair[0]}_{pair[1]}_{metric}.png')

        # draw convergence plot
        logger.info("Creating Sarsa-Learning Convergence Plot")
        # find the resutls with best to draw the convergence plot
        best_results = iteration_results_df[
            (iteration_results_df['gamma'] == best_params['gamma']) &
            (iteration_results_df['init_alpha'] == best_params['init_alpha']) &
            (iteration_results_df['min_alpha'] == best_params['min_alpha']) &
            (iteration_results_df['alpha_decay_ratio'] == best_params['alpha_decay_ratio']) &
            (iteration_results_df['init_epsilon'] == best_params['init_epsilon']) &
            (iteration_results_df['min_epsilon'] == best_params['min_epsilon']) &
            (iteration_results_df['epsilon_decay_ratio'] == best_params['epsilon_decay_ratio']) &
            (iteration_results_df['n_episodes'] == best_params['n_episodes'])
            ]

        for metric in ['mean_values', 'max_values', 'delta_values']:
            fig, _ = create_v_iters_plot(
                v_arr = best_results[metric].values[0],
                y_label = metric
            )
            save_plot(fig, self.fig_dir, f'convergence_sarsa_learning_{metric}')
            logger.info(f'Q-Learning Convergence Plot saved to {self.fig_dir}/convergence_sarsa_learning_{metric}.png')
        
        # draw policy plot
        logger.info("Creating Sarsa-Learning Policy Plot")
        pi = best_results['pi'].values[0]
        V = best_results['V'].values[0]
        blackjack_actions = {0: "S", 1: "H"}
        blackjack_map_size = (29, 10)
        # get formatted state values and policy map
        val_max, policy_map = Plots.get_policy_map(pi, V, blackjack_actions, blackjack_map_size)
        fig, _ = modified_plot_policy(val_max, policy_map, None, multiplier=2)
        save_plot(fig, self.fig_dir, f'policy_sarsa_learning')
        logger.info(f'Sarsa-Learning Policy Plot saved to {self.fig_dir}/policy_sarsa_learning.png')
        logger.info("Experiment Completed")








