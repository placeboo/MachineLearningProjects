import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict
import os
import pickle

from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from src.algorithms import vi_grid_search, pi_grid_search, q_learning_grid_search
from src.utils.plotting import create_param_heatmap, save_plot, create_v_iters_plot
from src.utils.logging import setup_logging

class BlackjackExperiment:
    def __init__(self,
                 env_name: str = 'blackjack',
                 result_dir: str = 'results',
                 fig_dir: str = 'figs'):
        self.env = gym.make('Blackjack-v1', render_mode=None)
        self.env = BlackjackWrapper(self.env)
        self.result_dir = os.path.join(result_dir, env_name)
        self.fig_dir = os.path.join(fig_dir, env_name)
        self.env_name = env_name

        # create directories
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def run_vi_pi_grid_search(self,
                              params: Dict=None,
                              test_iters: int = 200,
                              verbose: bool = True):
        # set up logging
        logger = setup_logging(result_dir=self.result_dir)
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
                verbose=verbose
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
                multiplier=1.5,
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

        logger.info('#' * 50)

        # run pi grid search
        logger.info("Running Policy Iteration Grid Search")
        best_params, highest_avg_reward, iteration_results = \
            pi_grid_search(
                env=self.env,
                params=params,
                test_iters=test_iters,
                verbose=verbose
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
                multiplier=1.5,
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











