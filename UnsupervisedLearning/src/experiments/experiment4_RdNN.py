import numpy as np
import pandas as pd

from src.dimensionality_reduction.pca import PCAReducer
from src.dimensionality_reduction.random_projection import RandomProjectionReducer
from src.dimensionality_reduction.ica import ICAReducer
from src.neural_network.nn_model import DimensionalityReductionNN, NNModel
from typing import Tuple, List, Dict
import time
import os
from src.utils.data_loader import save_pickle
from src.utils.plotting import set_plot_style

class ExperimentRdNN:
    def __init__(self,
                 rd: str,
                 n_components_range: List[int],
                 nn_params_grid: Dict,
                 save_dir: str,
                 random_state: int=17):
        self.random_state = random_state
        self.nn_params_grid = nn_params_grid
        self.n_components_range = n_components_range
        self.rd = rd
        if rd == 'pca':
            self.rd_method = PCAReducer(random_state=self.random_state)
        elif rd == 'rp':
            self.rd_method = RandomProjectionReducer(random_state=self.random_state)
        elif rd == 'ica':
            self.rd_method = ICAReducer(random_state=self.random_state)
        else:
            raise ValueError("Invalid dimensionality reduction method")
        self.save_dir = save_dir

    def run_experiment(self, X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       y_test: np.ndarray) -> Dict:

        results = {}

        for n in self.n_components_range:
            nn_model = DimensionalityReductionNN(random_state=self.random_state)
            print(f"Running RD with n_components={n}")
            X_train_transformed = self.rd_method.fit(X_train, n)
            X_test_transformed = self.rd_method.transform(X_test)
            print("Running NN")
            best_nn, best_params, cv_results = nn_model.train_tuning(X_train_transformed, y_train, self.nn_params_grid)
            print("Evaluating NN")
            metrics = nn_model.evaluate(X_test_transformed, y_test)
            results[n] = {
                'best_nn': best_nn,
                'best_params': best_params,
                'cv_results': cv_results,
                'metrics': metrics,
                'X_train_transformed': X_train_transformed,
                'X_test_transformed': X_test_transformed
            }
        return results

    def save_results(self, results: Dict):
        save_pickle(results, self.save_dir, f"results_{self.rd}")

class ExperimentNN:
    def __init__(self,
                 rd_config: List[Dict],
                 k_cluster: int,
                 nn_config: Dict,
                 X_train: np.ndarray,
                 X_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 save_dir: str,
                 random_state: int=17):
        """
        Args:
            rd_config (List[Dict]): List of dictionaries with keys 'rd' and 'n_component'
            k_cluster (int): Number of clusters
            nn_config (Dict): Dictionary with keys params for neural network
            X_train (np.ndarray): Training data
            X_test (np.ndarray): Test data
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            save_dir (str): Directory to save results
            random_state (int, optional): Random state. Defaults
        """

        self.random_state = random_state
        self.rd_config = rd_config
        self.k_cluster = k_cluster
        self.nn_config = nn_config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.save_dir = save_dir

    def run_experiment(self):

        results = {
            'base': {},
        }
        for key in self.rd_config:
            results[key['rd']] = {}

        # run base data without dimensionality reduction
        base_nn = NNModel(random_state=self.random_state,
                          param=self.nn_config)
        print("Running base NN")
        base_nn.train(self.X_train, self.y_train)
        results['base']['metrics'] = base_nn.evaluate(self.X_test, self.y_test)

        print('Run learning curve for base NN')
        base_nn_learning_curve = base_nn.nn_learning_curve(self.X_train, self.y_train)
        results['base']['learning_curve'] = base_nn_learning_curve


        for rd in self.rd_config:
            if rd['rd'] == 'pca':
                rd_method = PCAReducer(random_state=self.random_state)
            elif rd['rd'] == 'rp':
                rd_method = RandomProjectionReducer(random_state=self.random_state)
            elif rd['rd'] == 'ica':
                rd_method = ICAReducer(random_state=self.random_state)
            else:
                raise ValueError("Invalid dimensionality reduction method")

            X_train_transformed = rd_method.fit(self.X_train, rd['n_component'])
            X_test_transformed = rd_method.transform(self.X_test)
            # store transformed data

            # run nn model
            print(f"Running NN with {rd['rd']} and n_components={rd['n_component']}")
            nn = NNModel(random_state=self.random_state,
                         param=self.nn_config)
            nn.train(X_train_transformed, self.y_train)
            metrics = nn.evaluate(X_test_transformed, self.y_test)
            results[rd['rd']] = {
                'metrics': metrics
            }
            # run learning curve
            print(f"Run learning curve for {rd['rd']} NN")
            learning_curve_results = nn.nn_learning_curve(X_train_transformed, self.y_train)
            results[rd['rd']]['learning_curve'] = learning_curve_results

        return results

    def save_results(self, results: Dict):
        save_pickle(results, self.save_dir, "a1_setting_results")








