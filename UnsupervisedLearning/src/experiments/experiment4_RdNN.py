import numpy as np
from src.dimensionality_reduction.pca import PCAReducer
from src.dimensionality_reduction.random_projection import RandomProjectionReducer
from src.dimensionality_reduction.ica import ICAReducer
from src.neural_network.nn_model import DimensionalityReductionNN
from typing import Tuple, List, Dict
import os
from src.utils.data_loader import save_pickle

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
                'metrics': metrics
            }
        return results

    def save_results(self, results: Dict):
        save_pickle(results, self.save_dir, f"results_{self.rd}")