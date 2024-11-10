import numpy as np
from typing import Tuple, List, Dict
import pandas as pd

from src.clustering.kmeans import KMeansCluster
from src.clustering.em import EMCluster
from src.neural_network.nn_model import DimensionalityReductionNN

from src.utils.data_loader import save_pickle

class ClusterNN:
    def __init__(self,
                 cluster_method: str,
                 n_cluster: int,
                 nn_params_grid: Dict,
                 random_state: int=17,
                 save_dir: str='results/'):
        self.cluster_method = cluster_method
        if cluster_method == 'kmeans':
            self.cluster = KMeansCluster(random_state=random_state)
        elif cluster_method == 'em':
            self.cluster = EMCluster(random_state=random_state)
        else:
            raise ValueError("Invalid clustering method")

        self.n_cluster = n_cluster
        self.random_state = random_state
        self.nn_params_grid = nn_params_grid
        self.save_dir = save_dir
    def run_experiment(self, X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       y_test: np.ndarray) -> Dict:

        # run clustering
        print(f"Running clustering with n_clusters={self.n_cluster}")
        _ = self.cluster.fit(X_train, self.n_cluster)
        x_train_new_features = self.cluster.get_cluster_features(X_train)
        x_test_new_features = self.cluster.get_cluster_features(X_test)

        new_x_train = np.hstack([X_train, x_train_new_features])
        new_x_test = np.hstack([X_test, x_test_new_features])

        # run neural network
        nn_model = DimensionalityReductionNN(random_state=self.random_state)
        print("Running NN")
        best_nn, best_params, cv_results = nn_model.train_tuning(new_x_train, y_train, self.nn_params_grid)
        print("Evaluating NN")
        metrics = nn_model.evaluate(new_x_test, y_test)
        results = {
            'best_nn': best_nn,
            'best_params': best_params,
            'cv_results': cv_results,
            'metrics': metrics,
            'X_train_transformed': new_x_train,
            'X_test_transformed': new_x_test
        }
        return results

class ExperimentClusterNN:
    def __init__(self,
                 cluster_config: List[Dict],
                 nn_params_grid: Dict,
                 X_train: np.ndarray,
                 X_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray,
                 random_state: int=17):
        self.random_state = random_state
        self.cluster_config = cluster_config
        self.nn_params_grid = nn_params_grid
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def run_experiment(self) -> Dict:
        """
        pick the best by grid search.
        use the best model to do learning curve
        use the best model to do prediction and evaluation
        """
        results = {
            'base': {},
        }

        for key in self.cluster_config:
            results[key['cluster']] = {}

        # ran base model without dimensionality reduction
        base_nn = DimensionalityReductionNN(random_state=self.random_state)
        print("Running base NN grid search")
        best_nn, best_params, cv_results = base_nn.train_tuning(self.X_train, self.y_train, self.nn_params_grid)
        print("Evaluating base NN")
        metrics = base_nn.evaluate(self.X_test, self.y_test)
        y_pred = base_nn.predict(self.X_test)
        results['base']['best_params'] = best_params
        results['base']['metrics'] = metrics
        results['base']['best_nn'] = best_nn
        results['base']['y_pred'] = y_pred

        print("Running base NN learning curve")
        lr_data = base_nn.nn_learning_curve(self.X_train, self.y_train)
        results['base']['learning_curve'] = lr_data

        for cluster in self.cluster_config:
            if cluster['cluster']=='kmeans':
                cluster_model = KMeansCluster(random_state=self.random_state)
            elif cluster['cluster']=='em':
                cluster_model = EMCluster(random_state=self.random_state)
            else:
                raise ValueError("Invalid clustering method")

            print(f"Running {cluster['cluster']} with n_clusters={cluster['k']}")
            _ = cluster_model.fit(self.X_train, cluster['k'])
            x_train_new_features = cluster_model.get_cluster_features(self.X_train)
            x_test_new_features = cluster_model.get_cluster_features(self.X_test)

            new_x_train = np.hstack([self.X_train, x_train_new_features])
            new_x_test = np.hstack([self.X_test, x_test_new_features])

            # run grid search
            nn_model = DimensionalityReductionNN(random_state=self.random_state)
            print("Running NN grid search")
            best_nn, best_params, cv_results = nn_model.train_tuning(new_x_train, self.y_train, self.nn_params_grid)
            print("Evaluating NN")
            metrics = nn_model.evaluate(new_x_test, self.y_test)
            y_pred = nn_model.predict(new_x_test)
            results[cluster['cluster']]['best_params'] = best_params
            results[cluster['cluster']]['metrics'] = metrics
            results[cluster['cluster']]['best_nn'] = best_nn
            results[cluster['cluster']]['y_pred'] = y_pred

            print(f"Running {cluster['cluster']} learning curve")
            lr_data = nn_model.nn_learning_curve(new_x_train, self.y_train)
            results[cluster['cluster']]['learning_curve'] = lr_data

        return results





