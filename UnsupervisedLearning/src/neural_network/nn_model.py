import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, Any
import time

class DimensionalityReductionNN:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.nn = MLPClassifier(random_state=self.random_state, max_iter=1000)
        self.best_nn = None
        self.best_params = None
        self.training_time = None

    @staticmethod
    def fit_transform_dr(dr_model: Any, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform the data using a dimensionality reduction model."""
        X_train_transformed = dr_model.fit_transform(X_train)
        X_test_transformed = dr_model.transform(X_test)
        return X_train_transformed, X_test_transformed

    def train(self, X_train: np.ndarray, y_train: np.ndarray, param_grid: Dict):
        """Train the neural network.

        Parameters:
        -----------
        X_train : np.ndarray
            Training data after dimensionality reduction
        y_train : np.ndarray
            Training labels
        param_grid : Dict
        """
        grid_search = GridSearchCV(self.nn, param_grid, cv=5, n_jobs=-3, return_train_score=True, verbose=3)

        nn = MLPClassifier(random_state=self.random_state, max_iter=1000)
        start = time.time()
        grid_search.fit(X_train, y_train)
        self.training_time = time.time() - start

        self.best_nn = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        return self.best_nn, self.best_params

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the neural network.

        Parameters:
        -----------
        X_test : np.ndarray
            Test data after dimensionality reduction
        y_test : np.ndarray
            Test labels
        """
        y_pred = self.best_nn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

