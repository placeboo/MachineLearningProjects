import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from typing import Dict, Tuple, Any
import time

class DimensionalityReductionNN:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.nn = MLPClassifier(random_state=self.random_state, max_iter=1000, learning_rate='adaptive', early_stopping=True)
        self.best_nn = None
        self.best_params = None
        self.cv_results = None

    def train_tuning(self, X_train: np.ndarray,
                     y_train: np.ndarray,
                     param_grid: Dict,
                     scoring: str='f1'):
        """Train the neural network.

        Parameters:
        -----------
        X_train : np.ndarray
            Training data after dimensionality reduction
        y_train : np.ndarray
            Training labels
        param_grid : Dict
        """
        grid_search = GridSearchCV(self.nn, param_grid, cv=5, scoring=scoring,n_jobs=-1, return_train_score=True, verbose=3)
        grid_search.fit(X_train, y_train)
        self.best_nn = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_

        return self.best_nn, self.best_params, self.cv_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels."""
        return self.best_nn.predict(X)

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

    def nn_learning_curve(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          cv: int=5,
                          train_sizes: np.ndarray=np.linspace(0.1, 1.0, 5)) -> Dict:
        # split data for learning curve
        train_sizes, train_scores, val_scores, fit_times, _ \
        = learning_curve(self.best_nn,
                            X, y,
                            train_sizes=train_sizes,
                            cv=cv,
                            n_jobs=-1,
                            random_state=self.random_state,
                            return_times=True,
                            scoring='accuracy')
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'fit_times': fit_times
        }


class NNModel:
    def __init__(self,
                 param: Dict,
                 random_state: int=17):

        self.random_state = random_state
        self.nn = MLPClassifier(random_state=self.random_state, **param)

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray) -> Any:
        """Train the neural network."""
        self.nn.fit(X_train, y_train)
        return self.nn

    def evaluate(self,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> Dict:
        """Evaluate the neural network."""
        y_pred = self.nn.predict(X_test)
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

    def nn_learning_curve(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          cv: int=5,
                          train_sizes: np.ndarray=np.linspace(0.1, 1.0, 5)) -> Dict:
        # split data for learning curve
        train_sizes, train_scores, val_scores, fit_times, _ \
        = learning_curve(self.nn,
                         X, y,
                         train_sizes=train_sizes,
                         cv=cv,
                         n_jobs=-1,
                         random_state=self.random_state,
                         return_times=True,
                         scoring='accuracy')

        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'fit_times': fit_times
        }


