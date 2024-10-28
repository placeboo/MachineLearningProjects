from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
from typing import Dict, Tuple

class EMCluster:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, n_cluster: int) -> np.ndarray:
        """Fit the KMeans model to the data."""
        self.model = GaussianMixture(n_components=n_cluster, random_state=self.random_state)
        return self.model.fit_predict(X)

    def get_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Get clustering metrics."""
        return {
            'bic': self.model.bic(X),
            'aic': self.model.aic(X),
            'silhouette_score': silhouette_score(X, labels) if len(set(labels)) > 1 else 0,
            'calinski_harabasz_score': calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else 0
        }

    def get_cluster_features(self, X: np.ndarray) -> np.ndarray:
        """
        generate features for neural network input"
        - distance to all centroids
        - cluster assignment
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        distances = self.model.predict_proba(X)
        labels = self.model.predict(X).reshape(-1, 1)
        return np.hstack([distances, labels])