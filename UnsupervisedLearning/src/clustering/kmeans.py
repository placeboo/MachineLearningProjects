from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
from typing import Dict, Tuple

class KMeansCluster:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, n_cluster: int) -> np.ndarray:
        """Fit the KMeans model to the data."""
        self.model = KMeans(n_clusters=n_cluster, random_state=self.random_state)
        return self.model.fit_predict(X)

    def get_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Get clustering metrics."""
        return {
            'inertia': self.model.inertia_,
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

        distances = self.model.transform(X) # distance to all centroids
        labels = self.model.predict(X).reshape(-1, 1) # cluster assignment
        return np.hstack([distances, labels])