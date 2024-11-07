from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
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

        labels = self.model.predict(X).reshape(-1, 1)
        return labels

    def evaluate_with_ground_truth(self, y_true: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate clustering results using ground truth labels
        Only call this after choosing optimal k using unsupervised metrics

        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth labels
        labels : np.ndarray
            Cluster assignments

        Returns:
        --------
        Dict : Dictionary of evaluation metrics
        """
        return {
            'adjusted_rand': adjusted_rand_score(y_true, labels),
            'normalized_mutual_info': normalized_mutual_info_score(y_true, labels),
            'adjusted_mutual_info': adjusted_mutual_info_score(y_true, labels),
            'homogeneity': homogeneity_score(y_true, labels),
            'completeness': completeness_score(y_true, labels)
        }