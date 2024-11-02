from sklearn.random_projection import GaussianRandomProjection
import numpy as np
from typing import Dict, Tuple

class RandomProjectionReducer:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Fit the Random Projection model to the data."""
        self.model = GaussianRandomProjection(n_components=n_components, random_state=self.random_state)
        return self.model.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.transform(X)

    def get_metrics(self, X: np.ndarray, X_transformed: np.ndarray) -> Dict:
        """Get Random Projection metrics."""
        if self.model is None:
            raise ValueError("Model not fitted")

        # Calculate reconstruction error using pseudo-inverse
        components = self.model.components_
        pseudo_inv = np.linalg.pinv(components)
        X_reconstructed = np.dot(X_transformed, components)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        return {
            'reconstruction_error': reconstruction_error,
            'components_condition': np.linalg.cond(components),
            'components_rank': np.linalg.matrix_rank(components)
        }