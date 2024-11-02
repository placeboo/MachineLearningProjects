from sklearn.decomposition import PCA
import numpy as np
from typing import Dict, Tuple

class PCAReducer:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Fit the PCA model to the data."""
        self.model = PCA(n_components=n_components, random_state=self.random_state)
        return self.model.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.transform(X)

    def get_metrics(self) -> Dict:
        """Get PCA metrics."""
        if self.model is None:
            raise ValueError("Model not fitted")

        return {
            'explained_variance_ratio': self.model.explained_variance_ratio_.tolist(),
            'explained_variance': self.model.explained_variance_.tolist(),
            'singular_values': self.model.singular_values_.tolist(),
            'cumulative_explained_variance': np.cumsum(self.model.explained_variance_ratio_.tolist())
        }

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform the transformed data."""
        if self.model is None:
            raise ValueError("Model not fitted")

        return self.model.inverse_transform(X_transformed)

