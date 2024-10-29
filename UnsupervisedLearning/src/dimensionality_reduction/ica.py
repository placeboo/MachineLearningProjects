from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import kurtosis
from typing import Dict, Tuple

class ICAReducer:
    def __init__(self, random_state: int=17, max_iter: int=1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None

    def fit(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Fit the ICA model to the data."""
        self.model = FastICA(n_components=n_components, random_state=self.random_state, max_iter=self.max_iter)
        return self.model.fit_transform(X)

    def get_metrics(self, X_transformed: np.ndarray) -> Dict:
        """Get ICA metrics."""
        if self.model is None:
            raise ValueError("Model not fitted")
        # Calculate kurtosis for each component
        kurt_values = [kurtosis(comp) for comp in X_transformed.T]
        explained_variance = np.var(X_transformed, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        return {
            'kurtosis_values': kurt_values,
            'mean_kurtosis': np.mean(kurt_values),
            'abs_mean_kurtosis': np.mean(np.abs(kurt_values)),
            'explained_variance': explained_variance.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_explained_variance': cumulative_explained_variance.tolist(),
            'n_iter': self.model.n_iter_
        }

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform the transformed data."""
        if self.model is None:
            raise ValueError("Model not fitted")

        return self.model.inverse_transform(X_transformed)
