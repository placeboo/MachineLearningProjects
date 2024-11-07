import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from src.dimensionality_reduction.pca import PCAReducer
from src.dimensionality_reduction.ica import ICAReducer
from src.dimensionality_reduction.random_projection import RandomProjectionReducer
from src.dimensionality_reduction.umap import UMAPReducer


class DimensionalityReductionExperiment:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.pca = PCAReducer(random_state=self.random_state)
        self.ica = ICAReducer(random_state=self.random_state)
        self.rp = RandomProjectionReducer(random_state=self.random_state)
        self.umap = UMAPReducer(random_state=self.random_state)

    def run_pca_analysis(self, X: np.ndarray, n_components_range: List[int]) -> Tuple[Dict, Dict]:
        """Run PCA analysis for different numbers of components."""
        metrics = {
            'n_components': n_components_range,
            'explained_variance_ratio': [],
            'cumulative_explained_variance': [],
            'singular_values': []
        }

        transformed_data = {}
        for n in n_components_range:
            X_transformed = self.pca.fit(X, n)
            pca_metrics = self.pca.get_metrics()

            metrics['explained_variance_ratio'].append(pca_metrics['explained_variance_ratio'])
            metrics['cumulative_explained_variance'].append(pca_metrics['cumulative_explained_variance'])
            metrics['singular_values'].append(pca_metrics['singular_values'])

            transformed_data[n] = X_transformed

        return pd.DataFrame(metrics), transformed_data

    def run_ica_analysis(self, X: np.ndarray, n_components_range: List[int]) -> Tuple[pd.DataFrame, Dict]:
        """Run ICA analysis for different numbers of components."""
        metrics = {
            'n_components': n_components_range,
            'kurtosis_values': [],
            'mean_kurtosis': [],
            'abs_mean_kurtosis': [],
            'explained_variance_ratio': [],
            'cumulative_explained_variance': [],
            'n_iter': []
        }

        transformed_data = {}
        for n in n_components_range:
            X_transformed = self.ica.fit(X, n)
            ica_metrics = self.ica.get_metrics(X_transformed)

            metrics['kurtosis_values'].append(ica_metrics['kurtosis_values'])
            metrics['mean_kurtosis'].append(ica_metrics['mean_kurtosis'])
            metrics['abs_mean_kurtosis'].append(ica_metrics['abs_mean_kurtosis'])
            metrics['explained_variance_ratio'].append(ica_metrics['explained_variance_ratio'])
            metrics['cumulative_explained_variance'].append(ica_metrics['cumulative_explained_variance'])
            metrics['n_iter'].append(ica_metrics['n_iter'])

            transformed_data[n] = X_transformed

        return pd.DataFrame(metrics), transformed_data

    def run_rp_analysis(self, X: np.ndarray, n_components_range: List[int], n_trials: int = 5) -> Tuple[
        pd.DataFrame, Dict]:
        """Run Random Projection analysis for different numbers of components."""
        metrics = {
            'n_components': n_components_range,
            'reconstruction_error_mean': [],
            'reconstruction_error_std': [],
            'components_rank': []
        }

        transformed_data = {}
        for n in n_components_range:
            reconstruction_errors = []
            ranks = []

            # Multiple trials to account for randomness
            for trial in range(n_trials):
                X_transformed = self.rp.fit(X, n)
                rp_metrics = self.rp.get_metrics(X, X_transformed)

                reconstruction_errors.append(rp_metrics['reconstruction_error'])
                ranks.append(rp_metrics['components_rank'])

            metrics['reconstruction_error_mean'].append(np.mean(reconstruction_errors))
            metrics['reconstruction_error_std'].append(np.std(reconstruction_errors))
            metrics['components_rank'].append(np.mean(ranks))

            # Store last trial's transformed data
            transformed_data[n] = X_transformed

        return pd.DataFrame(metrics), transformed_data

    def run_umap_analysis(self, X: np.ndarray, n_components_range: List[int]) -> Tuple[pd.DataFrame, Dict]:
        """
        Run UMAP analysis for different numbers of components.

        Parameters:
        -----------
        X : np.ndarray
            Input data matrix
        n_components_range : List[int]
            List of number of components to try

        Returns:
        --------
        Tuple[pd.DataFrame, Dict]
            DataFrame with metrics and dictionary with transformed data
        """
        metrics = {
            'n_components': [],
            'trustworthiness': [],
            'nn_preservation': [],
            'distance_correlation': []
        }

        transformed_data = {}

        for n in n_components_range:
            # Fit and transform data
            X_transformed = self.umap.fit(X, n)
            umap_metrics = self.umap.get_metrics(X, X_transformed)

            # Store metrics
            metrics['n_components'].append(n)
            metrics['trustworthiness'].append(umap_metrics['trustworthiness'])
            metrics['nn_preservation'].append(umap_metrics['nn_preservation'])
            metrics['distance_correlation'].append(umap_metrics['distance_correlation'])

            # Store transformed data
            transformed_data[n] = X_transformed

        return pd.DataFrame(metrics), transformed_data
