"""
Run dimensionality reduction on the data set and do clustering analysis on the reduced data.
"""
import numpy as np
from typing import Tuple, List, Dict
import pandas as pd

from src.dimensionality_reduction.pca import PCAReducer
from src.dimensionality_reduction.ica import ICAReducer
from src.dimensionality_reduction.random_projection import RandomProjectionReducer
from src.experiments.experiment1_clustering import ClusteringExperiment


from src.clustering.kmeans import KMeansCluster
from src.clustering.em import EMCluster

class CombinedExperiment:
    def __init__(self, random_state: int=17):
        self.random_state = random_state

        self.pca = PCAReducer(random_state=self.random_state)
        self.ica = ICAReducer(random_state=self.random_state)
        self.rp = RandomProjectionReducer(random_state=self.random_state)

        self.clustering = ClusteringExperiment(random_state=self.random_state)

    def run_combined_analyis(self,
                             X: np.ndarray,
                             dr_components: List[int],
                             k_range: List[int]
        ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run combined analysis of dimensionality reduction followed by clustering.

        Parameters:
        -----------
        X : np.ndarray
            Input data
        dr_components : List[int]
            List of numbers of components to try for dimensionality reduction
        k_range : List[int]
            List of numbers of clusters to try

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames of clustering metrics for KMeans and EM with different dimensionality reduction methods and components
        """
        # initialize result dictionaries
        kmeans_metrics_df = pd.DataFrame()
        em_metrics_df = pd.DataFrame()
        transformed_data = {
            'pca': {},
            'ica': {},
            'rp': {}
        }

        # for each dim reduction method
        for dr_method in ['pca', 'ica', 'rp']:
            # get appropriate transfomer
            transformer = getattr(self, dr_method)

            # for each number of components
            for n_component in dr_components:
                # transform the data
                X_transformed = transformer.fit(X, n_component)
                transformed_data[dr_method][n_component] = X_transformed

                # run clustering analysis
                kmeans_metrics, em_metrics = self.clustering.run_clustering_analysis(X_transformed, k_range)

                kmeans_metrics['method'] = dr_method
                kmeans_metrics['n_components'] = n_component
                kmeans_metrics_df = pd.concat([kmeans_metrics_df, kmeans_metrics], ignore_index=True)
                em_metrics['method'] = dr_method
                em_metrics['n_components'] = n_component
                em_metrics_df = pd.concat([em_metrics_df, em_metrics], ignore_index=True)

        return kmeans_metrics_df, em_metrics_df, transformed_data


def find_optimal_combinations(kmeans_metrics_df: pd.DataFrame, em_metrics_df: pd.DataFrame) -> Dict:
    """
    find the optimal combination of dimensionality reduction method and number of components for each clustering algorithm

    Parameters:
    -----------
    kmeans_metrics_df : pd.DataFrame
        DataFrame of KMeans clustering metrics
    em_metrics_df : pd.DataFrame
        DataFrame of EM clustering metrics

    Returns:
    --------
    Dict
        Dictionary of optimal combinations for KMeans and EM
    """
    optimal_configs = {}

    kmeans_grouped = kmeans_metrics_df.groupby(['method', 'n_components'])
    for (method, n_comp), group in kmeans_grouped:
        # find the k with highest silhouette score for this configure
        best_k_idx = group['silhouette_score'].idxmax()
        best_k = group.loc[best_k_idx, 'k']
        best_score = group.loc[best_k_idx, 'silhouette_score']

        key = f'{method}_kmeans'
        if key not in optimal_configs or optimal_configs[key]['score'] < best_score:
            optimal_configs[key] = {
                'dr_method': method,
                'n_components': n_comp,
                'k': best_k,
                'score': best_score
            }

    em_grouped = em_metrics_df.groupby(['method', 'n_components'])
    for (method, n_comp), group in em_grouped:
        # find the k with highest silhouette score for this configure
        best_k_idx = group['silhouette_score'].idxmax()
        best_k = group.loc[best_k_idx, 'k']
        best_score = group.loc[best_k_idx, 'silhouette_score']

        key = f'{method}_em'
        if key not in optimal_configs or optimal_configs[key]['score'] < best_score:
            optimal_configs[key] = {
                'dr_method': method,
                'n_components': n_comp,
                'k': best_k,
                'score': best_score
            }

    return optimal_configs


def evaluate_clustering(transformed_data: Dict,
                        y: np.ndarray,
                        optimal_configs: Dict
                        ) -> Dict:
    """
    Evaluate clustering results using ground truth labels

    Parameters:
    -----------
    transformed_data : Dict
        Dictionary of transformed data with structure {method: {n_components: X_transformed}}
    y : np.ndarray
        Ground truth labels
    optimal_configs : Dict
        Dictionary of optimal configurations for KMeans and EM with structure {
            'pca_kmeans': {'dr_method': str, 'n_components': int, 'k': int, 'score': float},
            'ica_em': {'dr_method': str, 'n_components': int, 'k': int, 'score': float},
            ...
        }

    Returns:
    --------
    Dict
        Dictionary of evaluation metrics for KMeans and EM with structure
        {
            'pca_kmeans': {'metrics': Dict, 'composition': pd.DataFrame},
            'ica_em': {'metrics': Dict, 'composition': pd.DataFrame},
    """
    results = {}

    for key, config in optimal_configs.items():
        method, algo = key.split('_')
        X_transformed = transformed_data[method][config['n_components']]
        if algo == 'kmeans':
            model = KMeansCluster()
        else:
            model = EMCluster()

        labels = model.fit(X_transformed, config['k'])
        metrics = model.evaluate_with_ground_truth(y, labels)
        composition = pd.crosstab(labels, y, normalize='index') * 100
        results[key] = {
            'metrics': metrics,
            'composition': composition
        }

    return results


def run_combined_cluster(config: List[Dict], k_cluster: int, ground_truth: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    run the cluster on the transformed dataset from reduced dimensionality
    :param config: [{rd: str, X: np.ndarray}]
    :param k_cluster: int
    :return:
        - data frame of internal metrics | cluster method | rd method | silhouette score | calinski harabasz score|
        - external metrics. |cluster method | rd method | adjusted rand score | normalized mutual info score | homogeneity | completeness
        - predicted labels for each cluster method
    """
    metrics = pd.DataFrame()
    external_metrics = pd.DataFrame()
    predict_labels = {}
    for c in config:
        rd_method = c['rd']
        X = c['X']
        exp = ClusteringExperiment()
        kmeans_metrics, em_metrics = exp.run_clustering_analysis(X, k_cluster)
        kmeans_metrics = kmeans_metrics[['k', 'silhouette_score', 'calinski_harabasz_score']]
        kmeans_metrics['method'] = 'kmeans'
        kmeans_metrics['rd_method'] = rd_method

        em_metrics = em_metrics[['k', 'silhouette_score', 'calinski_harabasz_score']]
        em_metrics['method'] = 'em'
        em_metrics['rd_method'] = rd_method

        metrics = pd.concat([metrics, kmeans_metrics, em_metrics], ignore_index=True)

        # get the predicted labels
        kmeans_labels = exp.kmeans.fit(X, 2)
        em_labels = exp.em.fit(X, 2)
        predict_labels[rd_method] = {
            'kmeans': kmeans_labels,
            'em': em_labels
        }

        # get the external metrics
        kmeans_external_metrics = exp.kmeans.evaluate_with_ground_truth(ground_truth, kmeans_labels)
        kmeans_external_metrics['method'] = 'kmeans'
        kmeans_external_metrics['rd_method'] = rd_method

        em_external_metrics = exp.em.evaluate_with_ground_truth(ground_truth, em_labels)
        em_external_metrics['method'] = 'em'
        em_external_metrics['rd_method'] = rd_method

        external_metrics = pd.concat([external_metrics, pd.DataFrame([kmeans_external_metrics, em_external_metrics])], ignore_index=True)

    return metrics, predict_labels, external_metrics
