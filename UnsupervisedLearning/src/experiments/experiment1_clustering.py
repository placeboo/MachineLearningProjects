import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import logging

import pandas as pd

from src.clustering.kmeans import KMeansCluster
from src.clustering.em import EMCluster

class ClusteringExperiment:
    def __init__(self, random_state: int=17):
        self.random_state = random_state
        self.kmeans = KMeansCluster(random_state=self.random_state)
        self.em = EMCluster(random_state=self.random_state)

    def run_clustering_analysis(self, X: np.ndarray, k_range: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run clustering analysis for both EM and KMeans."""
        kmeans_metircs = {
            'k': k_range,
            'inertia': [],
            'silhouette_score': [],
            'calinski_harabasz_score': []
        }
        em_metircs = {
            'k': k_range,
            'bic': [],
            'aic': [],
            'silhouette_score': [],
            'calinski_harabasz_score': []
        }

        for k in k_range:
            # KMeans
            kmeans_labels = self.kmeans.fit(X, k)
            kmeans_metrics = self.kmeans.get_metrics(X, kmeans_labels)
            kmeans_metircs['inertia'].append(kmeans_metrics['inertia'])
            kmeans_metircs['silhouette_score'].append(kmeans_metrics['silhouette_score'])
            kmeans_metircs['calinski_harabasz_score'].append(kmeans_metrics['calinski_harabasz_score'])

            # EM
            em_labels = self.em.fit(X, k)
            em_metrics = self.em.get_metrics(X, em_labels)
            em_metircs['bic'].append(em_metrics['bic'])
            em_metircs['aic'].append(em_metrics['aic'])
            em_metircs['silhouette_score'].append(em_metrics['silhouette_score'])
            em_metircs['calinski_harabasz_score'].append(em_metrics['calinski_harabasz_score'])

        return pd.DataFrame(kmeans_metircs), pd.DataFrame(em_metircs)


    def evaluate_clustering(self, X: np.ndarray, y: np.ndarray, optimal_k: Dict) -> Dict:
        """
        Evaluate clustering results using ground truth labels
        Should only be called after optimal k is chosen using unsupervised metrics
        Args:
            X (np.ndarray): Data
            y (np.ndarray): Ground truth labels
            optimal_k (Dict): Dictionary of optimal k for KMeans and EM {kmeans: int, em: int}
        Returns:
            Dict: Dictionary of evaluation metrics for KMeans and EM
        """
        results = {}

        # KMeans
        kmeans_labels = self.kmeans.fit(X, optimal_k['kmeans'])
        kmeans_metrics = self.kmeans.evaluate_with_ground_truth(y, kmeans_labels)
        results['kmeans'] = kmeans_metrics

        # EM
        em_labels = self.em.fit(X, optimal_k['em'])
        em_metrics = self.em.evaluate_with_ground_truth(y, em_labels)
        results['em'] = em_metrics

        return results

    def analyze_cluster_composition(self, labels: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
        """
        Analyze the composition of each cluster in terms of true labels

        Parameters:
        -----------
        labels : np.ndarray
            Cluster assignments
        y_true : np.ndarray
            Ground truth labels

        Returns:
        --------
        pd.DataFrame : Cluster composition analysis
        """
        # Create cross-tabulation of clusters and true labels
        df = pd.DataFrame({
            'Cluster': labels,
            'True_Label': y_true
        })

        # Calculate composition percentages
        composition = pd.crosstab(
            df['Cluster'],
            df['True_Label'],
            normalize='index'
        ) * 100

        # Add cluster sizes
        cluster_sizes = df['Cluster'].value_counts().sort_index()
        composition['Size'] = cluster_sizes

        return composition