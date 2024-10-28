import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from typing import Optional, Tuple, Dict
import numpy as np

def set_plot_style():
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (4, 3),  # Single column width for IEEE
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'figure.facecolor': 'white',
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })


def save_plot(output_dir: str, filename: str, dpi: int = 600):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    file_path = file_path + '.png'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved at {file_path}")
    plt.close()


def get_metric_label(metric_col: str) -> str:
    """Get formatted label for different metrics"""
    metric_labels = {
        'silhouette_score': 'Silhouette Score',
        'calinski_harabasz_score': 'Calinski-Harabasz Score',
        'inertia': 'Inertia',
        'aic': 'AIC Score',
        'bic': 'BIC Score',
        'reconstruction_error': 'Reconstruction Error',
        'explained_variance': 'Explained Variance Ratio',
        'kurtosis': 'Kurtosis'
    }
    return metric_labels.get(metric_col, metric_col.replace('_', ' ').title())

def get_algorithm_style(algorithm: str) -> Tuple[str, str, str]:
    """Get consistent style for different algorithms"""
    styles = {
        'kmeans': ('o-', '#1f77b4', 'K-Means'),
        'em': ('s-', '#ff7f0e', 'EM'),
        'pca': ('d-', '#2ca02c', 'PCA'),
        'ica': ('^-', '#d62728', 'ICA'),
        'rp': ('v-', '#9467bd', 'Random Projection')
    }
    return styles.get(algorithm.lower(), ('o-', '#333333', algorithm))

def get_supervise_metrics(metric: str) -> str:
    """Get supervised metrics"""
    metrics = {
        'adjusted_rand': 'ARI',
        'normalized_mutual_info': 'NMI',
        'adjusted_mutual_info': 'AMI',
        'homogeneity': 'Homogeneity',
        'completeness': 'Completeness',
        'v_measure': 'V-Measure'
    }
    return metrics.get(metric, metric.replace('_', ' ').title())

def plot_metrics_vs_cluster(
        df: pd.DataFrame,
        metric_col: str,
        k_col: str,
        group_col: Optional[str] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        legend_loc: str = 'best',
        marker_size: int = 6,
        line_width: float = 1.5,
        output_dir: str = 'figs',
        dataset: str = 'dataset1',
        experiment: str = 'experiment1',
        filename: Optional[str] = None,
        algo_name: Optional[str] = None
):
    """
    Plot metrics vs number of clusters with optional grouping

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the metrics data
    metric_col : str
        Column name for the metric to plot
    k_col : str
        Column name for the number of clusters
    group_col : Optional[str]
        Column name for grouping (e.g., algorithm)
    title : Optional[str]
        Plot title (if None, will be generated from metric name)
    x_label : Optional[str]
        X-axis label (if None, will use default)
    y_label : Optional[str]
        Y-axis label (if None, will use metric name)
    y_lim : Optional[Tuple[float, float]]
        Y-axis limits
    legend_loc : str
        Legend location
    marker_size : int
        Size of markers
    line_width : float
        Width of lines
    output_dir : str
        Output directory for saving plots
    dataset : str
        Dataset name
    experiment : str
        Experiment name
    filename : Optional[str]
        Filename to save plot
    algo_name : Optional[str], if group_col is None, use this as algorithm name.
    """
    set_plot_style()

    plt.figure()

    # Get algorithm names
    if group_col:
        algorithms = sorted(df[group_col].unique())
    else:
        algorithms = [algo_name]
    algorithms_str = '_'.join([str(a) for a in algorithms])
    if group_col:
        for algorithm in sorted(df[group_col].unique()):
            mask = df[group_col] == algorithm
            style, color, label = get_algorithm_style(algorithm)

            plt.plot(df[mask][k_col], df[mask][metric_col],
                     style, label=label, markersize=marker_size,
                     linewidth=line_width, color=color)
    else:
        plt.plot(df[k_col], df[metric_col], 'o-',
                 markersize=marker_size, linewidth=line_width)

    # Set labels and title
    plt.xlabel(x_label or 'Number of Clusters ($k$)')
    plt.ylabel(y_label or get_metric_label(metric_col))
    if title:
        plt.title(title)

    # Customize plot
    if y_lim:
        plt.ylim(y_lim)
    if group_col:
        plt.legend(loc=legend_loc, frameon=True)

    # Ensure integer ticks for k
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    # Save plot
    output_dir = f'{output_dir}/{dataset}/{experiment}'
    filename = filename or f'{metric_col}_vs_{k_col}_{algorithms_str}'
    filename = filename.replace(' ', '_').lower()
    save_plot(output_dir, filename)
    plt.show()


def plot_cluster_evaluation(
        eval_results: Dict,
        dataset: str = 'dataset1',
        experiment: str = 'experiment1'):

    set_plot_style()
    metrics = list(eval_results['kmeans'].keys())
    algorithms = list(eval_results.keys())

    # prepare data
    data = {
        'Metric': [],
        'Algorithm': [],
        'Score': []
    }

    for algo in algorithms:
        for metric in metrics:
            data['Metric'].append(metric)
            data['Algorithm'].append(algo)
            data['Score'].append(eval_results[algo][metric])

    df = pd.DataFrame(data)
    # plot
    bar_width = 0.35
    index = np.arange(len(metrics))

    plt.bar(index, df[df['Algorithm'] == 'kmeans']['Score'],
            bar_width, label='K-Means')
    plt.bar(index + bar_width, df[df['Algorithm'] == 'em']['Score'],
            bar_width, label='EM')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    metrics = [get_supervise_metrics(metric) for metric in metrics]
    #plt.title('Clustering Evaluation Metrics Comparison')
    plt.xticks(index + bar_width / 2, metrics, rotation=45, ha='right')
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()

    output_dir = f'figs/{dataset}/{experiment}'
    filename = 'clustering_evaluation_metrics_comparison'
    save_plot(output_dir, filename)
    plt.show()


