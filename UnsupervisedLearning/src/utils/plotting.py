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
        'completeness': 'Completeness'
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
        marker_size: int = 3,
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


def plot_2d_projection(
        X_transformed: np.ndarray,
        y: np.ndarray,
        output_dir: str = 'figs',
        dataset: str = 'dataset1',
        experiment: str = 'experiment1',
        filename: Optional[str] = None,
        algo_name: Optional[str] = None,
        sample_size: int = 1000,
        random_state: int = 17
        ):
    """
        Plot 2D projection of data with sampling and binary label handling.

        Parameters:
        -----------
        X_transformed : np.ndarray
            Transformed data (n_samples, n_components)
        y : np.ndarray
            Labels
        output_dir : str
            Output directory for saving plots
        dataset : str
            Dataset name
        experiment : str
            Experiment name
        filename : Optional[str]
            Custom filename for the plot
        algo_name : Optional[str]
            Algorithm name for default filename
        sample_size : Optional[int]
            Number of points to sample. If None, use all points
        random_state : int
            Random state for reproducibility
    """
    set_plot_style()
    # Sample data if needed
    if sample_size and len(X_transformed) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_transformed), sample_size, replace=False)
        X_transformed = X_transformed[indices]
        y = y[indices]
    # Handle binary labels
    unique_labels = np.unique(y)
    if len(unique_labels) == 2:
        # Create a scatter plot for each class
        colors = ['#2ecc71', '#e74c3c']  # Green and Red
        labels = ['Class 0', 'Class 1']

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = y == label
            plt.scatter(X_transformed[mask, 0],
                        X_transformed[mask, 1],
                        c=color,
                        label=labels[i],
                        alpha=0.6,
                        edgecolors='white',
                        linewidth=0.5)

        plt.legend()
    else:
        # Multi-class case
        scatter = plt.scatter(X_transformed[:, 0],
                              X_transformed[:, 1],
                              c=y,
                              cmap='viridis',
                              alpha=0.6,
                              edgecolors='white',
                              linewidth=0.5)
        plt.colorbar(scatter, label='Class')

    # Labels and title
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    # Make the plot more visually appealing
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    filename = filename or f'2d_projection_{algo_name}'
    output_dir = f'{output_dir}/{dataset}/{experiment}'
    save_plot(output_dir, filename)


def plot_3d_projection(
        X_transformed: np.ndarray,
        y: np.ndarray,
        output_dir: str = 'figs',
        dataset: str = 'dataset1',
        experiment: str = 'experiment1',
        filename: Optional[str] = None,
        algo_name: Optional[str] = None,
        sample_size: Optional[int] = 1000,
        random_state: int = 42,
        elev: int = 30,
        azim: int = 45):
    """
    Plot 3D projection of data with sampling and binary label handling.

    Parameters:
    -----------
    X_transformed : np.ndarray
        Transformed data (n_samples, n_components)
    y : np.ndarray
        Labels
    output_dir : str
        Output directory for saving plots
    dataset : str
        Dataset name
    experiment : str
        Experiment name
    filename : Optional[str]
        Custom filename for the plot
    algo_name : Optional[str]
        Algorithm name for default filename
    sample_size : Optional[int]
        Number of points to sample. If None, use all points
    random_state : int
        Random state for reproducibility
    elev : int
        Elevation angle for 3D plot
    azim : int
        Azimuth angle for 3D plot
    """
    if X_transformed.shape[1] < 3:
        raise ValueError("Transformed data must have at least 3 dimensions")

    set_plot_style()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Sample data if needed
    if sample_size and len(X_transformed) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_transformed), sample_size, replace=False)
        X_transformed = X_transformed[indices]
        y = y[indices]

    # Handle binary labels
    unique_labels = np.unique(y)
    if len(unique_labels) == 2:
        # Create a scatter plot for each class
        colors = ['#2ecc71', '#e74c3c']  # Green and Red
        labels = ['Class 0', 'Class 1']

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = y == label
            ax.scatter(X_transformed[mask, 0],
                       X_transformed[mask, 1],
                       X_transformed[mask, 2],
                       c=color,
                       label=labels[i],
                       alpha=0.6,
                       edgecolors='white',
                       linewidth=0.5)

        ax.legend()
    else:
        # Multi-class case
        scatter = ax.scatter(X_transformed[:, 0],
                             X_transformed[:, 1],
                             X_transformed[:, 2],
                             c=y,
                             cmap='viridis',
                             alpha=0.6,
                             edgecolors='white',
                             linewidth=0.5)
        fig.colorbar(scatter, label='Class', ax=ax)

    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Labels and title
    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_zlabel('Third Component')
    #plt.title(f'3D Projection - {algo_name.upper() if algo_name else ""}')
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Make the plot more visually appealing
    # Make panes slightly transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.9))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.9))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.9))

    # Make grid lines lighter
    ax.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.1)
    ax.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.1)
    ax.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.1)

    # Save plot
    filename = filename or f'3d_projection_{algo_name}'
    output_dir = f'{output_dir}/{dataset}/{experiment}'
    save_plot(output_dir, filename)