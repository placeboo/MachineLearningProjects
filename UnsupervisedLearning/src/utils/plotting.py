import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from typing import Optional, Tuple, Dict
import numpy as np
from sklearn.manifold import TSNE

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


def plot_train_val_score(
        data: pd.DataFrame,
        x_col: str='n_components',
        y_train: str='mean_train_score',
        y_val: str='mean_test_score',
        output_dir: str = 'figs',
        dataset: str = 'dataset1',
        experiment: str = 'experiment4',
        filename: Optional[str] = None,
        algo_name: Optional[str] = None,
):
    set_plot_style()
    plt.figure()
    plt.plot(data[x_col], data[y_train], 'o-', label='Train Score')
    plt.plot(data[x_col], data[y_val], 'o-', label='Validation Score')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    output_dir = f'{output_dir}/{dataset}/{experiment}'
    filename = filename or f'train_val_score_{algo_name}'
    save_plot(output_dir, filename)


def visualize_clusters_tsne(
        X: np.ndarray,
        labels: np.ndarray,
        perplexity: float = 30.0,
        n_components: int = 2,
        random_state: int = 42,
        sample_size: Optional[int] = 5000,
        output_dir: str = 'figs',
        dataset: str = 'dataset1',
        experiment: str = 'experiment1',
        algorithm: str = 'kmeans',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    Visualize clustering results using t-SNE dimensionality reduction.

    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels from clustering algorithm
    true_labels : Optional[np.ndarray]
        True labels for comparison (if available)
    perplexity : float
        t-SNE perplexity parameter
    n_components : int
        Number of components for t-SNE (2 or 3)
    random_state : int
        Random state for reproducibility
    sample_size : Optional[int]
        Number of samples to use for visualization (None for all samples)
    output_dir : str
        Directory to save the plots
    dataset : str
        Name of the dataset
    experiment : str
        Name of the experiment
    algorithm : str
        Name of the clustering algorithm
    title : Optional[str]
        Custom title for the plot
    figsize : Tuple[int, int]
        Figure size for the plots
    """
    # Sample data if needed
    if sample_size and len(X) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X[indices]
        labels = labels[indices]

    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    X_tsne = tsne.fit_transform(X)

    # Set style
    set_plot_style()

    # Create figure

    fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))

    # Plot predicted clusters
    scatter1 = ax1.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap='Set1',
        alpha=0.3,
        s=50,
        edgecolors='w',
    )
    #ax1.set_title(f'Predicted Clusters ({algorithm.upper()})')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    legend1 = ax1.legend(*scatter1.legend_elements(),
                         title="Clusters",
                         loc="best")
    ax1.add_artist(legend1)

    # Plot true labels if available
    # if true_labels is not None:
    #     scatter2 = ax2.scatter(
    #         X_tsne[:, 0],
    #         X_tsne[:, 1],
    #         c=true_labels,
    #         cmap='Set1',
    #         alpha=0.3,
    #         s=50
    #     )
    #     ax2.set_title('True Labels')
    #     ax2.set_xlabel('t-SNE Component 1')
    #     ax2.set_ylabel('t-SNE Component 2')
    #     legend2 = ax2.legend(*scatter2.legend_elements(),
    #                          title="True Labels",
    #                          loc="best")
    #     ax2.add_artist(legend2)

    plt.tight_layout()

    # Save plot
    output_dir = f'{output_dir}/{dataset}/{experiment}'
    filename = f'tsne_visualization_{algorithm}'
    save_plot(output_dir, filename)


def plot_learning_curves(df, keys, output_dir='figs', dataset='dataset1',
                         experiment='experiment4', figsize=(10, 6), colors=None, filename='learning_curves'):
    """
    Plot learning curves for multiple algorithms.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing learning curve data organized by organize_learning_curve()
    keys : list
        List of algorithm keys in the results
    output_dir : str
        Directory to save the plot
    dataset : str
        Name of the dataset
    experiment : str
        Name of the experiment
    figsize : tuple
        Figure size (width, height)
    colors : dict
        Dictionary mapping keys to colors. If None, will use default color cycle
    filename : str
        Name of the output file
    """
    set_plot_style()
    plt.figure(figsize=figsize)

    # Default colors if none provided
    if colors is None:
        colors = {
            'base': '#000000',  # black
            'pca': '#1f77b4',  # blue
            'ica': '#ff7f0e',  # orange
            'rp': '#2ca02c',  # green
            'kmeans': '#d62728',  # red
            'em': '#9467bd',  # purple
        }

    for key in keys:
        # Plot training score
        plt.plot(df['train_sizes'],
                 1 - df[f'{key}_train_score_mean'],
                 label=f'{key.upper()} (Train)',
                 color=colors.get(key.lower(), None),
                 linestyle='-',
                 linewidth=2,
                 marker='o',
                 markersize=5,
                 markerfacecolor=colors.get(key.lower(), None),  # white fill
                 markeredgecolor=colors.get(key.lower(), None),
                 alpha=0.7
                 )

        # Add error bands for training score
        # plt.fill_between(df['train_sizes'],
        #                 1 - (df[f'{key}_train_score_mean'] - df[f'{key}_train_score_std']),
        #                 1 - (df[f'{key}_train_score_mean'] + df[f'{key}_train_score_std']),
        #                 alpha=0.1,
        #                 color=colors.get(key.lower(), None))

        # Plot validation score
        plt.plot(df['train_sizes'],
                 1 - df[f'{key}_val_score_mean'],
                 label=f'{key.upper()} (Val)',
                 color=colors.get(key.lower(), None),
                 linestyle='--',
                 linewidth=2,
                 marker='o',
                 markersize=5,
                 markerfacecolor=colors.get(key.lower(), None),  # white fill
                 markeredgecolor=colors.get(key.lower(), None),
                 alpha=0.7
                 )

        # Add error bands for validation score
        # plt.fill_between(df['train_sizes'],
        #                 1 - (df[f'{key}_val_score_mean'] - df[f'{key}_val_score_std']),
        #                 1 - (df[f'{key}_val_score_mean'] + df[f'{key}_val_score_std']),
        #                 alpha=0.1,
        #                 color=colors.get(key.lower(), None))

    plt.xlabel('Train Size')
    plt.ylabel('Error Rate')
    plt.title('Learning Curves')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    output_dir = f'{output_dir}/{dataset}/{experiment}'
    save_plot(output_dir, filename)