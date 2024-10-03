import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_results(results: pd.DataFrame, problem_name: str, output_dir: str = 'results'):
    """
    Save the experiment results to a CSV file.

    :param results: DataFrame containing experiment results
    :param problem_name: Name of the problem (e.g., 'four_peaks')
    :param output_dir: Directory to save the results (default is 'results')
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{problem_name}_results.csv")
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def load_results(problem_name: str, input_dir: str = 'results') -> pd.DataFrame:
    """
    Load the experiment results from a CSV file.

    :param problem_name: Name of the problem (e.g., 'four_peaks')
    :param input_dir: Directory to load the results from (default is 'results')
    :return: DataFrame containing experiment results
    """
    filename = os.path.join(input_dir, f"{problem_name}_results.csv")
    results = pd.read_csv(filename)
    results['best_state'] = results['best_state'].apply(eval)
    results['fitness_curve'] = results['fitness_curve'].apply(eval)
    return results


def set_plot_style():
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (3.5, 2.5),  # Single column width for IEEE
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': 'white',
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
    })


def save_plot(plot_type: str, output_dir: str = 'plots'):
    """
    Save the current plot with a standardized name.

    :param plot_type: Type of the plot (e.g., 'fitness', 'execution_time')
    :param output_dir: Directory to save the plots (default is 'plots')
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{plot_type}_vs_problem_size.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filepath}")
    plt.close()


def plot_performance_vs_size(results: pd.DataFrame, metric: str, output_dir: str = 'plots'):
    """
    Plot performance metric vs problem size for all algorithms.

    :param results: DataFrame containing experiment results
    :param metric: 'best_fitness' or 'execution_time'
    :param output_dir: Directory to save the plots
    """
    set_plot_style()

    plt.figure(figsize=(5, 3.5))  # Slightly larger figure for better readability

    sns.lineplot(data=results, x='problem_size', y=metric, hue='algorithm', marker='o')

    plt.xlabel('Problem Size')
    ylabel = 'Best Fitness' if metric == 'best_fitness' else 'Execution Time (seconds)'
    plt.ylabel(ylabel)
    title = 'Best Fitness vs Problem Size' if metric == 'best_fitness' else 'Execution Time vs Problem Size'
    plt.title(title)
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_plot(f"{metric}_vs_size", output_dir)


def plot_convergence_vs_size(results: pd.DataFrame, output_dir: str = 'plots'):
    """
    Plot convergence speed vs problem size for all algorithms.

    :param results: DataFrame containing experiment results
    :param output_dir: Directory to save the plots
    """
    set_plot_style()

    plt.figure(figsize=(5, 3.5))  # Slightly larger figure for better readability

    def calculate_convergence(row):
        fitness_curve = np.array(row['fitness_curve'])
        max_fitness = np.max(fitness_curve)
        return np.argmax(fitness_curve >= 0.95 * max_fitness)

    results['convergence_point'] = results.apply(calculate_convergence, axis=1)

    sns.lineplot(data=results, x='problem_size', y='convergence_point', hue='algorithm', marker='o')

    plt.xlabel('Problem Size')
    plt.ylabel('Iterations to 95% Max Fitness')
    plt.title('Convergence Speed vs Problem Size')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_plot("convergence_vs_size", output_dir)