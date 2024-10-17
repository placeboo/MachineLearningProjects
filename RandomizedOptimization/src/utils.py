import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from itertools import cycle
from scipy.stats import alpha
from stack_data import markers_from_ranges


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


def load_results(problem_name, input_dir) -> pd.DataFrame:
    """
    Load the experiment results from a CSV file.

    :param problem_name: Name of the problem (e.g., 'four_peaks')
    :param input_dir: Directory containing the results
    :return: DataFrame containing the experiment results
    """
    filename = os.path.join(input_dir, f"{problem_name}_results.csv")
    return pd.read_csv(filename)


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

    #plt.figure(figsize=(5, 3.5))  # Slightly larger figure for better readability

    sns.lineplot(data=results, x='problem_size', y=metric, hue='algorithm', marker='o',alpha=0.6, markersize=4)

    plt.xlabel('Problem Size')
    ylabel = 'Best Fitness' if metric == 'best_fitness' else 'Execution Time (seconds)'
    plt.ylabel(ylabel)
    title = 'Best Fitness vs Problem Size' if metric == 'best_fitness' else 'Execution Time vs Problem Size'
    #plt.title(title)
    plt.legend(title='Algorithm', loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()

    save_plot(f"{metric}_vs_size", output_dir)
    # show
    plt.show()


def plot_convergence_vs_size(results: pd.DataFrame, output_dir: str = 'plots'):
    """
    Plot convergence speed vs problem size for all algorithms.

    :param results: DataFrame containing experiment results
    :param output_dir: Directory to save the plots
    """
    set_plot_style()

    # plt.figure(figsize=(5, 3.5))  # Slightly larger figure for better readability

    def calculate_convergence(row):
        fitness_curve = np.array(row['fitness_curve'])
        max_fitness = np.max(fitness_curve)
        return np.argmax(fitness_curve >= 0.95 * max_fitness)

    results['convergence_point'] = results.apply(calculate_convergence, axis=1)

    sns.lineplot(data=results, x='problem_size', y='convergence_point', hue='algorithm', marker='o')

    plt.xlabel('Problem Size')
    plt.ylabel('Iterations to 95% Max Fitness')
    #plt.title('Convergence Speed vs Problem Size')
    plt.legend(title='Algorithm', loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()

    save_plot("convergence_vs_size", output_dir)


def plot_fitness_vs_iteration(results: pd.DataFrame, output_dir: str = 'plots'):
    """
    Plot fitness vs iteration for each algorithm and problem size.
    """
    set_plot_style()

    # Create a separate plot for each algorithm
    for algorithm in results['algorithm'].unique():
        # plt.figure(figsize=(5, 3.5))

        # Filter data for the current algorithm
        alg_data = results[results['algorithm'] == algorithm]

        # Plot fitness curves for each problem size
        for _, row in alg_data.iterrows():
            fitness_curve = np.array(row['fitness_curve'])
            iterations = range(1, len(fitness_curve) + 1)
            fitness = [pair[0] for pair in fitness_curve]
            plt.plot(iterations, fitness, label=f"Size {row['problem_size']}")

        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        #plt.title(f'Best Fitness vs Iteration - {algorithm}')
        plt.legend(title='Algorithm', loc='best', frameon=True, fancybox=False, edgecolor='black')
        plt.tight_layout()

        # Save the plot
        save_plot(f"fitness_vs_iteration_{algorithm}", output_dir)


def plot_iterations_vs_time(results: pd.DataFrame, output_dir: str = 'plots'):
    """
    Plot iterations vs execution time for all algorithms and problem sizes.
    """
    set_plot_style()

    # plt.figure(figsize=(8, 6))

    algorithms = results['algorithm'].unique()
    markers = ['o', 's', '^', 'D']  # Different markers for different problem sizes

    for alg, marker in zip(algorithms, markers):
        alg_data = results[results['algorithm'] == alg]
        plt.scatter(alg_data['execution_time'], alg_data['iterations'],
                    label=alg, marker=marker, s=30, alpha=0.6)

    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Iterations')
    #plt.title('Iterations vs Execution Time')
    plt.legend(title='Algorithm')
    plt.legend(title='Algorithm', loc='best', frameon=True, fancybox=False, edgecolor='black')

    # Add problem size annotations
    for _, row in results.iterrows():
        plt.annotate(f"{row['problem_size']}",
                     (row['execution_time'], row['iterations']),
                     xytext=(5, 5), textcoords='offset points', fontsize=6)

    plt.tight_layout()
    save_plot("iterations_vs_time", output_dir)

def plot_fitness_vs_iteration_wsize(results, size, fig_dir):
    set_plot_style()
    sub_results = results[results['problem_size'] == size]

    plt.figure()
    for algo in sub_results['algorithm'].unique():
        sub_results_algo = sub_results[sub_results['algorithm'] == algo]
        for i, row in sub_results_algo.iterrows():
            fitness_values = parse_fitness_curve(row['fitness_curve'])
            plt.plot(fitness_values, label=f'{algo}', alpha=0.6)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    #plt.title(f'Fitness vs Iteration - Size {size}')
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    save_plot(f'fitness_vs_iteration_{size}', fig_dir)


def load_processed_data(output_dir):
    X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def plot_best_model_comparison(best_results: pd.DataFrame, output_dir: str = 'plots'):
    set_plot_style()
    # compare validation f1 scores
    width = 0.3
    x = np.arange(len(best_results))
    plt.bar(x - width/2.0, best_results['train_accuracy'], width=width, label='Train', color='r', alpha=0.5)
    plt.bar(x + width/2.0, best_results['val_accuracy'], width=width, label='Val', color='b', alpha=0.5)

    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    #plt.title('Best Model Performance Comparison')
    plt.xticks(x, ['RHC', 'SA', 'GA'])
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    # Add value labels on top of each bar
    for i, v in enumerate(best_results['train_accuracy']):
        plt.text(i - width/2.0, v, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(best_results['val_accuracy']):
        plt.text(i + width/2.0, v, f'{v:.3f}', ha='center', va='bottom')
    plt.ylim(0, 1)
    plt.tight_layout()
    save_plot("best_model_comparison", output_dir)

def plot_nn_fitness_curves(best_results_df: pd.DataFrame, output_dir: str = 'plots'):
    """
    Plot fitness curves for the best models of each algorithm.

    :param best_results_df: DataFrame containing the best results and fitness curves
    :param output_dir: Directory to save the plots
    """
    set_plot_style()

    for _, row in best_results_df.iterrows():
        alg = row['algorithm']
        fitness_value = parse_fitness_curve(row['fitness_curve'])
        iterations = range(1, len(fitness_value) + 1)
        plt.plot(iterations, fitness_value, label=alg)

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    #plt.title('Fitness Curves for Best Models')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

    save_plot("nn_fitness_curves", output_dir)


def parse_fitness_curve(fitness_curve_str: str) -> tuple:
    """
    Parse the string representation of the fitness curve and return fitness values and iteration values.

    :param fitness_curve_str: String representation of the fitness curve
    :return: Tuple of (fitness_values, iteration_values) as NumPy arrays
    """
    # Remove the "array(" prefix and ")" suffix if present
    cleaned_str = fitness_curve_str.strip("array(").rstrip(")")

    # Remove any single quotes around the inner list
    cleaned_str = cleaned_str.strip("'")
    # Use ast.literal_eval to safely evaluate the string as a Python literal
    fitness_curve_list = ast.literal_eval(cleaned_str)
    # Convert the list to a NumPy array
    fitness_curve_array = np.array(fitness_curve_list)
    # Extract fitness values and iteration values
    fitness_values = fitness_curve_array[:, 0]

    return fitness_values


def plot_parameter_vs_metric(df, algorithm, primary_param, metric, secondary_param=None, output_dir='figures'):
    set_plot_style()
    alg_df = df[df['algorithm'] == algorithm]

    # Handle schedule parameter
    for param in [primary_param, secondary_param]:
        if param == 'param_schedule':
            alg_df[param] = alg_df[param].apply(lambda x: 'GeomDecay' if str(x) == '1.0' else str(x).split('(')[0])

    if pd.api.types.is_numeric_dtype(alg_df[primary_param]):
        if secondary_param:
            for secondary_value in alg_df[secondary_param].unique():
                subset = alg_df[alg_df[secondary_param] == secondary_value]
                plt.plot(subset[primary_param], subset[metric], 'o-', label=secondary_value, alpha=0.6, markersize=4)
        else:
            plt.plot(alg_df[primary_param], alg_df[metric], 'o-', alpha=0.6, markersize=4)
    else:
        if secondary_param:
            sns.barplot(data=alg_df, x=primary_param, y=metric, hue=secondary_param, alpha=0.6)
        else:
            sns.barplot(data=alg_df, x=primary_param, y=metric, alpha=0.6)

    #plt.title(f'{algorithm}: {metric} vs {primary_param.replace("param_", "")}')
    plt.xlabel(primary_param.replace("param_", "").replace("_", " ").title())
    plt.ylabel(metric.replace("_", " ").title())

    if secondary_param:
        plt.legend(title=secondary_param.replace("param_", "").replace("_", " ").title(),
                   loc='best', frameon=True, fancybox=False, edgecolor='black')

    # Rotate x-axis labels if they are not numeric
    if not pd.api.types.is_numeric_dtype(alg_df[primary_param]):
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_plot(f'{algorithm}_{metric}_vs_{primary_param}', output_dir)
    plt.close()


def plot_fitness_curve_with_params(df, algorithm, params, output_dir):
    set_plot_style()

    alg_df = df[df['algorithm'] == algorithm].copy()
    if 'schedule' in params:
        alg_df['param_schedule'] = alg_df['param_schedule'].apply(
            lambda x: 'GeomDecay' if str(x) == '1.0' else str(x).split('(')[0])

    param_columns = [f'param_{param}' for param in params]
    alg_df['param_combination'] = alg_df[param_columns].apply(
        lambda row: '_'.join(f"{col.replace('param_', '')}:{row[col]}" for col in param_columns),
        axis=1
    )
    # Plot fitness curves for each parameter combination
    for param_combo, group in alg_df.groupby('param_combination'):
        for _, row in group.iterrows():
            fitness_values = parse_fitness_curve(row['fitness_curve'])
            plt.plot(fitness_values, label=param_combo, alpha=0.6)

    #plt.title(f'{algorithm}: Fitness Curves for Different Parameters')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend(title='Parameter', loc='best', frameon=True, fancybox=False, edgecolor='black')
    plt.tight_layout()
    save_plot(f'{algorithm}_fitness_curves', output_dir)
    plt.close()
