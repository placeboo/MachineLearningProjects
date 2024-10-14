from src.optimization import run_nn_experiment
from src.utils import save_results, load_processed_data, save_plot, set_plot_style, plot_best_model_comparison
import numpy as np
import mlrose_ky as mlrose
import pandas as pd
import os
import matplotlib.pyplot as plt



def main(data_dir, hidden_nodes, hyperparameters, results_dir, fig_dir):
    # load data
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    # run experiment
    algorithms = list(hyperparameters.keys())
    results, best_models = run_nn_experiment(X_train, y_train, X_test, y_test, hidden_nodes, algorithms,
                                             hyperparameters)

    # save full results
    save_results(results, 'nn_full', results_dir)
    print(f'Full experiment results saved to {results_dir}/nn_full_results.csv')

    # save and plot best model results
    best_results = []
    for alg, (model, params,train_metrics, val_metrics, test_metrics, fitness_curve) in best_models.items():
        best_results.append({
            'algorithm': alg,
            'best_params': str(params),
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'fitness_curve': fitness_curve,
        })

    best_results_df = pd.DataFrame(best_results)
    save_results(best_results_df, 'nn_best', results_dir)
    print(f'Best model results saved to {results_dir}/nn_best_results.csv')

    # plot best model results
    plot_best_model_comparison(best_results_df, fig_dir)




if __name__ == "__main__":
    #data_dir = 'data/processed_feature_reduce'
    data_dir = 'data/spam'
    hidden_nodes = [100]

    # #Define hyperparameters to explore
    # hyperparameters = {
    #     'random_hill_climb': {
    #         'restarts': [5, 10, 20],
    #         'max_iters': [10, 50, 100, 200, 500]
    #     },
    #     'simulated_annealing': {
    #         'schedule': [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()],
    #         'max_attempts': [10, 20, 50],
    #          'max_iters': [10, 50, 100, 200, 500]
    #     },
    #     'genetic_alg': {
    #         'pop_size': [100, 200, 400],
    #         'mutation_prob': [0.1, 0.2, 0.3],
    #         'max_attempts': [10, 20, 50],
    #          'max_iters': [10, 50, 100, 200, 500]
    #     }
    # }

    hyperparameters = {
        'random_hill_climb': {
            'restarts': [5, 10, 20],
            'max_attempts': [100]
        },
        'simulated_annealing': {
            'schedule': [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()],
            'max_attempts': [10, 20, 50, 100],
        },
        'genetic_alg': {
            'pop_size': [100, 200, 400],
            'mutation_prob': [0.1, 0.2, 0.3, 0.4],
            'max_attempts': [100],
        }
    }



    results_dir = 'results/nn/spam'
    fig_dir = 'figures/nn/spam'

    main(data_dir, hidden_nodes, hyperparameters, results_dir, fig_dir)