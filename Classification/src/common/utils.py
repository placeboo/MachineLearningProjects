import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import json
import time

import pandas as pd
from sklearn.model_selection import learning_curve
import pickle

# save the model
def save_model(model, save_dir, model_name, dataset_name):
    """
    save the sklearn model
    :param model: the sklearn model
    :param save_dir: the directory to save the model
    :param model_name: the name of the model: knn, svm, nn
    :param dataset_name: the name of the datase: dataset1 or dataset2
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    model_file_path = f'{save_dir}/{model_name}_{dataset_name}.joblib'
    joblib.dump(model, model_file_path)
    print(f'Model saved successfully at {model_file_path}')

# load the model
def load_model(load_dir, model_name, dataset_name):

    model_file_path = f'{load_dir}/{model_name}_{dataset_name}.joblib'
    model = joblib.load(model_file_path)
    print(f'Model loaded successfully from {model_file_path}')
    return model


def save_cv_results(cv_results, save_dir, model_name, dataset_name):
    if os.path.exists(save_dir):
        cv_results_file_path = f'{save_dir}/{model_name}_{dataset_name}_cv_results.pkl'
        with open(cv_results_file_path, 'wb') as f:
            pickle.dump(cv_results, f)
        print(f'Cross-validation results saved successfully at {cv_results_file_path}')

def load_cv_results(load_dir, model_name, dataset_name):
    cv_results_file_path = f'{load_dir}/{model_name}_{dataset_name}_cv_results.pkl'
    with open(cv_results_file_path, 'rb') as f:
        cv_results = pickle.load(f)
    print(f'Cross-validation results loaded successfully from {cv_results_file_path}')
    return cv_results

# save plt figure
def save_plot(plt, save_dir, model_name, plot_name, dataset_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plot_file_path = f'{save_dir}/{model_name}_{plot_name}_{dataset_name}.png'
    plt.savefig(plot_file_path)
    print(f'Plot saved successfully at {plot_file_path}')

# save evaluation metrics
def save_metrics(metrics, save_dir, model_name, dataset_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    metrics_file_path = f'{save_dir}/{model_name}_{dataset_name}_metrics.json'
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f)
    print(f'Metrics saved successfully at {metrics_file_path}')

def load_metrics(load_dir, model_name, dataset_name):
    metrics_file_path = f'{load_dir}/{model_name}_{dataset_name}_metrics.json'
    with open(metrics_file_path, 'r') as f:
        metrics = json.load(f)
    print(f'Metrics loaded successfully from {metrics_file_path}')
    return metrics

def measure_time(fnc):
    """
    Decorator to measure the time of a function
    :param fnc: the function to measure
    :return: the wrapper function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fnc(*args, **kwargs)
        end = time.time()
        print(f'{fnc.__name__} took {end - start} seconds')
        return result
    return wrapper
############################################
#  Plots and Visualizations
############################################
def plot_leanring_curve(model, X, y, cv, train_size=np.linspace(0.1, 1.0, 5)):
    """
    plot the learning curve: error vs training size and fit time vs training size
    :param model: the best model from GridSearchCV
    :param X: the X_train
    :param y: the y_train
    :param cv: the k-fold
    :param train_size: relative training size
    """
    train_sizes, train_scores, val_scores, fit_times, _ = learning_curve(model, X, y, train_sizes=train_size, cv=cv, n_jobs=-1, random_state=17, verbose=2, return_times=True)

    train_error_mean = 1 - np.mean(train_scores, axis=1)
    val_error_mean = 1 - np.mean(val_scores, axis=1)
    fit_times = np.mean(fit_times, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_error_mean, label='Training error', color='r', linestyle='--', marker='o')
    ax.plot(train_sizes, val_error_mean, label='Validation error', color='b', linestyle='-', marker='o')
    ax.set_xlabel('Training size')
    ax.set_ylabel('Error')
    ax.set_title('Learning Curve')
    ax.legend()

    # plot the fit time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(train_sizes, fit_times, label='Fit time', color='g', linestyle='--', marker='o')
    ax2.set_xlabel('Training size')
    ax2.set_ylabel('Fit time')
    ax2.set_title('Fit Time')
    ax2.legend()

    return fig, fig2

def format_cv_results(cv_results):
    """
    :param cv_results: the cv_results from GridSearchCV
    :return: a pandas DataFrame
    """
    params = cv_results['params']
    mean_fit_time = cv_results['mean_fit_time']
    std_fit_time = cv_results['std_fit_time']
    mean_score_time = cv_results['mean_score_time']
    std_score_time = cv_results['std_score_time']
    mean_test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']
    mean_train_score = cv_results['mean_train_score']
    std_train_score = cv_results['std_train_score']

    results = pd.DataFrame(params)
    results['mean_fit_time'] = mean_fit_time
    results['std_fit_time'] = std_fit_time
    results['mean_score_time'] = mean_score_time
    results['std_score_time'] = std_score_time
    results['mean_test_score'] = mean_test_score
    results['std_test_score'] = std_test_score
    results['mean_train_score'] = mean_train_score
    results['std_train_score'] = std_train_score
    results['mean_train_error'] = 1 - results['mean_train_score']
    results['mean_test_error'] = 1 - results['mean_test_score']

    return results


def plot_complexity_curve(df, x_axis, title):
    """
    plot the complexity curve: error vs complexity parameter
    :param title:
    :param df: the dataframe containing the cv_results
    :param x_axis: the complexity parameter
    :param group: if none, no 
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x_axis], df['mean_train_error'], label='Training error', linestyle='--', marker='o', color='r')
    ax.plot(df[x_axis], df['mean_test_error'], label='Validation error', linestyle='-', marker='o', color='b')
    plt.xlabel(x_axis)
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    return plt