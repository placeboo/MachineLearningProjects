import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import time
import yaml

import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
import pickle

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_feature_names_from_preprocessor(preprocessor):
    feature_names = []
    for name, trans, column in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(column)
        elif name in ['cat', 'pdays']:
            if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                encoder = trans.named_steps['onehot']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_features = encoder.get_feature_names_out(column)
                else:
                    cat_features = [f"{col}_{val}" for col, vals in zip(column, encoder.categories_)
                                    for val in vals]
                feature_names.extend(cat_features)
            else:
                feature_names.extend(column)
    return feature_names
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
    if not os.path.exists(model_file_path):
        print(f"Error: Model file not found at {model_file_path}")
        return None

    try:
        model = joblib.load(model_file_path)
        print(f'Model loaded successfully from {model_file_path}')
        return model
    except Exception as e:
        print(f"Error loading model from {model_file_path}: {str(e)}")
        return None


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
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight')
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


def get_roc_data(model, X, y):
    # Determine the type of model
    model_type = type(model).__name__
    # Get the scores for the positive class
    if model_type in ['SVC', 'NuSVC']:
        # For SVM, check if probability is True, otherwise use decision_function
        if getattr(model, 'probability', False):
            y_scores = model.predict_proba(X)[:, 1]
        else:
            y_scores = model.decision_function(X)
    elif model_type in ['MLPClassifier', 'KNeighborsClassifier']:
        # Neural Network and KNN use predict_proba
        y_scores = model.predict_proba(X)[:, 1]
    elif model_type in ['XGBClassifier', 'GradientBoostingClassifier']:
        # XGBoost can use either predict_proba or predict
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X)[:, 1]
        else:
            y_scores = model.predict(X)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_auc': roc_auc
    }
    return roc_data
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

def plot_learning_curve(lr_data, model_name, y_label='Error'):
    """
    Plot the learning curve: error vs training size or epochs
    :param lr_data: dictionary with keys: train_sizes/epochs, train_scores, val_scores
    :param model_name: string, either 'nn' for neural network or other for different models
    :param y_label: string, label for y-axis
    """
    set_plot_style()
    if model_name == 'nn':
        x_label = 'Epochs'
        # take the first 20 epochs
        train_sizes = lr_data['epochs'][:20]
        train_mean = 1-np.array(lr_data['train_scores'][:20],dtype=float)
        val_mean = 1-np.array(lr_data['val_scores'][:20],dtype=float)
    else:
        x_label = 'Training size'
        train_sizes = lr_data['train_sizes']
        train_mean = 1-np.mean(lr_data['train_scores'], axis=1)
        val_mean = 1-np.mean(lr_data['val_scores'], axis=1)
    
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, label='Training score', color='r', linestyle='--', marker='o', markersize=4)
    ax.plot(train_sizes, val_mean, label='Validation score', color='b', linestyle='-', marker='o', markersize=4)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Learning Curve')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    
    plt.tight_layout()
    return fig


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


def plot_complexity_curve(df, x_axis, y_train, y_val, title, ylabel='Score'):
    """
    Plot the complexity curve: error vs complexity parameter
    :param df: dataframe containing the cv_results
    :param x_axis: complexity parameter column name
    :param y_train: training score column name
    :param y_val: validation score column name
    :param title: plot title
    :param ylabel: y-axis label
    :return: figure and axis objects
    """
    set_plot_style()
    fig, ax = plt.subplots()

    if df[x_axis].dtype in ['object', 'str']:
        x = np.arange(len(df[x_axis]))
        width = 0.35
        ax.bar(x - width/2, df[y_train], width, label='Training score', color='r', alpha=0.5)
        ax.bar(x + width/2, df[y_val], width, label='Validation score', color='b', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df[x_axis], rotation=45, ha='right')
    else:
        ax.plot(df[x_axis], df[y_train], label='Training score', linestyle='--', marker='o', color='r', markersize=4)
        ax.plot(df[x_axis], df[y_val], label='Validation score', linestyle='-', marker='o', color='b', markersize=4)

    ax.set_xlabel(x_axis)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()
    return fig, ax

def plot_training_time(df, x_axis, title):
    set_plot_style()
    fig, ax = plt.subplots()  # Explicitly set figure size

    if df[x_axis].dtype in ['object', 'str']:
        x = np.arange(len(df[x_axis]))
        width = 0.35
        ax.bar(x, df["mean_fit_time"], width, label='Training time', color='r', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df[x_axis], rotation=45, ha='right')
    else:
        ax.plot(df[x_axis], df["mean_fit_time"], label='Training time', linestyle='--', marker='o', color='r', markersize=4)

    ax.set_xlabel(x_axis)
    ax.set_ylabel('Training time (s)')
    ax.set_title(title)
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black') 
    plt.tight_layout()
    return fig, ax
