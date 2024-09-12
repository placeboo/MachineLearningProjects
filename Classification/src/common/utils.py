import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import json
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

############################################
#  Plots and Visualizations
############################################
def plot_leanring_curve(model, X, y, cv, train_size=np.linspace(0.1, 1.0, 5)):
    """
    plot the learning curve: error vs training size
    :param model: the best model from GridSearchCV
    :param X: the X_train
    :param y: the y_train
    :param cv: the k-fold
    :param train_size: relative training size
    """
    train_sizes, train_scores, val_scores = learning_curve(model, X, y, train_sizes=train_size, cv=cv, n_jobs=-1, random_state=17, verbose=2)

    train_error_mean = 1 - np.mean(train_scores, axis=1)
    val_error_mean = 1 - np.mean(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_error_mean, label='Training error', color='r', linestyle='--', marker='o')
    ax.plot(train_sizes, val_error_mean, label='Validation error', color='b', linestyle='-', marker='o')
    ax.set_xlabel('Training size')
    ax.set_ylabel('Error')
    ax.set_title('Learning Curve')
    ax.legend()
    return fig

