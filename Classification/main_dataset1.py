"""
This function is to train the model and save the best model, cross-validation results, evaluation metrics
"""
import pandas as pd
import yaml
import sys
import argparse
import numpy as np
import joblib
from scipy import sparse
import os
from sklearn.model_selection import learning_curve
from src.common.utils import save_model,save_cv_results,load_model, format_cv_results, get_roc_data, save_metrics
from src.dataset1_tabular.models import Classifier
from src.dataset1_tabular.preprocessing import load_csv, preprocess_data, save_processed_data


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path, model_name):
    config = load_config(config_path)

    # check if model name is valid
    if model_name not in config['models']:
        print(f'Invalid model name. Choose from {config["models"].keys()}')
        sys.exit(1)

    # copy the config file to the output directory
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.system(f'cp {config_path} {output_dir}')
    # load input data
    output_dir = config['dataset']['output_dir']
    if not config['dataset']['rerun']:
        # preprocess has been done. Load the processed data
        X_train = sparse.load_npz(os.path.join(output_dir, 'X_train.npz'))
        X_test = sparse.load_npz(os.path.join(output_dir, 'X_test.npz'))
        y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
        preprocessor = joblib.load(os.path.join(output_dir, 'preprocessor.joblib'))
        print('Data loaded successfully!')
    else: # preprocess data
        data = load_csv(config['dataset']['file_path'])
        X_train, X_test, y_train, y_test, preprocessor = \
            preprocess_data(data, \
                            config['dataset']['target_column'],\
                            config['dataset']['num_features'],\
                            config['dataset']['cat_features'])
        print('Data preprocessed successfully!')
    # save
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor, output_dir)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # train the model
    cv = config['cross_validation']['n_splits']
    clf = Classifier(model_name, X_train, y_train, \
                       X_test, y_test, \
                       config['models'][model_name]['param_grid'], \
                       cv)
    clf.train()
    # save the best model
    result_dir = config['output']['results_dir']
    save_model(clf.best_model, result_dir, model_name, config['dataset']['name'])

    metrics = clf.evaluation()
    # save cross-validation results
    save_cv_results(clf.cv_results, result_dir, model_name, config['dataset']['name'])
    # save evaluation metrics
    save_metrics(metrics, result_dir, model_name, config['dataset']['name'])
    # save learning curve data
    if model_name == 'nn':
        learning_curve_data = clf.nn_learning_curve_data()
    else:
        learning_curve_data = clf.create_learning_curve_data()
    save_metrics(learning_curve_data, result_dir, \
                 model_name, f'{config["dataset"]["name"]}_lc')
    # save roc curve data
    roc_data = get_roc_data(clf.best_model, X_train, y_train)
    save_metrics(roc_data, result_dir, model_name, \
                 f'{config["dataset"]["name"]}_roc')

    # save feature importance
    if hasattr(clf.best_model, 'feature_importances_'):
        feature_importances = clf.best_model.feature_importances_
        feature_names = clf.best_model.feature_names_in_  # Assuming scikit-learn 0.24+
        importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
        importances_df = importances_df.sort_values('importance', ascending=False)
        importances_df.to_csv(f"{output_dir}/{model_name}_feature_importances.csv", index=False)

    # save hyperparmaeter search results
    cv_results = format_cv_results(clf.cv_results)
    cv_results.to_csv(f"{output_dir}/{model_name}_cv_results.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a classifier')
    parser.add_argument('--config', '-c', help='path to config file', required=True)
    parser.add_argument('--model', '-m', help='model name', required=True)
    args = parser.parse_args()

    main(args.config, args.model)



