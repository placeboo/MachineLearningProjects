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
from src.common.utils import load_config, save_model,save_cv_results,get_feature_names_from_preprocessor, format_cv_results, get_roc_data, save_metrics
from src.dataset1_tabular.models import Classifier
from src.dataset1_tabular.preprocessing import load_csv, preprocess_data, save_processed_data, load_processed_data
from imblearn.over_sampling import SMOTE


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
        X_train, X_test, y_train, y_test, preprocessor = load_processed_data(output_dir)
        print('Data loaded successfully!')
        print(f"X_train shape: {X_train.shape}")
        print(f'y_train shape: {y_train.shape}')
    else: # preprocess data
        data = load_csv(config['dataset']['file_path'])
        X_train, X_test, y_train, y_test, preprocessor = \
            preprocess_data(data, \
                            config['dataset']['target_column'],\
                            config['dataset']['num_features'],\
                            config['dataset']['cat_features'],
                            config['dataset'].get('is_smote', False))
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
    if model_name == 'boosting':
        feature_names = get_feature_names_from_preprocessor(preprocessor)
        feature_importance = clf.best_model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {'feature': feature_names, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        feature_importance_df.to_csv(f"{result_dir}/{model_name}_feature_importance.csv", index=False)
        print(f"Feature importance saved to {result_dir}/{model_name}_feature_importance.csv")
    # save hyperparmaeter search results
    cv_results = format_cv_results(clf.cv_results)
    cv_results.to_csv(f"{result_dir}/{model_name}_cv_results.csv", index=False)
    print(f"Hyperparameter search results saved to {result_dir}/{model_name}_cv_results.csv")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a classifier')
    parser.add_argument('--config', '-c', help='path to config file', required=True)
    parser.add_argument('--model', '-m', help='model name', required=True)
    args = parser.parse_args()

    main(args.config, args.model)



