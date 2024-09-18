import yaml
import joblib
from scipy import sparse
import numpy as np
import os
import sys

from yellowbrick.model_selection import learning_curve

from src.dataset2_nlp.preprocessing import load_csv, preprocess_data, save_process_data, load_process_data
from src.common.utils import load_config, save_model, save_cv_results, format_cv_results, get_roc_data, save_metrics
from src.dataset2_nlp.models import Classifier
import argparse

def main(config_path, model_name=None):
    config = load_config(config_path)
    embedding_method = config['dataset']['embedding_method']
    # preprocessing and embedding
    data_output_dir = config['dataset']['output_dir']
    os.makedirs(data_output_dir, exist_ok=True)
    os.system(f'cp {config_path} {data_output_dir}')

    if not config['dataset']['rerun']:
        # load the processed data
        X_train, X_test, y_train, y_test = load_process_data(data_output_dir)
        print('Data loaded successfully!')
    else:
        data = load_csv(config['dataset']['file_path'])
        # load embedding kwargs from config if available
        embedding_config = config['dataset'].get('embedding_config', None)
        X_train, X_test, y_train, y_test, embedder = preprocess_data(data, embedding_method, embedding_config)
        print('Data preprocessed successfully!')
        # save data
        save_process_data(X_train, X_test, y_train, y_test, embedder, data_output_dir)
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
    model_output_dir = config['model_output_dir']
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)
    save_model(clf.best_model, model_output_dir, model_name, config['dataset']['name'])

    # save evaluation metrics
    metrics = clf.evaluation()
    save_metrics(metrics, model_output_dir, model_name, config['dataset']['name'])
    # save learning curve data
    if model_name == 'nn':
        learning_curve_data = clf.nn_learning_curve_data()
    else:
        learning_curve_data = clf.create_learning_curve_data()
    save_metrics(learning_curve_data, model_output_dir, \
                 model_name, f'{config["dataset"]["name"]}_lc')
    # save roc data
    roc_data = get_roc_data(clf.best_model, X_train, y_train)
    save_metrics(roc_data, model_output_dir, model_name, \
                 f'{config["dataset"]["name"]}_roc')

    # save hyperparameter tuning results
    cv_results = format_cv_results(clf.cv_results)
    cv_results.to_csv(f"{model_output_dir}/{model_name}_cv_results.csv", index=False)
    print(f"Hyperparameter search results saved to {model_output_dir}/{model_name}_cv_results.csv")
    # save config file
    with open(f"{model_output_dir}/config.yml", 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate comment spam classifier')
    parser.add_argument('--config', '-c', help='path to config file', required=True)
    parser.add_argument('--model', '-m', help='model name', default=None)
    args = parser.parse_args()
    if args.model is None:
        for model_name in ['knn', 'nn', 'svm', 'boosting']:
            print(f"Training {model_name}")
            main(args.config, model_name)
            print(f"Training of {model_name} complete!!!!!!")
    else:
        main(args.config, args.model)
    #main('dataset2_configs/config.yml')


