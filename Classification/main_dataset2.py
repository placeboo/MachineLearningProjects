import yaml
import joblib
from scipy import sparse
import numpy as np
import os
import sys
from src.dataset2_nlp.preprocessing import load_csv, preprocess_data, save_process_data, load_embedder
from src.common.utils import load_config
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
        X_train = sparse.load_npz(os.path.join(data_output_dir, 'X_train.npz'))
        X_test = sparse.load_npz(os.path.join(data_output_dir, 'X_test.npz'))
        y_train = np.load(os.path.join(data_output_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(data_output_dir, 'y_test.npy'))
        embedder = load_embedder(os.path.join(data_output_dir, f'{embedding_method}_embedder.joblib'), embedding_method)
        print('Data loaded successfully!')
    else:
        data = load_csv(config['dataset']['file_path'])
        # load embedding kwargs from config if available
        embedding_kwargs = config['dataset'].get('embedding_kwargs', {})
        X_train, X_test, y_train, y_test, embedder = preprocess_data(data, embedding_method, **embedding_kwargs)
        print('Data preprocessed successfully!')
        # save data
        save_process_data(X_train, X_test, y_train, y_test, embedder, data_output_dir)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate comment spam classifier')
    parser.add_argument('--config', '-c', help='path to config file', required=True)
    parser.add_argument('--model', '-m', help='model name', default=None)
    args = parser.parse_args()

    main(args.config, args.model)
    #main('dataset2_configs/config.yml')


