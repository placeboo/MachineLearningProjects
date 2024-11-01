import os
import numpy as np
import pickle
import json
import pandas as pd


def load_processed_data(output_dir):
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f'{output_dir} does not exist.')
    X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test


def save_csv(df, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    file_path = file_path + '.csv'
    df.to_csv(file_path, index=False)
    print(f"Dataframe saved at {file_path}")

def load_csv(file_path):
    return pd.read_csv(file_path)


def save_pickle(data, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    file_path = file_path + '.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved at {file_path}")

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_json(data, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    file_path = file_path + '.json'
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved at {file_path}")

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data