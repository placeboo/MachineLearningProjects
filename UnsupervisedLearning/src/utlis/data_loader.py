import os
import numpy as np

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