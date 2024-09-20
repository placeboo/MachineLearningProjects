"""
This file contains:
1. handling outlies using the IQR method
2. Preprocessing pipeline includes:
    - encoding categorical features
    - scaling numerical features
3. train-testing splitting
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from scipy import sparse
import numpy as np
import joblib
import os


NUM_FEATURES = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                'euribor3m', 'nr.employed']
CAT_FEATURES = ['job', 'marital', 'education', 'day_of_week', 'default', 'housing', 'loan', 'contact', 'month',
                'poutcome', 'pdays']
TARGET_COLUMN = 'subscribed'

def load_csv(file_path):
    return pd.read_csv(file_path, sep=';')


def convert_target(df, column):
    df[column] = df[column].map({
        'yes': 1,
        'no': -1
    })
    return df


def handle_outliers(df, columns, threshold=2):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - threshold * iqr
        ub = q3 + threshold * iqr
        df[col] = df[col].clip(lb, ub)
        return df


def covert_pdays_to_categorical(df):
    def categorize_pdays(pdays):
        if pdays == 999:
            return 'NoContact'
        elif 0 <= pdays <= 2:
            return 'Pdays0_2'
        elif 3 <= pdays <= 7:
            return 'Pdays3_7'
        elif 8 <= pdays <= 14:
            return 'Pdays8_14'
        else:
            return 'Pdays>14'

    df_ = df.copy()
    df_['pdays'] = df_['pdays'].apply(categorize_pdays)
    return df_


def create_preprocessing_pipeline(num_features, cat_features):
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    pdays_transformer = Pipeline(steps=[
        ('categorize', FunctionTransformer(covert_pdays_to_categorical, validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('pdays', pdays_transformer, ['pdays']),
            ('cat', cat_transformer, [col for col in cat_features if col != 'pdays'])
        ]
    )

    return preprocessor


def preprocess_data(df, target_column, num_features, cat_features, random_state=17, is_smote=False):

    df = convert_target(df, target_column)

    # handle outliers
    df = handle_outliers(df, num_features)

    preprocessor = create_preprocessing_pipeline(num_features, cat_features)

    # data split
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if is_smote:
        smote = SMOTE(random_state=random_state)
        X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def save_processed_data(X_train, X_test, y_train, y_test, preprocessor, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sparse.issparse(X_train):
        sparse.save_npz(os.path.join(output_dir, 'X_train.npz'), X_train)
        sparse.save_npz(os.path.join(output_dir, 'X_test.npz'), X_test)
    else:
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)

    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))

    print('Data saved successfully!')

def load_processed_data(output_dir):
    try:
        X_train = sparse.load_npz(os.path.join(output_dir, 'X_train.npz'))
        X_test = sparse.load_npz(os.path.join(output_dir, 'X_test.npz'))
    except:
        X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test



