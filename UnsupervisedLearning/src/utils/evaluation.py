import numpy as np
from typing import Dict, Tuple

import pandas as pd


def find_elbow_indice(arr: np.ndarray) -> int:
    """
    Find the elbow point in the data.
    :param arr: the array of metric to find the elbow point
    :return: the index of the elbow point
    """
    # Calculate the second derivative
    second_derivative = np.diff(np.diff(arr))
    # Find the index of the maximum value of the second derivative
    elbow_idx = np.argmax(np.abs(second_derivative)) + 2
    return elbow_idx


def organize_experiment4_results(data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Organize the result of experiment 4.
    :return:
    the top 1 highest test score
    """
    results = {
        'n_components': [],
        'mean_fit_time': [],
        'mean_score_time': [],
        'mean_train_score': [],
        'mean_test_score': [],
        'params':[]
    }

    eval_result = []

    for n_comp, values in data.items():
        cv_results = values['cv_results']
        cv_results = pd.DataFrame(cv_results)
        # find the top 1 highest test score
        cv_result = cv_results.loc[cv_results['rank_test_score'] == 1]
        results['n_components'].append(n_comp)
        results['params'].append(cv_result['params'].values[0])
        results['mean_fit_time'].append(cv_result['mean_fit_time'].values[0])
        results['mean_score_time'].append(cv_result['mean_score_time'].values[0])
        results['mean_test_score'].append(cv_result['mean_test_score'].values[0])
        results['mean_train_score'].append(cv_result['mean_train_score'].values[0])

        tmp_eval_result = {
            'n_components': n_comp,
            **values['metrics']
        }
        eval_result.append(tmp_eval_result)
    result_df = pd.DataFrame(results)
    eval_result_df = pd.DataFrame(eval_result)

    # find the optimal n_components with the highest test score
    best_idx = result_df['mean_test_score'].idxmax()
    best_n_components = result_df.loc[best_idx, 'n_components']
    return result_df, eval_result_df, best_n_components

def organize_experiment5_results(data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = {}

    cv_results = data['cv_results']
    cv_results = pd.DataFrame(cv_results)
    cv_result = cv_results.loc[cv_results['rank_test_score'] == 1]
    results['params'] = cv_result['params'].values
    results['mean_fit_time'] = cv_result['mean_fit_time'].values[0]
    results['mean_score_time'] = cv_result['mean_score_time'].values[0]
    results['mean_test_score'] = cv_result['mean_test_score'].values[0]
    results['mean_train_score'] = cv_result['mean_train_score'].values[0]

    eval_result = pd.DataFrame([data['metrics']])

    return pd.DataFrame(results), eval_result