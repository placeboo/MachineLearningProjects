import numpy as np

def find_elbow_indice(arr: np.ndarray) -> int:
    """
    Find the elbow point in the data.
    :param arr: the array of metric to find the elbow point
    :return: the index of the elbow point
    """
    # Calculate the second derivative
    second_derivative = np.diff(np.diff(arr))
    # Find the index of the maximum value of the second derivative
    elbow_idx = np.argmin(np.abs(second_derivative)) + 2
    return elbow_idx