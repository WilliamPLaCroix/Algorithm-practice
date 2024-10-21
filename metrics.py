import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for a list of true values and predicted values.
    Shape: (n_classes, n_classes)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    mapping = {label: i for i, label in enumerate(set(y_pred).union(set(y_true)))}
    confusion_matrix = np.zeros((len(mapping), len(mapping)))
    for i in range(len(y_true)):
        confusion_matrix[mapping[y_true[i]], mapping[y_pred[i]]] += 1
    return confusion_matrix