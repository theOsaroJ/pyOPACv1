# active_learning/uncertainty_sampling.py

import numpy as np

def select_most_uncertain_samples(uncertainties, num_samples):
    """
    Select indices of samples with highest uncertainty.

    Args:
        uncertainties (list or np.array): Uncertainty values.
        num_samples (int): Number of samples to select.

    Returns:
        list: Indices of selected samples.
    """
    uncertainties = np.array(uncertainties)
    # Get indices sorted by uncertainty in descending order
    sorted_indices = uncertainties.argsort()[::-1]
    return sorted_indices[:num_samples].tolist()
