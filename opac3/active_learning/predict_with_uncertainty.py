# active_learning/predict_with_uncertainty.py

import torch
import numpy as np
from torch.utils.data import DataLoader

def predict_with_uncertainty(model, dataset, batch_size=32, num_samples=10):
    """
    Predicts outputs and estimates uncertainties using Monte Carlo Dropout.

    Args:
        model: Trained model with dropout layers.
        dataset: Dataset to predict.
        batch_size: Batch size for DataLoader.
        num_samples: Number of forward passes for uncertainty estimation.

    Returns:
        predictions: Mean predictions.
        uncertainties: Standard deviation of predictions.
    """
    model.train()  # Enable dropout during inference
    dataloader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    uncertainties = []

    for batch in dataloader:
        inputs = batch['descriptors']
        preds = []
        for _ in range(num_samples):
            outputs = model(inputs)
            preds.append(outputs.detach().numpy())
        preds = np.array(preds)
        mean_preds = preds.mean(axis=0)
        std_preds = preds.std(axis=0)
        predictions.extend(mean_preds)
        # Compute mean uncertainty across output dimensions
        uncertainties.extend(std_preds.mean(axis=1))

    return predictions, uncertainties
