# opac3/models/predictor.py

import torch
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def predict_properties(model, descriptors: list, descriptor_names: list):
    """
    Predicts properties for a list of descriptors using the trained model.
    """
    model.eval()
    with torch.no_grad():
        inputs = []
        for descriptor in descriptors:
            descriptor_values = [descriptor[name] for name in descriptor_names]
            inputs.append(descriptor_values)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        predictions = model(inputs_tensor)
    return predictions.numpy()
