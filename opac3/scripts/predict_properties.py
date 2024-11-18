# predict_properties.py

import argparse
import pandas as pd
import torch
import json
from opac3.models.trainer import PropertyPredictor
from opac3.data.dataset import MoleculeDataset
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Predict properties of new molecules using a trained model.')
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--predictions-output', type=str, required=True, help='Output file for predictions (CSV).')
    args = parser.parse_args()

    # Load descriptors
    df_descriptors = pd.read_csv(args.descriptors_file)
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    descriptors = df_descriptors[descriptor_columns].to_dict('records')

    # Create dataset
    dataset = MoleculeDataset(descriptors, targets=None)

    # Load model parameters
    params_file = args.model_file + '.params.json'
    with open(params_file, 'r') as f:
        model_params = json.load(f)
    input_dim = model_params['input_dim']
    hidden_dim = model_params['hidden_dim']
    output_dim = model_params['output_dim']

    # Initialize the model
    model = PropertyPredictor(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(args.model_file, weights_only=True))
    model.eval()

    # Make predictions
    predictions = []
    with torch.no_grad():
        for sample in dataset:
            inputs = sample['descriptors'].unsqueeze(0)  # Add batch dimension
            outputs = model(inputs)
            predictions.append(outputs.numpy().flatten())

    # Generate property names dynamically
    property_names = [f'Predicted_Property_{i+1}' for i in range(output_dim)]

    # Convert predictions to DataFrame
    df_predictions = pd.DataFrame(predictions, columns=property_names)
    df_predictions['mol_id'] = df_descriptors['mol_id']

    # Save predictions
    df_predictions.to_csv(args.predictions_output, index=False)
    logger.info(f"Saved predictions to {args.predictions_output}.")

if __name__ == '__main__':
    main()
