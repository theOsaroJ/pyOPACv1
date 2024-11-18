# active_learning/predict_new_data.py

import argparse
import pandas as pd
import torch
from data_loader import MoleculeDataset
from logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Predict properties using the trained AL model.')
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--predictions-output', type=str, required=True, help='Output file for predictions (CSV).')
    args = parser.parse_args()

    # Load descriptors
    df_descriptors = pd.read_csv(args.descriptors_file)
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    descriptors = df_descriptors[descriptor_columns].to_dict('records')

    # Create dataset
    dataset = MoleculeDataset(descriptors)

    # Load the entire model
    model = torch.load(args.model_file)
    model.eval()

    # Make predictions
    predictions = []
    with torch.no_grad():
        for sample in dataset:
            inputs = sample['descriptors'].unsqueeze(0)  # Add batch dimension
            outputs = model(inputs)
            predictions.append(outputs.numpy().flatten())

    # Get output property names
    output_dim = predictions[0].shape[0]
    property_names = [f'Predicted_Property_{i+1}' for i in range(output_dim)]

    # Convert predictions to DataFrame
    df_predictions = pd.DataFrame(predictions, columns=property_names)
    df_predictions['mol_id'] = df_descriptors['mol_id']

    # Save predictions
    df_predictions.to_csv(args.predictions_output, index=False)
    logger.info(f"Saved predictions to {args.predictions_output}.")

if __name__ == '__main__':
    main()
