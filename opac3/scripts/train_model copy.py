# opac3/scripts/train_model.py

import os
import argparse
import pandas as pd
import torch
from opac3.data.dataset import MoleculeDataset
from opac3.models.trainer import train_model
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train a property prediction model.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--targets-file', type=str, required=True, help='CSV file containing target properties.')
    parser.add_argument('--model-output', type=str, required=True, help='File to save the trained model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    args = parser.parse_args()

    # Load descriptors and targets
    df_descriptors = pd.read_csv(args.descriptors_file)
    df_targets = pd.read_csv(args.targets_file)

    # Debug: Print columns
    print("df_descriptors columns:", df_descriptors.columns.tolist())
    print("df_targets columns:", df_targets.columns.tolist())

    # Ensure 'mol_id' is of the same type
    df_descriptors['mol_id'] = df_descriptors['mol_id'].astype(int)
    df_targets['mol_id'] = df_targets['mol_id'].astype(int)

    # Merge on mol_id
    df = pd.merge(df_descriptors, df_targets, on='mol_id')

    # Debug: Print merged df info
    print("Merged df shape:", df.shape)
    print("Merged df columns:", df.columns.tolist())

    # Prepare descriptors and targets
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    target_columns = [col for col in df_targets.columns if col != 'mol_id']

    # Debug: Print column lists
    print("Descriptor columns:", descriptor_columns)
    print("Target columns:", target_columns)

    # Check if all descriptor columns are in df
    missing_descriptor_cols = set(descriptor_columns) - set(df.columns)
    if missing_descriptor_cols:
        print(f"Missing descriptor columns in df: {missing_descriptor_cols}")
        # Handle the missing columns as appropriate (e.g., raise an error or remove missing columns)

    # Ensure all required columns are present
    required_columns = ['mol_id'] + descriptor_columns + target_columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Missing columns in merged DataFrame: {missing_columns}")
        raise KeyError(f"Columns {missing_columns} not found in the merged DataFrame.")

    # Proceed to create descriptors and targets
    descriptors = df[['mol_id'] + descriptor_columns].to_dict('records')
    targets = df[['mol_id'] + target_columns].to_dict('records')

    # Create dataset
    dataset = MoleculeDataset(descriptors, targets)

    # Train model
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    model = train_model(dataset, input_dim, output_dim, epochs=args.epochs)

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.model_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}.")

    # Save model
    torch.save(model.state_dict(), args.model_output)
    logger.info(f"Saved trained model to {args.model_output}.")

if __name__ == '__main__':
    main()
