# opac3/scripts/train_model.py
#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
import os
import json
from sklearn.model_selection import train_test_split

from opac3.data.dataset import MoleculeDataset
from opac3.models.trainer import train_model, evaluate_model
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train a property prediction model.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--targets-file', type=str, required=True, help='CSV file containing target properties.')
    parser.add_argument('--model-output', type=str, required=True, help='File to save the trained model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion of dataset in test split.')

    # Hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimizer.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Number of neurons in the hidden layer.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 regularization).')

    args = parser.parse_args()

    # Load descriptors and targets
    df_descriptors = pd.read_csv(args.descriptors_file)
    df_targets = pd.read_csv(args.targets_file)

    # Ensure 'mol_id' is the same type in both
    df_descriptors['mol_id'] = df_descriptors['mol_id'].astype(int)
    df_targets['mol_id'] = df_targets['mol_id'].astype(int)

    # Merge on mol_id
    df = pd.merge(df_descriptors, df_targets, on='mol_id')

    # Identify descriptor and target columns
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    target_columns = [col for col in df_targets.columns if col != 'mol_id']

    # Split data into train/test sets
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

    # Create training dataset
    train_descriptors = train_df[descriptor_columns].to_dict('records')
    train_targets = train_df[target_columns].to_dict('records')
    train_dataset = MoleculeDataset(train_descriptors, train_targets)

    # Create test dataset
    test_descriptors = test_df[descriptor_columns].to_dict('records')
    test_targets = test_df[target_columns].to_dict('records')
    test_dataset = MoleculeDataset(test_descriptors, test_targets)

    # Train model
    input_dim = train_dataset.input_dim   # from MoleculeDataset
    output_dim = train_dataset.output_dim
    model = train_model(
        dataset=train_dataset,
        input_dim=input_dim,
        output_dim=output_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        weight_decay=args.weight_decay
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(args.model_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}.")

    # Save the trained model state_dict
    torch.save(model.state_dict(), args.model_output)
    logger.info(f"Saved trained model to {args.model_output}.")

    # Save model parameters
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': output_dim
    }
    params_output = args.model_output + '.params.json'
    with open(params_output, 'w') as f:
        json.dump(model_params, f)
    logger.info(f"Saved model parameters to {params_output}.")

    # Evaluate model on the test set, returning per-target metrics
    test_loss, per_target_metrics = evaluate_model(model, test_dataset, batch_size=args.batch_size)
    logger.info(f"Test Loss (PyTorch MSE): {test_loss:.4f}")

    # per_target_metrics is a list of dictionaries, one per target dimension
    for metric_info in per_target_metrics:
        i = metric_info['target_index']
        mse_i = metric_info['MSE']
        mae_i = metric_info['MAE']
        r2_i = metric_info['R2_Score']
        logger.info(f"[Target {i}] MSE={mse_i:.4f}, MAE={mae_i:.4f}, R2={r2_i:.4f}")

if __name__ == '__main__':
    main()
