# active_learning/active_learning.py

import argparse
import pandas as pd
import torch
import os
from copy import deepcopy
from data_loader import MoleculeDataset
from trainer import train_model
from predict_with_uncertainty import predict_with_uncertainty
from uncertainty_sampling import select_most_uncertain_samples
from logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Active Learning for Molecular Property Prediction.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--targets-file', type=str, required=True, help='CSV file containing target properties.')
    parser.add_argument('--initial-train-size', type=int, default=10, help='Initial number of samples for training.')
    parser.add_argument('--query-size', type=int, default=5, help='Number of samples to query in each AL iteration.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of Active Learning iterations.')
    parser.add_argument('--model-output', type=str, default='models/al_trained_model.pth', help='File to save the trained model.')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer size.')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay.')
    args = parser.parse_args()

    # Load data
    df_descriptors = pd.read_csv(args.descriptors_file)
    df_targets = pd.read_csv(args.targets_file)

    # Merge on 'mol_id'
    df = pd.merge(df_descriptors, df_targets, on='mol_id')
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    target_columns = [col for col in df_targets.columns if col != 'mol_id']

    # Initialize labeled and unlabeled datasets
    initial_train_df = df.sample(n=args.initial_train_size, random_state=42)
    unlabeled_df = df.drop(initial_train_df.index).reset_index(drop=True)

    # Adjust iterations if necessary
    max_iterations = max((len(df) - args.initial_train_size) // args.query_size, 0)
    iterations = min(args.iterations, max_iterations)
    if iterations == 0:
        logger.info("Not enough data for the specified number of iterations and query size.")
        return

    for iteration in range(iterations):
        logger.info(f"Active Learning Iteration {iteration + 1}/{iterations}")

        # Prepare training dataset
        train_descriptors = initial_train_df[descriptor_columns].to_dict('records')
        train_targets = initial_train_df[target_columns].to_dict('records')
        train_dataset = MoleculeDataset(train_descriptors, train_targets)

        # Train model
        input_dim = len(descriptor_columns)
        output_dim = len(target_columns)
        model = train_model(
            train_dataset,
            input_dim,
            output_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            weight_decay=args.weight_decay
        )

        # Check if unlabeled_df is empty
        if unlabeled_df.empty:
            logger.info("No more unlabeled samples available. Stopping Active Learning.")
            break  # Exit the loop since there's nothing more to query

        # Predict on unlabeled data with uncertainty estimation
        unlabeled_descriptors = unlabeled_df[descriptor_columns].to_dict('records')
        unlabeled_dataset = MoleculeDataset(unlabeled_descriptors)
        predictions, uncertainties = predict_with_uncertainty(model, unlabeled_dataset)

        # Select samples with highest uncertainty
        current_query_size = min(args.query_size, len(unlabeled_df))
        query_indices = select_most_uncertain_samples(uncertainties, current_query_size)

        # Get queried samples
        queried_samples = unlabeled_df.iloc[query_indices]

        # Add queried samples to training data
        initial_train_df = pd.concat([initial_train_df, queried_samples], ignore_index=True)

        # Remove queried samples from unlabeled data
        unlabeled_df = unlabeled_df.drop(queried_samples.index).reset_index(drop=True)

        logger.info(f"Queried {len(queried_samples)} samples.")

    # Save the final model
    output_dir = os.path.dirname(args.model_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}.")

    # Save the entire model object
    torch.save(model, args.model_output)
    logger.info(f"Active Learning completed. Model saved to {args.model_output}")

if __name__ == '__main__':
    main()
