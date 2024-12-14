# active_learning/active_learning.py

#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
import os
from copy import deepcopy

from data_loader import MoleculeDataset
from trainer import train_model, evaluate_model
from predict_with_uncertainty import predict_with_uncertainty
from uncertainty_sampling import select_most_uncertain_samples
from logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Active Learning for Molecular Property Prediction.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--targets-file', type=str, required=True, help='CSV file containing target properties.')
    parser.add_argument('--initial-train-size', type=int, default=10, help='Initial number of samples for training.')
    parser.add_argument('--query-size', type=int, default=5, help='Number of samples to query each AL iteration.')
    parser.add_argument('--iterations', type=int, default=10, help='Number of Active Learning iterations.')
    parser.add_argument('--model-output', type=str, default='models/al_trained_model.pth', help='Final model output path.')
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

    # Initialize labeled/unlabeled splits
    initial_train_df = df.sample(n=args.initial_train_size, random_state=42)
    unlabeled_df = df.drop(initial_train_df.index).reset_index(drop=True)

    # Determine how many iterations are possible
    max_iterations = max((len(df) - args.initial_train_size) // args.query_size, 0)
    iterations = min(args.iterations, max_iterations)
    if iterations == 0:
        logger.info("Not enough data for the specified iterations and query size.")
        return

    model = None  # We'll reuse this reference each iteration

    for iteration in range(iterations):
        logger.info(f"Active Learning Iteration {iteration + 1}/{iterations}")

        # Prepare training dataset
        train_descriptors = initial_train_df[descriptor_columns].to_dict('records')
        train_targets = initial_train_df[target_columns].to_dict('records')
        train_dataset = MoleculeDataset(train_descriptors, train_targets)

        # Prepare test dataset (everything not in initial_train_df)
        test_df = df.drop(initial_train_df.index).reset_index(drop=True)
        test_descriptors = test_df[descriptor_columns].to_dict('records')
        test_targets = test_df[target_columns].to_dict('records')
        test_dataset = MoleculeDataset(test_descriptors, test_targets)

        # Train or continue training the model
        input_dim = len(descriptor_columns)
        output_dim = len(target_columns)

        # Pass the existing model to continue training from previous iteration
        model = train_model(
            dataset=train_dataset,
            existing_model=model,
            input_dim=input_dim,
            output_dim=output_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            weight_decay=args.weight_decay
        )

        # Evaluate on test set
        test_loss, per_target_metrics = evaluate_model(model, test_dataset, batch_size=args.batch_size)
        logger.info(f"Test Loss (PyTorch aggregated MSE): {test_loss:.4f}")
        for m in per_target_metrics:
            i = m['target_index']
            logger.info(f"[Target {i}] MSE={m['MSE']:.4f}, MAE={m['MAE']:.4f}, R2={m['R2_Score']:.4f}")

        # If unlabeled_df is empty, no more samples to query
        if unlabeled_df.empty:
            logger.info("No more unlabeled samples. Stopping Active Learning.")
            break

        # Predict on unlabeled data with uncertainty
        unlabeled_descriptors = unlabeled_df[descriptor_columns].to_dict('records')
        unlabeled_dataset = MoleculeDataset(unlabeled_descriptors, targets=None)  # no targets
        predictions, uncertainties = predict_with_uncertainty(model, unlabeled_dataset)

        # Select samples with highest uncertainty
        current_query_size = min(args.query_size, len(unlabeled_df))
        query_indices = select_most_uncertain_samples(uncertainties, current_query_size)

        # Merge queried samples into training data
        queried_samples = unlabeled_df.iloc[query_indices]
        initial_train_df = pd.concat([initial_train_df, queried_samples], ignore_index=True)

        # Remove them from unlabeled
        unlabeled_df = unlabeled_df.drop(queried_samples.index).reset_index(drop=True)
        logger.info(f"Queried {len(queried_samples)} samples in iteration {iteration + 1}.")

    # Save final model
    output_dir = os.path.dirname(args.model_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory {output_dir}.")

    torch.save(model, args.model_output)  # Save entire model object
    logger.info(f"Active Learning completed. Final model saved to {args.model_output}")

    # Save final AL training set
    final_train_df_path = "data/final_al_training_data.csv"
    initial_train_df.to_csv(final_train_df_path, index=False)
    logger.info(f"Saved final AL training data to {final_train_df_path}")
if __name__ == '__main__':
    main()
