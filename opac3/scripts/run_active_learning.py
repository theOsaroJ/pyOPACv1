# opac3/scripts/run_active_learning.py

import argparse
import pandas as pd
from opac3.data.dataset import MoleculeDataset
from opac3.models.trainer import train_model
from opac3.active_learning.al_loop import active_learning_loop
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run active learning loop.')
    parser.add_argument('--initial-descriptors-file', type=str, required=True, help='CSV file containing initial descriptors.')
    parser.add_argument('--initial-targets-file', type=str, required=True, help='CSV file containing initial targets.')
    parser.add_argument('--unlabeled-descriptors-file', type=str, required=True, help='CSV file containing unlabeled descriptors.')
    parser.add_argument('--unlabeled-targets-file', type=str, required=True, help='CSV file containing unlabeled targets (for simulation).')
    parser.add_argument('--budget', type=int, default=10, help='Number of samples to label in each iteration.')
    args = parser.parse_args()

    # Load initial dataset
    df_initial_descriptors = pd.read_csv(args.initial_descriptors_file)
    df_initial_targets = pd.read_csv(args.initial_targets_file)
    df_initial = pd.merge(df_initial_descriptors, df_initial_targets, on='mol_id')
    initial_descriptors = df_initial[df_initial_descriptors.columns].to_dict('records')
    initial_targets = df_initial[df_initial_targets.columns].to_dict('records')
    initial_dataset = MoleculeDataset(initial_descriptors, initial_targets)

    # Load unlabeled data
    df_unlabeled_descriptors = pd.read_csv(args.unlabeled_descriptors_file)
    df_unlabeled_targets = pd.read_csv(args.unlabeled_targets_file)
    df_unlabeled = pd.merge(df_unlabeled_descriptors, df_unlabeled_targets, on='mol_id')
    unlabeled_data = []
    for idx, row in df_unlabeled.iterrows():
        descriptor = row[df_unlabeled_descriptors.columns].to_dict()
        target = row[df_unlabeled_targets.columns].to_dict()
        unlabeled_data.append({'descriptor': descriptor, 'target': target})

    # Initialize model (can be None if training from scratch each time)
    model = None

    # Run active learning loop
    active_learning_loop(initial_dataset, unlabeled_data, model, args.budget)

if __name__ == '__main__':
    main()
