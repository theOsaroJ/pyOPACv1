# opac3/active_learning/al_loop.py

import random
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

def active_learning_loop(initial_dataset, unlabeled_data, model, budget, strategy='random'):
    """
    Runs an active learning loop.
    Args:
        initial_dataset: The initial labeled dataset.
        unlabeled_data: The pool of unlabeled data.
        model: The model to be trained.
        budget: Number of samples to label in each iteration.
        strategy: Active learning strategy ('random', 'uncertainty', etc.).
    """
    current_dataset = initial_dataset
    while len(unlabeled_data) > 0:
        # Train the model on current dataset
        input_dim = current_dataset.input_dim
        output_dim = current_dataset.output_dim
        model = train_model(current_dataset, input_dim, output_dim)

        # Select samples to label
        if strategy == 'random':
            selected_indices = random.sample(range(len(unlabeled_data)), min(budget, len(unlabeled_data)))
        else:
            # Implement other strategies (e.g., uncertainty sampling)
            selected_indices = random.sample(range(len(unlabeled_data)), min(budget, len(unlabeled_data)))

        # Simulate labeling (in practice, you would obtain labels from an oracle or experiments)
        new_samples = [unlabeled_data[idx] for idx in selected_indices]
        for idx in sorted(selected_indices, reverse=True):
            del unlabeled_data[idx]

        # Add new samples to current dataset
        current_dataset.descriptors.extend([sample['descriptor'] for sample in new_samples])
        current_dataset.targets.extend([sample['target'] for sample in new_samples])

        logger.info(f"Added {len(new_samples)} new samples to the dataset.")
