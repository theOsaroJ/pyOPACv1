# opac3/data/dataset.py

from torch.utils.data import Dataset
import torch

class MoleculeDataset(Dataset):
    def __init__(self, descriptors: list, targets: list = None):
        """
        Args:
            descriptors (list): List of descriptor dictionaries.
            targets (list, optional): List of target property dictionaries.
        """
        self.descriptors = descriptors
        self.targets = targets
        # Get descriptor names and ensure consistent order
        self.descriptor_names = sorted(descriptors[0].keys())
        if self.targets is not None and len(self.targets) > 0:
            # Get target names and ensure consistent order
            self.target_names = sorted(self.targets[0].keys())
        else:
            self.target_names = []

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, idx):
        descriptor = self.descriptors[idx]
        descriptor_values = [descriptor[name] for name in self.descriptor_names]
        sample = {
            'descriptors': torch.tensor(descriptor_values, dtype=torch.float32),
        }
        if self.targets is not None and len(self.targets) > 0:
            target = self.targets[idx]
            target_values = [target[name] for name in self.target_names]
            sample['targets'] = torch.tensor(target_values, dtype=torch.float32)
        return sample

    @property
    def input_dim(self):
        return len(self.descriptor_names)

    @property
    def output_dim(self):
        if self.target_names:
            return len(self.target_names)
        else:
            return 0  # No targets provided
