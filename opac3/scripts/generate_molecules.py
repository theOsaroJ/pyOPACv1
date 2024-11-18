# opac3/scripts/generate_molecules.py

import argparse
import torch
from opac3.data.dataset import MoleculeDataset
from opac3.models.generator import train_vae, MoleculeVAE
from opac3.utils.logger import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train a VAE and generate new molecule descriptors.')
    parser.add_argument('--descriptors-file', type=str, required=True, help='CSV file containing descriptors.')
    parser.add_argument('--vae-model-output', type=str, required=True, help='File to save the trained VAE model.')
    parser.add_argument('--generated-descriptors-output', type=str, required=True, help='File to save generated descriptors (CSV).')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of new descriptors to generate.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of VAE training epochs.')
    args = parser.parse_args()

    # Load descriptors
    df_descriptors = pd.read_csv(args.descriptors_file)
    descriptor_columns = [col for col in df_descriptors.columns if col != 'mol_id']
    descriptors = df_descriptors[descriptor_columns].to_dict('records')

    # Create dataset
    dummy_targets = [{'dummy': 0} for _ in descriptors]  # Targets are not used in VAE training
    dataset = MoleculeDataset(descriptors, dummy_targets)

    # Train VAE
    input_dim = dataset.input_dim
    vae_model = train_vae(dataset, input_dim, epochs=args.epochs)

    # Save VAE model
    torch.save(vae_model.state_dict(), args.vae_model_output)
    logger.info(f"Saved trained VAE model to {args.vae_model_output}.")

    # Generate new descriptors
    vae_model.eval()
    with torch.no_grad():
        latent_dim = vae_model.fc21.out_features
        z = torch.randn(args.num_samples, latent_dim)
        generated_descriptors = vae_model.decode(z).numpy()

    # Convert to DataFrame
    generated_descriptors_df = pd.DataFrame(generated_descriptors, columns=descriptor_columns)
    generated_descriptors_df.to_csv(args.generated_descriptors_output, index=False)
    logger.info(f"Saved generated descriptors to {args.generated_descriptors_output}.")

if __name__ == '__main__':
    main()
