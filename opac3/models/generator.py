# opac3/models/generator.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

class MoleculeVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MoleculeVAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Assuming descriptors are normalized between 0 and 1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    VAE loss function.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def train_vae(dataset, input_dim, latent_dim=20, epochs=100, batch_size=32, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MoleculeVAE(input_dim, 128, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch['descriptors']
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model
