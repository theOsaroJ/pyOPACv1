# opac3/models/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from opac3.utils.logger import get_logger

logger = get_logger(__name__)

class PropertyPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PropertyPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_model(dataset, input_dim, output_dim, epochs=100, batch_size=32, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PropertyPredictor(input_dim, 128, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch['descriptors'], batch['targets']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model
