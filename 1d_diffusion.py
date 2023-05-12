import math

import humanize
import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# TODO: Fix training loop. What should the input tensor be?


np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def generate_legendre_dataset(n=10, max_degree=10, coeff_bounds=(-5, 5), domain_bounds=(-1, 1), domain_size=100) -> np.ndarray:
    domain = np.linspace(*domain_bounds, domain_size)
    coeffs = np.random.randint(*coeff_bounds, size=(n, max_degree))
    dataset = np.zeros((n, domain_size))
    for i in range(n):
        dataset[i, :] = np.polynomial.legendre.Legendre(list(coeffs[i, :]))(domain)
    return dataset

dataset = generate_legendre_dataset(n=10_000)

class LegendreDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)
    
# Split the dataset into train, validation, and test sets
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Create Dataset instances
train_dataset = LegendreDataset(train_data)
val_dataset = LegendreDataset(val_data)
test_dataset = LegendreDataset(test_data)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, num_heads):
        super(LinearBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, input_size)
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.activation = nn.ReLU()
        self.multihead_attention = nn.MultiheadAttention(input_size, num_heads)
        self.linear2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = x.transpose(1, 2)  # Swap sequence and channel dimensions
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # Swap back sequence and channel dimensions
        attn_output, _ = self.multihead_attention(x, x, x)
        x = attn_output
        x = self.linear2(x)
        x = self.activation(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, embedding_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

    def forward(self, time):
        batch_size, _ = time.size()
        device = time.device
        
        if self.embedding_dim > 1:
            div_term = math.log(10000) / (self.embedding_dim - 1)
            div_term = torch.exp(torch.arange(0, self.embedding_dim, device=device) * -div_term)
        else:
            div_term = torch.tensor([1.0], device=device)

        pos = torch.arange(0, self.seq_len, device=device).float()
        pos = pos.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        embeddings = pos * div_term.unsqueeze(0).unsqueeze(1)
        
        return embeddings.squeeze(-1).sin()





class LinearUNet(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        super().__init__()
        self.encoder1 = LinearBlock(input_dim, 256, num_heads)
        self.encoder2 = LinearBlock(256, 128, num_heads)
        self.encoder3 = LinearBlock(128, 64, num_heads)
        self.encoder4 = LinearBlock(64, 32, num_heads)
        self.latent = nn.Sequential(
            LinearBlock(32, 32, num_heads),
            LinearBlock(32, 32, num_heads),
        )
        self.decoder1 = LinearBlock(32, 64, num_heads)
        self.decoder2 = LinearBlock(64, 128, num_heads)
        self.decoder3 = LinearBlock(128, 256, num_heads)
        self.decoder4 = LinearBlock(256, input_dim, num_heads)

    def forward(self, x):
        x = x.unsqueeze(2)  # Add channel dimension
        e1 = self.encoder1(x.transpose(0, 1))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        l = self.latent(e4)
        # Skip connections
        x = l + e4
        x = self.decoder1(x)
        x += e3
        x = self.decoder2(x)
        x += e2
        x = self.decoder3(x)
        x += e1
        x = self.decoder4(x)
        return x.squeeze(2)  # Remove the channel dimension

# GPT suggested training code
import torch.optim as optim

# Hyperparameters
num_epochs = 10
learning_rate = 1e-3

# Create the model
input_dim = 100
num_heads = 4
activation = nn.ReLU()

model = LinearUNet(input_dim, num_heads)
model = model.to(device)

# Create the position embeddings
pos_embedding = SinusoidalPositionEmbeddings(seq_len=100)
pos_embedding = pos_embedding.to(device)

# Noise schedule
num_timesteps = 1000
# NB: For DDPMs, the noise schedule is typically a monotonic decreasing function.
# We start the model training in a high noise environment and gradually decrease.
# the noise level as the model learns. It is not a requirement for the noise schedule
# to be linear though.
noise_schedule = torch.linspace(1.0, 0.0, num_timesteps).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply position embeddings
        embedded_inputs = pos_embedding(inputs)
        inputs = inputs + embedded_inputs

        # Add noise to inputs according to the noise schedule
        noise_level = noise_schedule[epoch % num_timesteps]
        # We don't need to explicityly pass the current noise level to the model
        # because the noise level is already baked into the inputs.
        noisy_inputs = inputs + torch.randn_like(inputs) * noise_level

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # The input size should be (batch_size, sequence_length, input_size), e.g.
        # (32, 1, 100) for a batch of 32 sequences of length 1 with 100 features.
        # Since the polynomials are univariate, the sequence length is 1.
        noisy_inputs = noisy_inputs.unsqueeze(1)
        outputs = model(noisy_inputs)

        # Loss calculation and backpropagation
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print training loss for this epoch
    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

    # Validate the model
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = pos_embedding(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    # save the model
    torch.save(model.state_dict(), f"model_{epoch}.pth")
