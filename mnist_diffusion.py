import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTDataset(datasets.MNIST):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, image

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNISTDataset(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SimpleDDPM(nn.Module):
    def __init__(self):
        super(SimpleDDPM, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        
        # Middle layer
        self.middle = ConvBlock(64, 64)
        
        # Decoder
        self.dec3 = ConvBlock(128, 32)
        self.dec2 = ConvBlock(64, 16)
        self.dec1 = ConvBlock(32, 1)

        self.pool = nn.MaxPool2d(2)
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        # Middle layer
        x = self.middle(x)
        
        # Decoder
        x = self.upconv(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.dec3(x)
        
        x = self.upconv(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec2(x)
        
        x = self.upconv(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec1(x)
        
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleDDPM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        noisy_images = images + torch.randn_like(images) * 0.1
        optimizer.zero_grad()

        outputs = model(noisy_images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
