"""
imports

define image dimensions

define generator model
    takes noise, returns image

define discriminator model
    takes image, returns classification
    
train models
    load dataset
    generate half batch of fake images
    train discriminator on half real, half fake
    
define optimizer
build and compile models

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Define image dimensions
img_size = 32  # CIFAR10 images are 32x32
channels = 3
latent_dim = 100

# Define generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * (img_size // 4) * (img_size // 4)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, img_size // 4, img_size // 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 4) * (img_size // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

def train_models(continue_training=True):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset using torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Load saved models if continuing training
    if continue_training:
        generator.load_state_dict(torch.load('generator.pth'))
        discriminator.load_state_dict(torch.load('discriminator.pth'))
        print("Loaded saved models.")

    # Define optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    num_epochs = 10
    fixed_noise = torch.randn(64, latent_dim, device=device)
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(trainloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train discriminator
            d_optimizer.zero_grad()

            # Train on real images
            real_labels = torch.ones(batch_size, 1, device=device)
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)

            # Train on fake images
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

        # Generate and save images
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                generated_images = generator(fixed_noise).cpu()
                grid = make_grid(generated_images, normalize=True, nrow=8)
                plt.figure(figsize=(10, 10))
                plt.imshow(grid.permute(1, 2, 0))
                plt.axis('off')
                plt.savefig(f'generated_images_epoch_{epoch+1}.png')
                plt.close()

    # Save models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

if __name__ == "__main__":
    train_models(continue_training=True)