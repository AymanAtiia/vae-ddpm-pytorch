import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DDPM
from utils import plot_training_curves, Logger
import os
from datetime import datetime

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
DATA_DIR = "../data/celeba"
SAVE_DIR = "./checkpoints"
LOG_DIR = "./logs"
IMAGE_SIZE = 64
TIMESTEPS = 1000

def loss_function(noise_pred, noise):
    """MSE loss between predicted and actual noise."""
    return nn.functional.mse_loss(noise_pred, noise)

def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
            
            # Forward pass
            noise_pred, noise = model(data, t)
            loss = loss_function(noise_pred, noise)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(dataloader)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(LOG_DIR, f"training_{timestamp}.log"))
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CelebA(root=DATA_DIR, split='train', transform=transform, download=True)
    val_dataset = datasets.CelebA(root=DATA_DIR, split='valid', transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.register_schedule(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
            
            # Forward pass
            noise_pred, noise = model(data, t)
            loss = loss_function(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        logger.log(f"Epoch {epoch}: Train={avg_loss:.4f}, Val={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ddpm_best.pth'))
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ddpm_final.pth'))
    plot_training_curves(train_losses, val_losses, os.path.join(LOG_DIR, f"curves_{timestamp}.png"))
    logger.close()

if __name__ == "__main__":
    train()


