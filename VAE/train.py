import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
from utils import plot_training_curves, Logger
import os
from datetime import datetime

# Constants
LATENT_DIM = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
DATA_DIR = "../data/celeba"
SAVE_DIR = "./checkpoints"
LOG_DIR = "./logs"
IMAGE_SIZE = 64

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar)
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader.dataset)

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
    ])
    train_dataset = datasets.CelebA(root=DATA_DIR, split='train', transform=transform, download=True)
    val_dataset = datasets.CelebA(root=DATA_DIR, split='valid', transform=transform, download=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = VAE(LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    epoch_losses, epoch_recon_losses, epoch_kl_losses = [], [], []
    best_val_loss = float('inf')
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = total_recon_loss = total_kl_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon_loss / len(train_loader.dataset)
        avg_kl = total_kl_loss / len(train_loader.dataset)
        
        epoch_losses.append(avg_loss)
        epoch_recon_losses.append(avg_recon)
        epoch_kl_losses.append(avg_kl)
        
        val_loss = evaluate(model, val_loader, device)
        logger.log(f"Epoch {epoch}: Train={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Val={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vae_best.pth'))
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vae_final.pth'))
    plot_training_curves(epoch_losses, epoch_recon_losses, epoch_kl_losses, 
                        os.path.join(LOG_DIR, f"curves_{timestamp}.png"))
    logger.close()

if __name__ == "__main__":
    train()

