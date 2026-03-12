import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
import os

# Constants
LATENT_DIM = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
DATA_DIR = "./data/celeba"
SAVE_DIR = "./checkpoints"
IMAGE_SIZE = 64

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CelebA(root=DATA_DIR, split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model
    model = VAE(LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()/len(data):.4f}, '
                      f'Recon: {recon_loss.item()/len(data):.4f}, KL: {kl_loss.item()/len(data):.4f}')
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'vae_epoch_{epoch+1}.pth'))
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'vae_final.pth'))
    print("Training complete!")

if __name__ == "__main__":
    train()

