import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
LATENT_DIM = 128
CHECKPOINT_PATH = "./checkpoints/vae_best.pth"
DATA_DIR = "../data/celeba"
IMAGE_SIZE = 64
NUM_SAMPLES = 1000
OUTPUT_DIR = "./samples"

def visualize_latent_space():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.CelebA(root=DATA_DIR, split='train', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Encode images to get latent codes
    latent_codes = []
    with torch.no_grad():
        count = 0
        for data, _ in dataloader:
            if count >= NUM_SAMPLES:
                break
            data = data.to(device)
            mu, _ = model.encoder(data)
            latent_codes.append(mu.cpu().numpy())
            count += len(data)
    
    latent_codes = np.vstack(latent_codes)[:NUM_SAMPLES]
    print(f"Encoded {len(latent_codes)} samples to latent space")
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {explained_var.sum():.2%}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, s=10)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Latent Space (2D PCA)')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "latent_space_visualization.png"), dpi=150)
    plt.close()
    
    print(f"Visualization saved to {OUTPUT_DIR}/latent_space_visualization.png")

if __name__ == "__main__":
    visualize_latent_space()

