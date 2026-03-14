import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VAE
import os

# Constants
LATENT_DIM = 128
CHECKPOINT_PATH = "./checkpoints/vae_best.pth"
DATA_DIR = "../data/celeba"
OUTPUT_DIR = "./samples"
NUM_SAMPLES = 32
IMAGE_SIZE = 64

def analyze_reconstruction_quality():
    """Analyze reconstruction quality: show best, worst, and average cases."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.CelebA(root=DATA_DIR, split='test', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Get a batch
    data, _ = next(iter(dataloader))
    data = data[:NUM_SAMPLES].to(device)
    
    # Reconstruct
    with torch.no_grad():
        recon, _, _ = model(data)
    
    # Compute reconstruction errors (MSE per image)
    mse_per_image = torch.mean((data - recon) ** 2, dim=(1, 2, 3))
    mse_sorted, indices = torch.sort(mse_per_image)
    
    # Get best, worst, and random samples
    best_indices = indices[:8]
    worst_indices = indices[-8:]
    random_indices = indices[torch.linspace(0, len(indices)-1, 8).long()]
    
    # Create comparison grids
    def create_comparison(indices_list):
        originals = data[indices_list]
        reconstructions = recon[indices_list]
        
        # Interleave: original, reconstruction, original, reconstruction...
        comparison = torch.zeros(len(indices_list) * 2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        for i, idx in enumerate(indices_list):
            comparison[i * 2] = originals[i]
            comparison[i * 2 + 1] = reconstructions[i]
        
        return comparison
    
    best_comparison = create_comparison(best_indices)
    worst_comparison = create_comparison(worst_indices)
    random_comparison = create_comparison(random_indices)
    
    # Save
    save_image(best_comparison, os.path.join(OUTPUT_DIR, "reconstruction_best.png"), nrow=2)
    save_image(worst_comparison, os.path.join(OUTPUT_DIR, "reconstruction_worst.png"), nrow=2)
    save_image(random_comparison, os.path.join(OUTPUT_DIR, "reconstruction_random.png"), nrow=2)
    
    print(f"Saved reconstruction analysis to {OUTPUT_DIR}/")
    print(f"Best MSE: {mse_sorted[0]:.4f}, Worst MSE: {mse_sorted[-1]:.4f}, Mean MSE: {mse_sorted.mean():.4f}")
    print("Each image pair shows: Original (left) | Reconstruction (right)")

if __name__ == "__main__":
    analyze_reconstruction_quality()

