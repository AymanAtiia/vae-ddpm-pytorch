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
IMAGE_SIZE = 64

def latent_arithmetic():
    """Demonstrate latent space arithmetic: attribute manipulation."""
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
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    
    # Get a batch with attributes
    data, attributes = next(iter(dataloader))
    data = data.to(device)
    
    # CelebA attributes: 0=Smiling, 1=Male, 2=Young, etc.
    # We'll try to find attribute vectors by averaging differences
    with torch.no_grad():
        # Encode all images
        mu_all, _ = model.encoder(data)
        
        # Find images with/without smiling (attribute 31 in CelebA)
        smiling_idx = 31
        has_attr = attributes[:, smiling_idx] == 1
        no_attr = attributes[:, smiling_idx] == 0
        
        # Compute attribute vector: average of "with" - average of "without"
        # This gives us a 128D direction vector pointing toward the attribute
        attr_vector = mu_all[has_attr].mean(dim=0) - mu_all[no_attr].mean(dim=0)
        
        # Normalize
        attr_vector = attr_vector / (attr_vector.norm() + 1e-8)
        
        # Pick a few base images (non-smiling)
        base_indices = torch.where(no_attr)[0][:4]
        
        results = []
        for idx in base_indices:
            z_base = mu_all[idx:idx+1]
            
            # Original
            img_original = model.decoder(z_base)
            results.append(img_original)
            
            # With attribute added (different strengths)
            for strength in [0.5, 1.0, 1.5]:
                z_modified = z_base + strength * attr_vector.unsqueeze(0)
                img_modified = model.decoder(z_modified)
                results.append(img_modified)
        
        # Create grid
        grid = torch.cat(results, dim=0)
        save_image(grid, os.path.join(OUTPUT_DIR, "latent_arithmetic.png"), nrow=4)
        print(f"Saved latent arithmetic to {OUTPUT_DIR}/latent_arithmetic.png")
        print("Each row: Original | +0.5*attr | +1.0*attr | +1.5*attr")

if __name__ == "__main__":
    latent_arithmetic()

