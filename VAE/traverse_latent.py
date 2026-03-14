import torch
from torchvision.utils import save_image
from model import VAE
import os

# Constants
LATENT_DIM = 128
CHECKPOINT_PATH = "./checkpoints/vae_best.pth"
OUTPUT_DIR = "./samples"
NUM_DIMS_TO_SHOW = 8  # Number of dimensions to visualize
STEPS_PER_DIM = 10  # Number of steps for each dimension
STD_RANGE = 3.0  # Range in standard deviations

def traverse_latent_dimensions():
    """Traverse individual latent dimensions to see what each controls."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start from a random latent code (or zero)
    z_base = torch.zeros(1, LATENT_DIM).to(device)
    
    all_images = []
    
    with torch.no_grad():
        for dim in range(NUM_DIMS_TO_SHOW):
            dim_images = []
            
            # Traverse this dimension from -STD_RANGE to +STD_RANGE
            for step in range(STEPS_PER_DIM):
                z = z_base.clone()
                # Vary this dimension
                value = -STD_RANGE + (2 * STD_RANGE * step / (STEPS_PER_DIM - 1))
                z[0, dim] = value
                
                # Decode
                img = model.decoder(z)
                dim_images.append(img)
            
            # Concatenate images for this dimension
            all_images.append(torch.cat(dim_images, dim=0))
    
    # Create grid: each row is a dimension, each column is a step
    grid = torch.cat(all_images, dim=0)
    save_image(grid, os.path.join(OUTPUT_DIR, "latent_traversal.png"), 
               nrow=STEPS_PER_DIM)
    print(f"Saved latent dimension traversal to {OUTPUT_DIR}/latent_traversal.png")
    print(f"Each row shows one latent dimension, columns show values from -{STD_RANGE}σ to +{STD_RANGE}σ")

if __name__ == "__main__":
    traverse_latent_dimensions()

