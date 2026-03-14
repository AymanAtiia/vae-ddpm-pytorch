import torch
from torchvision.utils import save_image
from model import VAE
import os

# Constants
LATENT_DIM = 128
CHECKPOINT_PATH = "./checkpoints/vae_best.pth"
OUTPUT_DIR = "./samples"
NUM_SAMPLES = 64

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    with torch.no_grad():
        z = torch.randn(NUM_SAMPLES, LATENT_DIM).to(device)
        samples = model.decoder(z)
        save_image(samples, os.path.join(OUTPUT_DIR, "generated_samples.png"), nrow=8)

if __name__ == "__main__":
    generate()

