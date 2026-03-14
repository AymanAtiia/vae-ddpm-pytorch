import torch
from torchvision.utils import save_image
from model import DDPM
import os

# Constants
CHECKPOINT_PATH = "./checkpoints/ddpm_best.pth"
OUTPUT_DIR = "./samples"
NUM_SAMPLES = 64
TIMESTEPS = 1000
IMAGE_SIZE = 64

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.register_schedule(device)
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate samples
    print("Generating samples...")
    with torch.no_grad():
        shape = (NUM_SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE)
        samples = model.p_sample_loop(shape, device)
        # Convert from [-1, 1] to [0, 1] for saving
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)
        save_image(samples, os.path.join(OUTPUT_DIR, "generated_samples.png"), nrow=8)
    
    print(f"Generated {NUM_SAMPLES} samples saved to {OUTPUT_DIR}/generated_samples.png")

if __name__ == "__main__":
    generate()


