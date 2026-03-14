import torch
from torchvision.utils import save_image
from model import DDPM
import os

# Constants
CHECKPOINT_PATH = "./checkpoints/ddpm_best.pth"
OUTPUT_DIR = "./samples"
NUM_SAMPLES = 1
TIMESTEPS = 1000
IMAGE_SIZE = 64
SAVE_INTERVALS = [999, 750, 500, 250, 100, 50, 0]  # Timesteps to save (0 to TIMESTEPS-1)

def visualize_reverse_diffusion():
    """Visualize the reverse diffusion process: from noise to image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.register_schedule(device)
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start from pure noise
    shape = (NUM_SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE)
    img = torch.randn(shape, device=device)
    
    # Store images at different timesteps
    images = []
    
    print("Generating reverse diffusion trajectory...")
    with torch.no_grad():
        for i in reversed(range(TIMESTEPS)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = model.p_sample(img, t)
            
            # Save at specified intervals
            if i in SAVE_INTERVALS:
                # Convert from [-1, 1] to [0, 1] for saving
                img_normalized = (img + 1.0) / 2.0
                img_normalized = torch.clamp(img_normalized, 0.0, 1.0)
                images.append(img_normalized)
                print(f"Saved timestep {i}")
    
    # Create grid
    grid = torch.cat(images, dim=0)
    output_path = os.path.join(OUTPUT_DIR, "reverse_diffusion_trajectory.png")
    save_image(grid, output_path, nrow=len(SAVE_INTERVALS))
    print(f"Saved reverse diffusion trajectory to {output_path}")

if __name__ == "__main__":
    visualize_reverse_diffusion()

