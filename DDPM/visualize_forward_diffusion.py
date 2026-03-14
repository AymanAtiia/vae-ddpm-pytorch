import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DDPM
import os

# Constants
CHECKPOINT_PATH = "./checkpoints/ddpm_best.pth"
DATA_DIR = "../data/celeba"
OUTPUT_DIR = "./samples"
TIMESTEPS = 1000
IMAGE_SIZE = 64
SAVE_INTERVALS = [0, 50, 100, 250, 500, 750, 999]  # Timesteps to save (0 to TIMESTEPS-1)

def visualize_forward_diffusion():
    """Visualize the forward diffusion process: progressively adding noise to a real image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (just for the noise schedule)
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.register_schedule(device)
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load a real image
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    dataset = datasets.CelebA(root=DATA_DIR, split='test', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    # Get one image
    x_0, _ = next(iter(dataloader))
    x_0 = x_0.to(device)
    
    # Store images at different timesteps
    images = []
    
    print("Generating forward diffusion trajectory...")
    with torch.no_grad():
        for t_val in SAVE_INTERVALS:
            t = torch.full((1,), t_val, device=device, dtype=torch.long)
            x_t = model.q_sample(x_0, t)
            
            # Convert from [-1, 1] to [0, 1] for saving
            img_normalized = (x_t + 1.0) / 2.0
            img_normalized = torch.clamp(img_normalized, 0.0, 1.0)
            images.append(img_normalized)
            print(f"Saved timestep {t_val}")
    
    # Create grid
    grid = torch.cat(images, dim=0)
    output_path = os.path.join(OUTPUT_DIR, "forward_diffusion_trajectory.png")
    save_image(grid, output_path, nrow=len(SAVE_INTERVALS))
    print(f"Saved forward diffusion trajectory to {output_path}")

if __name__ == "__main__":
    visualize_forward_diffusion()

