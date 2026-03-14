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
NUM_EXAMPLES = 8
NOISE_TIMESTEP = 100  # How much to noise the image before denoising

def visualize_reconstruction_denoising():
    """Take real images, noise them, then denoise them back."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.register_schedule(device)
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load real images
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CelebA(root=DATA_DIR, split='test', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=NUM_EXAMPLES, shuffle=True, num_workers=4)
    
    # Get a batch
    x_0, _ = next(iter(dataloader))
    x_0 = x_0.to(device)
    
    print(f"Adding noise at timestep {NOISE_TIMESTEP}...")
    with torch.no_grad():
        # Add noise
        t_noise = torch.full((NUM_EXAMPLES,), NOISE_TIMESTEP, device=device, dtype=torch.long)
        x_noisy = model.q_sample(x_0, t_noise)
        
        # Denoise back to t=0
        print("Denoising back to clean image...")
        x_denoised = x_noisy.clone()
        for i in reversed(range(NOISE_TIMESTEP + 1)):
            t = torch.full((NUM_EXAMPLES,), i, device=device, dtype=torch.long)
            x_denoised = model.p_sample(x_denoised, t)
    
    # Create comparison grid: original, noisy, denoised
    x_0_viz = (x_0 + 1.0) / 2.0
    x_noisy_viz = (x_noisy + 1.0) / 2.0
    x_denoised_viz = (x_denoised + 1.0) / 2.0
    
    # Interleave: original, noisy, denoised for each example
    comparison = torch.zeros(NUM_EXAMPLES * 3, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    for i in range(NUM_EXAMPLES):
        comparison[i * 3] = x_0_viz[i]
        comparison[i * 3 + 1] = x_noisy_viz[i]
        comparison[i * 3 + 2] = x_denoised_viz[i]
    
    output_path = os.path.join(OUTPUT_DIR, "reconstruction_denoising.png")
    save_image(comparison, output_path, nrow=3)
    print(f"Saved reconstruction denoising to {output_path}")
    print("Each row shows: Original | Noisy (t={}) | Denoised".format(NOISE_TIMESTEP))

if __name__ == "__main__":
    visualize_reconstruction_denoising()


