import torch
import matplotlib.pyplot as plt
from model import DDPM
import os

# Constants
CHECKPOINT_PATH = "./checkpoints/ddpm_best.pth"
OUTPUT_DIR = "./samples"
TIMESTEPS = 1000

def plot_noise_schedule():
    """Plot the noise schedule (beta, alpha, alpha_cumprod)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (just for the noise schedule)
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.register_schedule(device)
    
    # Get schedules
    beta = model.beta.cpu().numpy()
    alpha = model.alpha.cpu().numpy()
    alpha_cumprod = model.alpha_cumprod.cpu().numpy()
    
    timesteps = range(TIMESTEPS)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Beta schedule
    axes[0].plot(timesteps, beta)
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Beta')
    axes[0].set_title('Beta Schedule (Noise Level)')
    axes[0].grid(True)
    
    # Alpha schedule
    axes[1].plot(timesteps, alpha)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Alpha')
    axes[1].set_title('Alpha Schedule (1 - Beta)')
    axes[1].grid(True)
    
    # Alpha cumprod schedule
    axes[2].plot(timesteps, alpha_cumprod)
    axes[2].set_xlabel('Timestep')
    axes[2].set_ylabel('Alpha Cumprod')
    axes[2].set_title('Cumulative Product of Alpha')
    axes[2].grid(True)
    
    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "noise_schedule.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved noise schedule plot to {output_path}")

if __name__ == "__main__":
    plot_noise_schedule()


