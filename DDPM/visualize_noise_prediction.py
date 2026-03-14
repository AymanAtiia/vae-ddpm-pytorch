import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DDPM
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
CHECKPOINT_PATH = "./checkpoints/ddpm_best.pth"
DATA_DIR = "../data/celeba"
OUTPUT_DIR = "./samples"
TIMESTEPS = 1000
IMAGE_SIZE = 64
BATCH_SIZE = 32
SELECTED_TIMESTEPS = [100, 300, 600, 900]  # Different noise levels to visualize
NUM_BATCHES = 10  # Number of batches to collect statistics

def visualize_noise_prediction():
    """Visualize predicted noise vs true noise distributions at different timesteps."""
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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    print("Collecting noise prediction statistics...")
    
    # Collect statistics for each timestep
    results = {}
    
    with torch.no_grad():
        for t_val in SELECTED_TIMESTEPS:
            t = torch.full((BATCH_SIZE,), t_val, device=device, dtype=torch.long)
            
            all_true_noise = []
            all_pred_noise = []
            all_errors = []
            mse_per_batch = []
            
            # Collect data from multiple batches
            for batch_idx, (x_0, _) in enumerate(dataloader):
                if batch_idx >= NUM_BATCHES:
                    break
                    
                x_0 = x_0.to(device)
                
                # Sample noise and create noisy image
                noise_true = torch.randn_like(x_0)
                x_t = model.q_sample(x_0, t[:len(x_0)], noise_true)
                
                # Predict noise
                noise_pred = model.model(x_t, t[:len(x_0)])
                
                # Flatten and collect
                noise_true_flat = noise_true.cpu().numpy().flatten()
                noise_pred_flat = noise_pred.cpu().numpy().flatten()
                errors_flat = (noise_true_flat - noise_pred_flat)
                
                all_true_noise.append(noise_true_flat)
                all_pred_noise.append(noise_pred_flat)
                all_errors.append(errors_flat)
                
                # Compute MSE per batch
                mse = torch.mean((noise_true - noise_pred) ** 2).item()
                mse_per_batch.append(mse)
            
            # Concatenate all batches
            true_noise_all = np.concatenate(all_true_noise)
            pred_noise_all = np.concatenate(all_pred_noise)
            errors_all = np.concatenate(all_errors)
            
            results[t_val] = {
                'true_noise': true_noise_all,
                'pred_noise': pred_noise_all,
                'errors': errors_all,
                'mse_mean': np.mean(mse_per_batch),
                'mse_std': np.std(mse_per_batch)
            }
            
            print(f"Timestep {t_val}: MSE = {results[t_val]['mse_mean']:.4f} ± {results[t_val]['mse_std']:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(len(SELECTED_TIMESTEPS), 3, figsize=(15, 4 * len(SELECTED_TIMESTEPS)))
    
    for idx, t_val in enumerate(SELECTED_TIMESTEPS):
        data = results[t_val]
        
        # Plot 1: Distribution of true noise vs predicted noise
        axes[idx, 0].hist(data['true_noise'], bins=100, alpha=0.5, label='True Noise', density=True, color='blue')
        axes[idx, 0].hist(data['pred_noise'], bins=100, alpha=0.5, label='Predicted Noise', density=True, color='red')
        axes[idx, 0].set_xlabel('Noise Value')
        axes[idx, 0].set_ylabel('Density')
        axes[idx, 0].set_title(f'Noise Distributions at t={t_val}')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error distribution (predicted - true)
        axes[idx, 1].hist(data['errors'], bins=100, alpha=0.7, color='green', density=True)
        axes[idx, 1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[idx, 1].set_xlabel('Error (Predicted - True)')
        axes[idx, 1].set_ylabel('Density')
        axes[idx, 1].set_title(f'Prediction Error Distribution (t={t_val})\nMSE: {data["mse_mean"]:.4f}')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot of predicted vs true (subsampled for visualization)
        sample_size = min(10000, len(data['true_noise']))
        indices = np.random.choice(len(data['true_noise']), sample_size, replace=False)
        axes[idx, 2].scatter(data['true_noise'][indices], data['pred_noise'][indices], 
                           alpha=0.1, s=1)
        # Perfect prediction line
        min_val = min(data['true_noise'].min(), data['pred_noise'].min())
        max_val = max(data['true_noise'].max(), data['pred_noise'].max())
        axes[idx, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[idx, 2].set_xlabel('True Noise')
        axes[idx, 2].set_ylabel('Predicted Noise')
        axes[idx, 2].set_title(f'Predicted vs True Noise (t={t_val})')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "noise_prediction_distributions.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved noise prediction distributions to {output_path}")

if __name__ == "__main__":
    visualize_noise_prediction()

