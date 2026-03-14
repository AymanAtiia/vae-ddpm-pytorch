import torch
import sys
import time
sys.path.append('VAE')
from model import VAE
from evaluate import evaluate_model

# Constants
LATENT_DIM = 128
CHECKPOINT_PATH = "VAE/checkpoints/vae_best.pth"
DATA_DIR = "data/celeba"
NUM_SAMPLES = 5000
BATCH_SIZE = 128

def generate_vae_samples(num_samples, device):
    """Generate samples from VAE model."""
    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    samples = []
    batch_times = []
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, num_samples - len(samples))
            
            # Time single batch generation
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            batch = model.decoder(z)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            samples.append(batch)
    
    # Calculate timing statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_time_per_sample = avg_batch_time / BATCH_SIZE
    total_time = sum(batch_times)
    
    print(f"\nGeneration Timing:")
    print(f"  Average batch time: {avg_batch_time*1000:.2f} ms")
    print(f"  Average time per sample: {avg_time_per_sample*1000:.2f} ms")
    print(f"  Total generation time: {total_time:.2f} s")
    
    return torch.cat(samples, dim=0)[:num_samples]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate_model(generate_vae_samples, DATA_DIR, split='test', 
                            num_samples=NUM_SAMPLES, device=device)
    print(f"\nFinal Results:")
    print(f"FID: {results['fid']:.2f}")
    print(f"KID: {results['kid']:.4f}")
    print(f"LPIPS: {results['lpips']:.4f}")
    print(f"IS: {results['is_mean']:.2f} ± {results['is_std']:.2f}")

