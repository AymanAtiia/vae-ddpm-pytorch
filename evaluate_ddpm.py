import torch
import sys
import time
sys.path.append('DDPM')
from model import DDPM
from evaluate import evaluate_model

# Constants
TIMESTEPS = 1000
CHECKPOINT_PATH = "DDPM/checkpoints/ddpm_best.pth"
DATA_DIR = "data/celeba"
NUM_SAMPLES = 5000
BATCH_SIZE = 32  # Smaller batch size for DDPM since generation is slower

def generate_ddpm_samples(num_samples, device):
    """Generate samples from DDPM model."""
    model = DDPM(timesteps=TIMESTEPS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.register_schedule(device)
    model.eval()
    
    samples = []
    batch_times = []
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, num_samples - len(samples))
            shape = (batch_size, 3, 64, 64)
            
            # Time single batch generation
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            # Generate batch
            batch = model.p_sample_loop(shape, device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Convert from [-1, 1] to [0, 1] for evaluation
            batch = (batch + 1.0) / 2.0
            batch = torch.clamp(batch, 0.0, 1.0)
            
            samples.append(batch)
            print(f"Generated {len(samples) * BATCH_SIZE}/{num_samples} samples...")
    
    # Calculate timing statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_time_per_sample = avg_batch_time / BATCH_SIZE
    total_time = sum(batch_times)
    
    print(f"\nGeneration Timing:")
    print(f"  Average batch time: {avg_batch_time:.2f} s")
    print(f"  Average time per sample: {avg_time_per_sample:.2f} s")
    print(f"  Total generation time: {total_time:.2f} s")
    
    return torch.cat(samples, dim=0)[:num_samples]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate_model(generate_ddpm_samples, DATA_DIR, split='test', 
                            num_samples=NUM_SAMPLES, device=device)
    print(f"\nFinal Results:")
    print(f"FID: {results['fid']:.2f}")
    print(f"KID: {results['kid']:.4f}")
    print(f"LPIPS: {results['lpips']:.4f}")
    print(f"IS: {results['is_mean']:.2f} ± {results['is_std']:.2f}")


