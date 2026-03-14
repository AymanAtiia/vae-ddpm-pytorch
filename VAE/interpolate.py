import torch
from torchvision.utils import save_image
from model import VAE
import os

# Constants
LATENT_DIM = 128
CHECKPOINT_PATH = "./checkpoints/vae_best.pth"
OUTPUT_DIR = "./samples"
NUM_FRAMES = 60

def create_latent_morphing():
    """Create morphing video by interpolating between two random points in latent space."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Sample two random latent codes
    z_start = torch.randn(1, LATENT_DIM).to(device)
    z_end = torch.randn(1, LATENT_DIM).to(device)
    print("Interpolating between two random latent codes")
    
    # Generate frames
    frames = []
    with torch.no_grad():
        for i in range(NUM_FRAMES):
            t = i / (NUM_FRAMES - 1)
            z_interp = (1 - t) * z_start + t * z_end
            frame = model.decoder(z_interp)
            frames.append(frame)
    
    # Save combined grid image
    all_frames = torch.cat(frames, dim=0)
    grid_path = os.path.join(OUTPUT_DIR, "latent_morphing.png")
    save_image(all_frames, grid_path, nrow=10)
    print(f"Saved morphing grid to {grid_path}")
    
    # Create video
    try:
        import imageio
        video_path = os.path.join(OUTPUT_DIR, "latent_morphing.mp4")
        frame_images = []
        for frame in frames:
            # Convert tensor to numpy array for imageio
            frame_np = frame.squeeze(0).cpu().permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype('uint8')
            frame_images.append(frame_np)
        
        imageio.mimwrite(video_path, frame_images, fps=10)
        print(f"Created morphing video: {video_path}")
    except ImportError:
        print("imageio not available, skipping video creation")

if __name__ == "__main__":
    create_latent_morphing()
