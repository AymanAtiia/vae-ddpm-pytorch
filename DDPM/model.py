import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Block(nn.Module):
    """Basic conv block with time embedding."""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        t_emb = t_emb[(..., ) + (None, ) * 2]  # Add spatial dimensions
        
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        h = h + t_emb  # Add time embedding
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # Down/Up sample
        return self.transform(h)


class UNet(nn.Module):
    """Simple U-Net for denoising."""
    def __init__(self, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Downsampling
        self.down1 = Block(3, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        self.down3 = Block(128, 256, time_emb_dim)
        
        # Bottleneck
        self.bot1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot3 = nn.Conv2d(512, 256, 3, padding=1)
        
        # Upsampling
        self.up1 = Block(256, 128, time_emb_dim, up=True)
        self.up2 = Block(128, 64, time_emb_dim, up=True)
        self.up3 = Block(64, 64, time_emb_dim, up=True)
        self.out = nn.Conv2d(64, 3, 1)  # Predict noise (no activation)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)
        
        # U-Net
        down1 = self.down1(x, t)  # 64x64 -> 32x32
        down2 = self.down2(down1, t)  # 32x32 -> 16x16
        down3 = self.down3(down2, t)  # 16x16 -> 8x8
        
        # Bottleneck
        bot = self.bot1(down3)
        bot = self.bot2(bot)
        bot = self.bot3(bot)
        
        # Upsampling with skip connections
        up1 = self.up1(torch.cat((bot, down3), dim=1), t)  # 8x8 -> 16x16
        up2 = self.up2(torch.cat((up1, down2), dim=1), t)  # 16x16 -> 32x32
        up3 = self.up3(torch.cat((up2, down1), dim=1), t)  # 32x32 -> 64x64
        
        output = self.out(up3)
        return output


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model."""
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.model = UNet()
        
        # Linear noise schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = nn.functional.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute for sampling
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = self.beta * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def register_schedule(self, device):
        """Move precomputed values to device."""
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to images."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_sample(self, x, t):
        """Reverse diffusion: denoise one step."""
        # Predict noise
        noise_pred = self.model(x, t)
        
        # Compute coefficients
        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1, 1, 1)
                
        # Compute posterior mean
        posterior_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)
        
        # Sample (no noise at t=0)
        posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        # Set noise to zero for t=0
        noise = torch.where((t == 0).reshape(-1, 1, 1, 1), torch.zeros_like(noise), noise)
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape, device):
        """Generate samples by iteratively denoising."""
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        
        return img

    def forward(self, x, t):
        """Training forward pass: predict noise."""
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to image
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_noisy, t)
        
        return noise_pred, noise

