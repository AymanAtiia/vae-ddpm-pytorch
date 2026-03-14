# VAE and DDPM Implementation from Scratch

PyTorch implementations of Variational Autoencoder (VAE) and Denoising Diffusion Probabilistic Model (DDPM) architectures from scratch, trained on the CelebA dataset with comprehensive quantitative and qualitative evaluation.

## Table of Contents

- [Setup](#setup)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Generation](#generation)
- [Evaluation](#evaluation)
- [Visualization Scripts](#visualization-scripts)

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Checkpoints (Optional)

Pre-trained model checkpoints are available for download:
- [Download Checkpoints](https://drive.google.com/file/d/1gwWl0x26vdRcY1AIr0-B5W0NqVXHvJ7s/view?usp=sharing)

After downloading, extract the checkpoints to:
- `VAE/checkpoints/` for VAE models
- `DDPM/checkpoints/` for DDPM models

### 4. Dataset

The CelebA dataset will be automatically downloaded on first run. The data will be stored in `data/celeba/`.

## Repository Structure

```
vae-ddpm-pytorch/
├── VAE/                          # VAE implementation
│   ├── model.py                  # VAE architecture (Encoder, Decoder, VAE)
│   ├── train.py                  # Training script
│   ├── generate.py               # Sample generation script
│   ├── evaluate_vae.py           # Evaluation wrapper (in root)
│   ├── visualize_latent.py       # 2D PCA visualization of latent space
│   ├── interpolate.py            # Latent space morphing/interpolation
│   ├── traverse_latent.py        # Latent dimension traversal
│   ├── analyze_reconstruction.py # Reconstruction quality analysis
│   ├── latent_arithmetic.py      # Latent space arithmetic (attribute manipulation)
│   ├── utils.py                  # Logging and plotting utilities
│   ├── checkpoints/               # Saved model weights
│   ├── logs/                      # Training logs and curves
│   └── samples/                   # Generated samples and visualizations
├── DDPM/                         # DDPM implementation
│   ├── model.py                  # DDPM architecture (U-Net, forward/reverse diffusion)
│   ├── train.py                  # Training script
│   ├── generate.py               # Sample generation script
│   ├── evaluate_ddpm.py          # Evaluation wrapper (in root)
│   ├── visualize_forward_diffusion.py    # Forward diffusion visualization
│   ├── visualize_reverse_diffusion.py    # Reverse diffusion visualization
│   ├── visualize_reconstruction_denoising.py  # Denoising process visualization
│   ├── visualize_noise_prediction.py     # Noise prediction analysis
│   ├── plot_noise_schedule.py    # Noise schedule visualization
│   ├── utils.py                  # Logging and plotting utilities
│   ├── checkpoints/               # Saved model weights
│   ├── logs/                      # Training logs and curves
│   └── samples/                   # Generated samples and visualizations
├── evaluate.py                    # Shared evaluation module (FID, KID, LPIPS, IS)
├── evaluate_vae.py               # VAE evaluation script
├── evaluate_ddpm.py              # DDPM evaluation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Getting Started

### File Descriptions

#### VAE Module (`VAE/`)

- **`model.py`**: Defines the VAE architecture with convolutional encoder/decoder and 128-dimensional latent space.
- **`train.py`**: Main training script that loads CelebA, implements VAE loss (BCE + KL divergence), and saves best model based on validation loss.
- **`generate.py`**: Generates new samples by sampling random latent codes and decoding them to images.
- **`visualize_latent.py`**: Visualizes latent space structure using 2D PCA scatter plot.
- **`interpolate.py`**: Creates latent space morphing videos by linearly interpolating between two random latent codes.
- **`traverse_latent.py`**: Visualizes what individual latent dimensions control by traversing them while keeping others fixed.
- **`analyze_reconstruction.py`**: Analyzes reconstruction quality by identifying best/worst reconstructions on validation set.
- **`latent_arithmetic.py`**: Demonstrates semantic manipulation in latent space by computing and applying attribute vectors.
- **`utils.py`**: Utility functions for logging and plotting training curves.

#### DDPM Module (`DDPM/`)

- **`model.py`**: Defines the DDPM architecture with U-Net for noise prediction and forward/reverse diffusion processes (1000 timesteps).
- **`train.py`**: Main training script that loads CelebA, implements noise prediction loss (MSE), and saves best model based on validation loss.
- **`generate.py`**: Generates new samples by performing reverse diffusion process (1000 steps) and converting from [-1, 1] to [0, 1] range.
- **`visualize_forward_diffusion.py`**: Visualizes how images degrade with increasing noise during forward diffusion.
- **`visualize_reverse_diffusion.py`**: Visualizes the denoising process during generation showing how noise is gradually removed.
- **`visualize_reconstruction_denoising.py`**: Analyzes denoising quality by testing reconstruction on noisy images at different noise levels.
- **`visualize_noise_prediction.py`**: Analyzes the model's noise prediction accuracy by comparing predicted vs actual noise distributions.
- **`plot_noise_schedule.py`**: Visualizes the β schedule over timesteps showing how noise increases during forward diffusion.
- **`utils.py`**: Utility functions for logging and plotting training curves.

#### Evaluation Module (Root)

- **`evaluate.py`**: Shared evaluation module implementing FID, KID, LPIPS, and Inception Score metrics.
- **`evaluate_vae.py`**: VAE evaluation wrapper that loads trained model, generates samples, and computes all metrics.
- **`evaluate_ddpm.py`**: DDPM evaluation wrapper that loads trained model, generates samples, and computes all metrics.

## Training

### Train VAE

```bash
cd VAE
python train.py
```

Training parameters (defined in `VAE/train.py`):
- Latent dimension: 128
- Batch size: 128
- Learning rate: 1e-3
- Epochs: 50
- Image size: 64x64

Checkpoints are saved in `VAE/checkpoints/`:
- `vae_best.pth`: Best model (lowest validation loss)
- `vae_final.pth`: Final model after all epochs

Logs and training curves are saved in `VAE/logs/`.

### Train DDPM

```bash
cd DDPM
python train.py
```

Training parameters (defined in `DDPM/train.py`):
- Timesteps: 1000
- Batch size: 128
- Learning rate: 1e-4
- Epochs: 50
- Image size: 64x64

Checkpoints are saved in `DDPM/checkpoints/`:
- `ddpm_best.pth`: Best model (lowest validation loss)
- `ddpm_final.pth`: Final model after all epochs

Logs and training curves are saved in `DDPM/logs/`.

## Generation

### Generate VAE Samples

```bash
cd VAE
python generate.py
```

Generated samples are saved to `VAE/samples/generated_samples.png`.

### Generate DDPM Samples

```bash
cd DDPM
python generate.py
```

Generated samples are saved to `DDPM/samples/generated_samples.png`.

## Evaluation

### Evaluate VAE

```bash
python evaluate_vae.py
```

This will:
1. Load the trained VAE model from `VAE/checkpoints/vae_best.pth`
2. Generate 5000 samples
3. Compute FID, KID, LPIPS, and Inception Score
4. Print the results

### Evaluate DDPM

```bash
python evaluate_ddpm.py
```

This will:
1. Load the trained DDPM model from `DDPM/checkpoints/ddpm_best.pth`
2. Generate 5000 samples (this may take longer due to 1000-step reverse diffusion)
3. Compute FID, KID, LPIPS, and Inception Score
4. Print the results

## Visualization Scripts

### VAE Visualizations

#### Latent Space Visualization (2D PCA)
```bash
cd VAE
python visualize_latent.py
```
Output: `VAE/samples/latent_space_visualization.png`

#### Latent Space Morphing
```bash
cd VAE
python interpolate.py
```
Outputs:
- `VAE/samples/latent_morphing.png`: Grid of interpolated frames
- `VAE/samples/latent_morphing.mp4`: Morphing video

#### Latent Dimension Traversal
```bash
cd VAE
python traverse_latent.py
```
Output: `VAE/samples/latent_traversal.png`

#### Reconstruction Analysis
```bash
cd VAE
python analyze_reconstruction.py
```
Outputs:
- `VAE/samples/reconstruction_best.png`
- `VAE/samples/reconstruction_worst.png`
- `VAE/samples/reconstruction_random.png`

#### Latent Space Arithmetic
```bash
cd VAE
python latent_arithmetic.py
```
Output: `VAE/samples/latent_arithmetic.png`

### DDPM Visualizations

#### Forward Diffusion
```bash
cd DDPM
python visualize_forward_diffusion.py
```
Output: `DDPM/samples/forward_diffusion_trajectory.png`

#### Reverse Diffusion
```bash
cd DDPM
python visualize_reverse_diffusion.py
```
Output: `DDPM/samples/reverse_diffusion_trajectory.png`

#### Denoising Process
```bash
cd DDPM
python visualize_reconstruction_denoising.py
```
Output: `DDPM/samples/reconstruction_denoising.png`

#### Noise Prediction Analysis
```bash
cd DDPM
python visualize_noise_prediction.py
```
Output: `DDPM/samples/noise_prediction_distributions.png`

#### Noise Schedule
```bash
cd DDPM
python plot_noise_schedule.py
```
Output: `DDPM/samples/noise_schedule.png`

