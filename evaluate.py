import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy import linalg
import lpips
from sklearn.metrics.pairwise import polynomial_kernel

# Constants
IMAGE_SIZE = 64
BATCH_SIZE = 50
NUM_SAMPLES = 5000  # Standard for FID/IS evaluation


class InceptionFeatureExtractor:
    """Extract features from Inception v3 for FID computation."""
    def __init__(self, device):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        # Replace final layer to get features instead of logits
        self.model.fc = torch.nn.Identity()
        if hasattr(self.model, 'AuxLogits') and self.model.AuxLogits is not None:
            self.model.AuxLogits.fc = torch.nn.Identity()
        self.model.eval()
        self.model = self.model.to(device)
    
    def __call__(self, images):
        """Extract features from images."""
        # Resize to 299x299 for Inception v3
        if images.size(-1) != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            features = self.model(images)
        return features.cpu().numpy()


class InceptionScoreModel:
    """Compute Inception Score using Inception v3 predictions."""
    def __init__(self, device):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.eval()
        self.model = self.model.to(device)
    
    def __call__(self, images):
        """Get predictions from images."""
        # Resize to 299x299 for Inception v3
        if images.size(-1) != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()


def compute_fid(real_features, fake_features):
    """Compute Fréchet Inception Distance."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Compute sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Compute sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def compute_inception_score(probs, splits=10):
    """Compute Inception Score."""
    scores = []
    for i in range(splits):
        part = probs[i * (len(probs) // splits): (i + 1) * (len(probs) // splits)]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([np.sum(p * np.log(p / py)) for p in part])))
    return np.mean(scores), np.std(scores)


def compute_kid(real_features, fake_features, subsample_size=1000):
    """Compute Kernel Inception Distance."""
    # Subsample for efficiency
    n_real = min(len(real_features), subsample_size)
    n_fake = min(len(fake_features), subsample_size)
    
    # Random subsampling
    real_idx = np.random.choice(len(real_features), n_real, replace=False)
    fake_idx = np.random.choice(len(fake_features), n_fake, replace=False)
    
    real_sub = real_features[real_idx]
    fake_sub = fake_features[fake_idx]
    
    # Compute polynomial kernel (degree 3, as in original paper)
    # K(x, y) = (gamma * <x, y> + coef0)^degree
    # Default: gamma=1/n_features, coef0=1, degree=3
    gamma = 1.0 / real_sub.shape[1]
    k_rr = polynomial_kernel(real_sub, real_sub, degree=3, gamma=gamma, coef0=1.0)
    k_ff = polynomial_kernel(fake_sub, fake_sub, degree=3, gamma=gamma, coef0=1.0)
    k_rf = polynomial_kernel(real_sub, fake_sub, degree=3, gamma=gamma, coef0=1.0)
    
    # Compute KID: mean of kernel matrices
    kid = (k_rr.mean() + k_ff.mean() - 2 * k_rf.mean())
    return kid


def compute_lpips(real_images, fake_images, lpips_model, device, num_pairs=1000):
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Computes average LPIPS between random pairs of real and fake images.
    """
    # Ensure images are in [-1, 1] range for LPIPS
    real_norm = (real_images * 2.0) - 1.0
    fake_norm = (fake_images * 2.0) - 1.0
    
    # Sample random pairs
    n_pairs = min(num_pairs, len(real_images), len(fake_images))
    real_idx = np.random.choice(len(real_images), n_pairs, replace=False)
    fake_idx = np.random.choice(len(fake_images), n_pairs, replace=False)
    
    distances = []
    with torch.no_grad():
        for i in range(0, n_pairs, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, n_pairs)
            real_batch = real_norm[real_idx[i:end_idx]].to(device)
            fake_batch = fake_norm[fake_idx[i:end_idx]].to(device)
            
            # Compute LPIPS distance
            dist = lpips_model(real_batch, fake_batch)
            distances.append(dist.cpu().numpy())
    
    return np.mean(distances)


def get_real_features(dataloader, feature_extractor, num_samples):
    """Extract features from real images."""
    features = []
    count = 0
    for data, _ in dataloader:
        if count >= num_samples:
            break
        data = data.to(feature_extractor.device)
        feat = feature_extractor(data)
        features.append(feat)
        count += len(data)
    return np.vstack(features)[:num_samples]


def evaluate_model(generate_fn, data_dir, split='test', num_samples=NUM_SAMPLES, device=None):
    """
    Evaluate a generative model using FID, KID, LPIPS, and Inception Score.
    
    Args:
        generate_fn: Function that generates images. Should return tensor of shape (N, 3, H, W) in [0, 1]
        data_dir: Directory containing dataset
        split: Dataset split to use for real images ('test' recommended)
        num_samples: Number of samples to evaluate
        device: Device to use (auto-detect if None)
    
    Returns:
        dict with 'fid', 'kid', 'lpips', 'is_mean', and 'is_std' keys
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating model on {device}...")
    print(f"Generating {num_samples} samples...")
    
    # Generate fake images
    fake_images = generate_fn(num_samples, device)
    if isinstance(fake_images, np.ndarray):
        fake_images = torch.from_numpy(fake_images)
    if fake_images.max() > 1.0:
        fake_images = fake_images / 255.0
    fake_images = fake_images.to(device)
    
    # Load real images
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.CelebA(root=data_dir, split=split, transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Get real images for LPIPS
    print("Loading real images...")
    real_images_list = []
    count = 0
    for data, _ in dataloader:
        if count >= num_samples:
            break
        real_images_list.append(data)
        count += len(data)
    real_images = torch.cat(real_images_list, dim=0)[:num_samples].to(device)
    
    # Extract features for FID/KID
    print("Extracting features from real images...")
    feature_extractor = InceptionFeatureExtractor(device)
    real_features = get_real_features(dataloader, feature_extractor, num_samples)
    
    print("Extracting features from generated images...")
    fake_features_list = []
    for i in range(0, len(fake_images), BATCH_SIZE):
        batch = fake_images[i:i+BATCH_SIZE]
        feat = feature_extractor(batch)
        fake_features_list.append(feat)
    fake_features = np.vstack(fake_features_list)
    
    # Compute FID
    print("Computing FID...")
    fid = compute_fid(real_features, fake_features)
    
    # Compute KID
    print("Computing KID...")
    kid = compute_kid(real_features, fake_features)
    
    # Compute LPIPS
    print("Computing LPIPS...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_score = compute_lpips(real_images, fake_images, lpips_model, device)
    
    # Compute Inception Score
    print("Computing Inception Score...")
    is_model = InceptionScoreModel(device)
    probs_list = []
    for i in range(0, len(fake_images), BATCH_SIZE):
        batch = fake_images[i:i+BATCH_SIZE]
        probs = is_model(batch)
        probs_list.append(probs)
    probs = np.vstack(probs_list)
    is_mean, is_std = compute_inception_score(probs)
    
    results = {
        'fid': fid,
        'kid': kid,
        'lpips': lpips_score,
        'is_mean': is_mean,
        'is_std': is_std
    }
    
    return results

