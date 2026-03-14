import matplotlib.pyplot as plt
import os


class Logger:
    """Logger that writes to both console and file."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file = open(log_file, 'w')
    
    def log(self, message):
        print(message)
        self.file.write(message + '\n')
        self.file.flush()
    
    def close(self):
        self.file.close()


def plot_training_curves(epoch_losses, epoch_recon_losses, epoch_kl_losses, save_path):
    """Plot and save training curves for total loss, reconstruction loss, and KL divergence."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epoch_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epoch_recon_losses, label='Reconstruction Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_kl_losses, label='KL Divergence', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

