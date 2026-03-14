import matplotlib.pyplot as plt

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

def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



