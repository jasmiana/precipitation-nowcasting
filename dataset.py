import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MovingMNISTDataset(Dataset):
    """
    MovingMNIST dataset shape: (20, 10000, 64, 64)
    We split it into 10 frames input, 10 frames target.
    """
    def __init__(self, root='data/MovingMNIST', is_train=True, download=True):
        super(MovingMNISTDataset, self).__init__()
        self.root = root
        self.is_train = is_train
        
        file_path = os.path.join(root, 'mnist_test_seq.npy')
        
        if not os.path.exists(file_path):
            if download:
                os.makedirs(root, exist_ok=True)
                print("Downloading Moving MNIST dataset...")
                url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
                urllib.request.urlretrieve(url, file_path)
                print("Download complete.")
            else:
                raise FileNotFoundError(f"{file_path} not found.")
                
        # Load data: expected shape (20, 10000, 64, 64)
        self.data = np.load(file_path)
        
        # Split train/test (e.g. 9000 for training, 1000 for testing)
        if self.is_train:
            self.data = self.data[:, :9000, ...]
        else:
            self.data = self.data[:, 9000:, ...]
            
    def __len__(self):
        # self.data shape [20, N, 64, 64]
        return self.data.shape[1]

    def __getitem__(self, idx):
        # extract sequence of 20 frames
        seq = self.data[:, idx, ...] # shape: (20, 64, 64)
        
        # convert to tensor and normalize [0, 1]
        seq = torch.from_numpy(seq).float() / 255.0
        
        # Add channel dimension: (T, C, H, W) -> (20, 1, 64, 64)
        seq = seq.unsqueeze(1)
        
        # input: first 10 frames, target: last 10 frames
        x = seq[:10]  # (10, 1, 64, 64)
        y = seq[10:]  # (10, 1, 64, 64)
        
        return x, y

def get_dataloaders(batch_size=512, num_workers=4):
    train_set = MovingMNISTDataset(is_train=True)
    test_set = MovingMNISTDataset(is_train=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True,
                              prefetch_factor=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True,
                             prefetch_factor=2)
    
    return train_loader, test_loader

if __name__ == '__main__':
    # Dry run
    loader, _ = get_dataloaders(batch_size=2)
    x, y = next(iter(loader))
    print("Moving MNIST Loaded.")
    print("Input shape x:", x.shape) # (2, 10, 1, 64, 64)
    print("Target shape y:", y.shape) # (2, 10, 1, 64, 64)
