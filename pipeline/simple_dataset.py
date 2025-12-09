import torch
from torch.utils.data import Dataset
import numpy as np

class SingleObjectDataset(Dataset):
    def __init__(self, npz_path='experiment_data.npz', device='cpu'):
        """
        Loads the generated single-object dataset.
        
        Args:
            npz_path: Path to the .npz file created by create_dataset()
            device: 'cpu' or 'cuda'. 
                    Note: For small datasets, loading all data to GPU ('cuda') 
                    at start is faster than moving batches during training.
        """
        super().__init__()
        
        # 1. Load Data
        data = np.load(npz_path)
                
        # Load and normalize RGB to [0, 1]
        rgb = data['rgb'].astype(np.float32) / 255.0  # CRITICAL FIX
        rays_o = data['rays_o'].astype(np.float32)
        rays_d = data['rays_d'].astype(np.float32)
        
        # Reshape from (20, 128, 128, 3) to (20*128*128, 3)
        n_views, H, W, _ = rgb.shape
        self.rgb_pixels = torch.from_numpy(rgb.reshape(-1, 3)).to(device)
        self.rays_o = torch.from_numpy(rays_o.reshape(-1, 3)).to(device)
        self.rays_d = torch.from_numpy(rays_d.reshape(-1, 3)).to(device)
        self.points = torch.from_numpy(data['object_points']).to(device)
        
        # Compute per-pixel weights (higher weight for non-black pixels)
        is_foreground = (self.rgb_pixels.sum(dim=1) > 0.01).float()
        n_fg = is_foreground.sum()
        n_bg = len(is_foreground) - n_fg
        
        # Balance weights so foreground and background contribute equally
        self.weights = torch.ones_like(is_foreground)
        if n_fg > 0 and n_bg > 0:
            self.weights[is_foreground == 1] = n_bg / n_fg  # Upweight foreground
            # Background keeps weight 1.0
        
        print(f"Dataset loaded: {len(self)} rays")
        print(f"RGB range: [{self.rgb_pixels.min():.3f}, {self.rgb_pixels.max():.3f}]")
        print(f"RGB mean: {self.rgb_pixels.mean():.3f}")
        print(f"Foreground pixels: {n_fg.item()} ({100*n_fg/len(self):.1f}%)")
        print(f"Foreground weight: {self.weights[is_foreground == 1][0].item():.2f}x")

    def __len__(self):
        return len(self.rgb_pixels)

    def __getitem__(self, idx):
        return {
            'rays_o': self.rays_o[idx],
            'rays_d': self.rays_d[idx],
            'rgb': self.rgb_pixels[idx],
            'weight': self.weights[idx]
        }

    def get_points(self):
        return self.points
    