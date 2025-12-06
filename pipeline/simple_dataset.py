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
        
        # Raw Shapes: [N_images, H, W, 3]
        _rgb = data['rgb'] 
        _rays_o = data['rays_o']
        _rays_d = data['rays_d']
        
        # 2. Preprocess: Normalize and Flatten
        # We flatten all pixels from all images into one long list of rays.
        # This lets us shuffle pixels globally (better for training stability).
        
        # RGB: Normalize 0-255 -> 0.0-1.0
        # Shape: [N_total_pixels, 3]
        self.rgb = torch.from_numpy(_rgb.reshape(-1, 3)).float() / 255.0
        
        # Rays: Shape [N_total_pixels, 3]
        self.rays_o = torch.from_numpy(_rays_o.reshape(-1, 3)).float()
        self.rays_d = torch.from_numpy(_rays_d.reshape(-1, 3)).float()
        
        # Optional: Move entire dataset to GPU (If it fits in VRAM)
        # 50 images * 128*128 * 3 floats is tiny (~30MB), so this is safe.
        if device == 'cuda' and torch.cuda.is_available():
            self.rgb = self.rgb.to(device)
            self.rays_o = self.rays_o.to(device)
            self.rays_d = self.rays_d.to(device)
            print(f"Loaded {len(self.rgb)} rays to GPU memory.")
        else:
            print(f"Loaded {len(self.rgb)} rays to CPU memory.")

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        """
        Returns a single ray-pixel pair.
        """
        return {
            'rays_o': self.rays_o[idx],  # [3]
            'rays_d': self.rays_d[idx],  # [3] - This is your w_out (inverted)
            'rgb':    self.rgb[idx]      # [3] - Ground Truth Color
        }