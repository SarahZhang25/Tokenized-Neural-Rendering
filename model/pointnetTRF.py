import torch
import torch.nn as nn
from model.pointnet import PointNet
from model.transformerRF import TransformerRF

class PointNetRF(nn.Module):
    def __init__(self, token_dim=256, num_freqs=6):
        super().__init__()
        
        # 1. Encoder: PointNet
        # We use the official PointNet implementation but extract local features
        self.pointnet = PointNet(classes=10) # Classes doesn't matter, we won't use the head
        
        # Projection layer to match PointNet output (1024) to Transformer dim (256)
        self.feature_proj = nn.Linear(1024, token_dim)
        
        # 2. Decoder: TransformerRF
        self.renderer = TransformerRF(token_dim=token_dim, num_layers=2, nhead=4, num_freqs=num_freqs)

    def forward(self, points, rays_o, rays_d):
        """
        Args:
            points: [Batch, Num_Points, 3] (XYZ coordinates)
            rays_o: [Batch, 3]
            rays_d: [Batch, 3]
        """
        # 1. Encode Points
        # PointNet expects [Batch, 3, Num_Points]
        points_transposed = points.transpose(1, 2)
        
        # Get local features: [Batch, Num_Points, 1024]
        local_features = self.pointnet(points_transposed, return_local_features=True)
        
        # Project to token dimension: [Batch, Num_Points, 256]
        object_tokens = self.feature_proj(local_features)
        
        # 2. Render
        # The renderer uses Cross-Attention between Rays (Query) and Object Tokens (Key/Value)
        rgb = self.renderer(object_tokens, rays_o, rays_d)
        
        return rgb