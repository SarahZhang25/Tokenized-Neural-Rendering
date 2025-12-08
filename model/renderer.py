"""
Transformer-Radiance-Field Module
Two main components:
- Transformer Encoder: learn contextual relationships between input tokens
- MLP Decoder: generate RGB values from token context and spatial coordinates
Architecture:
Tokens 
-> Transformer Encoder (self attention between tokens)
-> Projection Layer into context

Token Context, (x,y,z,theta,phi) -> MLP Decoder -> (r,g,b,)
Inputs:
"""
import torch
import torch.nn as nn
from model.utils import PositionalEncoding

class TransformerRF(nn.Module):
    def __init__(self, token_dim=128, num_layers=2, nhead=4, num_freqs=6):
        super().__init__()
        self.token_dim = token_dim
        
        # Transformer to refine tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Ray encoder: maps (rays_o, rays_d) -> query vector
        # Using positional encoding for better spatial reasoning
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs) 
        # PositionalEncoding returns the original input concatenated with encoded features,
        #  so it's 3 + 36 = 39 dimensions per ray for 6 number of frequencies.
        # With both rays_o and rays_d, you get 39 * 2 = 78 total dimensions.
        input_dim = (3 + num_freqs * 2 * 3) * 2  # 6 frequencies, 2 (sin+cos), 3 coords, 2 (o and d)
        self.ray_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # 72 from pos encoding + 6 original inputs
            nn.ReLU(),
            nn.Linear(128, token_dim)
        )
        
        # Final MLP to predict RGB
        self.rgb_head = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, object_code, rays_o, rays_d):
        """
        object_code: [Batch, Dim] or [Batch, N_tokens, Dim]
        rays_o: [Batch, 3] - ray origins
        rays_d: [Batch, 3] - ray directions
        Returns: [Batch, 3] - predicted RGB
        """
        # Ensure tokens have sequence dimension
        if object_code.dim() == 2:
            tokens = object_code.unsqueeze(1)  # [Batch, 1, Dim]
        else:
            tokens = object_code  # [Batch, N, Dim]
        
        # Refine tokens with transformer
        refined_tokens = self.transformer(tokens)  # [Batch, N, Dim]
        
        # Encode ray with positional encoding
        ray_emb = torch.cat([
            self.pos_enc(rays_o),
            self.pos_enc(rays_d)
        ], dim=-1)  # [Batch, 72]
        
        # Generate query from ray
        query = self.ray_encoder(ray_emb)  # [Batch, Dim]
        query = query.unsqueeze(1)  # [Batch, 1, Dim]
        
        # Attention: query attends to refined tokens
        attn_scores = torch.bmm(query, refined_tokens.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (self.token_dim**0.5), dim=-1)
        
        # Weighted sum of tokens
        attended = torch.bmm(attn_weights, refined_tokens).squeeze(1)  # [Batch, Dim]
        
        # Predict RGB
        rgb = self.rgb_head(attended)  # [Batch, 3]
        
        return rgb