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
        
        # Ray encoder
        self.pos_enc = PositionalEncoding(num_freqs=num_freqs) 
        
        # Calculate dimensions dynamically
        # 3 coords + (num_freqs * 2 * 3) encoded = total per vector
        ray_feat_dim = 3 + num_freqs * 2 * 3
        total_ray_dim = ray_feat_dim * 2  # rays_o + rays_d
        
        self.ray_encoder = nn.Sequential(
            nn.Linear(total_ray_dim, 128),
            nn.ReLU(),
            nn.Linear(128, token_dim)
        )
        
        # Final MLP to predict RGB
        # CRITICAL FIX: Input dim is token_dim * 2 because we concat(query, attended)
        self.rgb_head = nn.Sequential(
            nn.Linear(token_dim * 2, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, object_code, rays_o, rays_d):
        # Ensure tokens have sequence dimension
        if object_code.dim() == 2:
            tokens = object_code.unsqueeze(1)
        else:
            tokens = object_code
        
        # 1. Refine tokens
        refined_tokens = self.transformer(tokens)
        
        # 2. Encode Ray
        ray_emb = torch.cat([self.pos_enc(rays_o), self.pos_enc(rays_d)], dim=-1)
        query = self.ray_encoder(ray_emb)       # [Batch, Dim]
        query_seq = query.unsqueeze(1)          # [Batch, 1, Dim]
        
        # 3. Attention (Ray attends to Object)
        attn_scores = torch.bmm(query_seq, refined_tokens.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (self.token_dim**0.5), dim=-1)
        attended = torch.bmm(attn_weights, refined_tokens).squeeze(1) # [Batch, Dim]
        
        # 4. CRITICAL FIX: Skip Connection
        # Concatenate the Ray Query with the Object Feature
        # This ensures the MLP sees the ray direction even if attention collapses
        combined_features = torch.cat([query, attended], dim=-1) # [Batch, Dim * 2]
        
        rgb = self.rgb_head(combined_features)
        
        return rgb