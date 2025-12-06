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
    def __init__(self, token_dim=128, num_layers=3, nhead=4):
        super().__init__()
        
        # Transformer Encoder
        # Allows tokens to share global context BEFORE rendering
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, 
            nhead=nhead, 
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True # Critical for training stability!
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention Pooling
        # Instead of MaxPool, we let the Ray "attend" to the most relevant tokens
        # A simplified dot-product attention
        self.attention_query = nn.Linear(39, token_dim) # Maps encoded ray dir to token space
        
        # Decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(token_dim + 39, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3), 
            nn.Sigmoid()
        )
        
        # Positional Encoding helper (from previous response)
        self.pos_enc = PositionalEncoding(num_freqs=6)

    def forward(self, tokens, w_out):
        """
        tokens: [Batch, N, Dim] (e.g., 500 triangle tokens OR 16 object slots)
        w_out:  [Batch, 3]      (Ray Directions)
        """
        batch_size = tokens.shape[0]
        
        # --- Stage 1: Reasoning (Transformer) ---
        # The tokens interact. Triangle A talks to Triangle B.
        # Shape: [Batch, N, Dim]
        refined_tokens = self.transformer(tokens)
        
        # --- Stage 2: Ray-Token Integration ---
        # Which tokens matter for THIS specific ray?
        
        # Encode ray direction: [Batch, 39]
        ray_emb = self.pos_enc(w_out) 
        
        # Project ray to query: [Batch, 1, Dim]
        query = self.attention_query(ray_emb).unsqueeze(1)
        
        # Calculate Attention Scores: (Query . Key)
        # [Batch, 1, Dim] x [Batch, Dim, N] -> [Batch, 1, N]
        attn_scores = torch.bmm(query, refined_tokens.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (128**0.5), dim=-1)
        
        # Weighted Sum of tokens: [Batch, 1, Dim]
        context_vector = torch.bmm(attn_weights, refined_tokens).squeeze(1)
        
        # --- Stage 3: Decoding ---
        # Concat Context + Ray Direction
        decoder_input = torch.cat([context_vector, ray_emb], dim=-1)
        rgb = self.decoder_mlp(decoder_input)
        
        return rgb