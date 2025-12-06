"""
TriangleEncoder adapted from https://github.com/microsoft/renderformer/blob/main/renderformer/models/renderformer.py
"""

import torch
from torch import nn
from torch.amp import autocast

from model.utils import NeRFEncoding
from model.renderer import TransformerRF

# from renderformer.encodings.nerf_encoding import NeRFEncoding
# from renderformer.layers.attention import TransformerEncoder
# from renderformer.models.view_transformer import ViewTransformer
# from renderformer.models.config import RenderFormerConfig


class TriangleEncoder(nn.Module):
    def __init__(
            self, 
            pe_type: str = 'nerf',
            use_vn_encoder: bool = True,
            vertex_pe_num_freqs: int = 6,
            vn_pe_num_freqs: int = 6,
            latent_dim: int = 128,
            texture_channels: int = 3,
            texture_encode_patch_size: int = 4,
            num_register_tokens: int = 2,
            texture_encoder_norm_type: str = 'layer_norm',
            vn_encoder_norm_type: str = 'layer_norm'
    ):
        super(TriangleEncoder, self).__init__()
        self.pe_type = pe_type
        self.use_vn_encoder = use_vn_encoder
        self.vertex_pe_num_freqs = vertex_pe_num_freqs
        self.vn_pe_num_freqs = vn_pe_num_freqs
        self.latent_dim = latent_dim
        self.texture_channels = texture_channels
        self.texture_encode_patch_size = texture_encode_patch_size
        self.num_register_tokens = num_register_tokens
        self.texture_encoder_norm_type = texture_encoder_norm_type
        self.vn_encoder_norm_type = vn_encoder_norm_type

        if pe_type == 'nerf':
            # vertex PE and projections
            self.tri_vpos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=vertex_pe_num_freqs,
                include_input=True
            )
            # triangle pe projection
            self.tri_encoding_proj = nn.Linear(
                self.tri_vpos_pe.get_out_dim(),
                latent_dim
            )
            # reuse this config for ablation...
            if vn_encoder_norm_type == 'layer_norm':
                self.tri_encoding_norm = nn.LayerNorm(latent_dim)
            elif vn_encoder_norm_type == 'rms_norm':
                self.tri_encoding_norm = nn.RMSNorm(latent_dim)
            elif vn_encoder_norm_type == 'none':
                self.tri_encoding_norm = nn.Identity()
            self.rope_dim = None
        elif pe_type == 'rope':
            self.rope_dim = vertex_pe_num_freqs
        else:
            raise ValueError(f"Invalid positional encoding type: {pe_type}")

        if use_vn_encoder:
            self.vn_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=vn_pe_num_freqs,
                include_input=True
            )
            self.vn_encoding_proj = nn.Linear(
                self.vn_pe.get_out_dim(),
                latent_dim
            )
            if vn_encoder_norm_type == 'layer_norm':
                self.vn_encoder_norm = nn.LayerNorm(latent_dim)
            elif vn_encoder_norm_type == 'rms_norm':
                self.vn_encoder_norm = nn.RMSNorm(latent_dim)
            elif vn_encoder_norm_type == 'none':
                self.vn_encoder_norm = nn.Identity()
            else:
                raise ValueError(f"Invalid vertex normal encoder normalization type: {vn_encoder_norm_type}")

        # texture encoder
        self.texture_encoder = nn.Linear(
            texture_channels * texture_encode_patch_size * texture_encode_patch_size,
            latent_dim
        )
        if texture_encoder_norm_type == 'layer_norm':
            self.texture_encoder_norm = nn.LayerNorm(latent_dim)
        elif texture_encoder_norm_type == 'rms_norm':
            self.texture_encoder_norm = nn.RMSNorm(latent_dim)
        else:
            raise ValueError(f"Invalid texture encoder normalization type: {texture_encoder_norm_type}")

        # learnable tokens
        self.tri_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, num_register_tokens, latent_dim))
        self.skip_token_num = num_register_tokens

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    @autocast(device_type="cuda", dtype=torch.float32)  # avoid bf16 for its low precision
    def process_tri_vpos_list(self, tri_vpos_list, valid_mask):
        """
        Process tri_vpos_list for RoPE positional encoding.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param valid_mask: [batch_size, max_num_tri]
        :return: processed tri_vpos_list, updated valid_mask
        """
        mask_weight = (valid_mask.float() / (valid_mask.sum(dim=1, keepdim=True) + 1e-5))[..., None]
        weighted_tri_pos = mask_weight * tri_vpos_list
        center_pos = weighted_tri_pos.sum(dim=1).reshape(-1, 3, 3).mean(dim=1, keepdim=True).repeat(1, self.skip_token_num, 3)
        tri_vpos_list = torch.cat([center_pos, tri_vpos_list], dim=1)

        # construct valid mask, things you want is True
        valid_mask = torch.cat([
            torch.ones((tri_vpos_list.size(0), self.skip_token_num), dtype=torch.bool, device=valid_mask.device),
            valid_mask
        ], dim=1)

        return tri_vpos_list, valid_mask

    def forward(self, tri_vpos_list, texture_patch_list, valid_mask, vns):
        """
        From input triangle list + texture patches, construct the sequence for transformer.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        :param valid_mask: [batch_size, max_num_tri]
        :param vns: [batch_size, max_num_tri, 3, 3], padded
        :return: [batch_size, max_num_tri + 2, latent_dim]
        """
        batch_size = tri_vpos_list.size(0)

        # vertex normal encoding
        if self.use_vn_encoder:
            vn_emb = self.vn_encoder_norm(self.vn_encoding_proj(self.vn_pe(vns)))
        else:
            vn_emb = 0.

        # texture encoding
        tri_tex_emb = self.texture_encoder_norm(self.texture_encoder(
            texture_patch_list.reshape(texture_patch_list.size(0), texture_patch_list.size(1), -1)
        ))

        # construct sequence
        tokens = []
        tokens.append(self.reg_tokens.expand(batch_size, -1, -1))

        if self.pe_type == 'nerf':
            tri_vpos_pe = self.tri_vpos_pe(tri_vpos_list)
            tri_emb = self.tri_encoding_norm(self.tri_encoding_proj(tri_vpos_pe)) + self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        elif self.pe_type == 'rope':
            tri_emb = self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        else:
            raise ValueError(f"Invalid positional encoding type: {self.pe_type}")

        seq = torch.cat(tokens, dim=1)

        # pad triangle pos (for RoPE) and valid mask (for all)
        # use center pos for RoPE on auxiliary tokens
        tri_vpos_list, valid_mask = self.process_tri_vpos_list(tri_vpos_list, valid_mask)

        return seq, valid_mask, tri_vpos_list

class TriangleRenderer(nn.Module):
    def __init__(
            self, 
            token_dim=128,
            num_layers=4,
            pe_type: str = 'nerf',
            use_vn_encoder: bool = True,
            vertex_pe_num_freqs: int = 6,
            vn_pe_num_freqs: int = 6,
            latent_dim: int = 128,
            texture_channels: int = 3,
            texture_encode_patch_size: int = 4,
            num_register_tokens: int = 2,
            texture_encoder_norm_type: str = 'layer_norm',
            vn_encoder_norm_type: str = 'layer_norm'
        ):
        super(TriangleRenderer, self).__init__()
        
        # Tokenizer
        self.tokenizer = TriangleEncoder(
            pe_type=pe_type,
            use_vn_encoder=use_vn_encoder,
            vertex_pe_num_freqs=vertex_pe_num_freqs,
            vn_pe_num_freqs=vn_pe_num_freqs,
            latent_dim=latent_dim,
            texture_channels=texture_channels,
            texture_encode_patch_size=texture_encode_patch_size,
            num_register_tokens=num_register_tokens,
            texture_encoder_norm_type=texture_encoder_norm_type,
            vn_encoder_norm_type=vn_encoder_norm_type
        )

        # Rest of Model
        self.renderer = TransformerRF(
            token_dim=token_dim, 
            num_layers=num_layers
        )

    def forward(self, tri_vpos_list, texture_patch_list, valid_mask, vns, w_out):
        """
        Full forward pass from triangle inputs to rendered RGB.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        :param valid_mask: [batch_size, max_num_tri]
        :param vns: [batch_size, max_num_tri, 3, 3], padded
        :param w_out: [batch_size, 3]
        :return: rendered RGB: [batch_size, 3]
        """
        # Tokenization
        tokens, valid_mask, tri_vpos_list = self.tokenizer(
            tri_vpos_list,
            texture_patch_list,
            valid_mask,
            vns
        )

        # Rendering
        rgb = self.renderer(tokens, w_out)

        return rgb