"""
Vision Transformer component for processing diff images

This implements a simplified ViT specifically designed for our diff images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PatchEmbedding(nn.Module):
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Patch embeddings [B, num_patches + 1, embed_dim]
        """
        batch_size = x.size(0)
        
        x = self.projection(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        
        x = x + self.pos_embedding
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    
    This is the core of the transformer - allows patches to attend to each other
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention
        
        Args:
            x: Input embeddings [B, seq_len, embed_dim]
            
        Returns:
            Output embeddings [B, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        
        qkv = self.qkv(x)  # [B, seq_len, embed_dim * 3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)  # [B, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        out = self.proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention + MLP
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block
        
        Args:
            x: Input embeddings [B, seq_len, embed_dim]
            
        Returns:
            Output embeddings [B, seq_len, embed_dim]
        """
        x = x + self.attn(self.norm1(x))
        
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process diff image through Vision Transformer
        
        Args:
            x: Input diff image [B, C, H, W]
            
        Returns:
            CLS token representation [B, embed_dim]
        """ 
        x = self.patch_embed(x)  # [B, num_patches + 1, embed_dim]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x[:, 0]  # [B, embed_dim]
