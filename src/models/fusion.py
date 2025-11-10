"""
this layer combnes features from Vision Transformer and CodeBERT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossModalAttention(nn.Module):
    """
    Allows image features to attend to text features and vice versa
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-modal attention
        """
        batch_size, seq_len_q, embed_dim = query.size()
        seq_len_k = key.size(1)
        
        # Project to Q, K, V
        q = self.q_proj(query)  # [B, seq_len_q, embed_dim]
        k = self.k_proj(key)    # [B, seq_len_k, embed_dim]
        v = self.v_proj(value)  # [B, seq_len_v, embed_dim]
        
        # Reshape for mukti head
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [B, num_heads, seq_len_q, head_dim]
        k = k.transpose(1, 2)   # [B, num_heads, seq_len_k, head_dim]
        v = v.transpose(1, 2)   # [B, num_heads, seq_len_k, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, num_heads, seq_len_q, head_dim]

        out = out.transpose(1, 2).reshape(batch_size, seq_len_q, embed_dim)

        out = self.out_proj(out)
        
        return out


class MultimodalFusionLayer(nn.Module):
    """
    Fusion layer that combines image and text features
    
    Uses cross-modal attention and residual connections
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim

        self.img_to_text_attn = CrossModalAttention(embed_dim, num_heads)
        self.text_to_img_attn = CrossModalAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        img_features: torch.Tensor,
        diff_features: torch.Tensor,
        context_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse image and text features
        
        """
        img_seq = img_features.unsqueeze(1)  # [B, 1, embed_dim]
        text_seq = torch.stack([diff_features, context_features], dim=1)  # [B, 2, embed_dim]
        
        img_attended = self.img_to_text_attn(img_seq, text_seq, text_seq)  # [B, 1, embed_dim]
        img_attended = img_attended.squeeze(1)  # [B, embed_dim]

        text_attended = self.text_to_img_attn(text_seq, img_seq, img_seq)  # [B, 2, embed_dim]
        text_attended = text_attended.mean(dim=1)  # [B, embed_dim]

        img_enhanced = img_features + self.ffn1(self.norm1(img_attended))
        text_enhanced = (diff_features + context_features) / 2 + self.ffn2(self.norm2(text_attended))
        
        # Combine features
        combined = torch.cat([img_enhanced, text_enhanced], dim=-1)  # [B, embed_dim * 2]
        fused_features = self.combine(combined)  # [B, embed_dim]
        
        return fused_features

