"""
Tests for Vision Transformer component
"""

import pytest
import torch

from src.models.vision_transformer import VisionTransformer, PatchEmbedding, MultiHeadAttention, TransformerBlock


def test_patch_embedding():
    """Test patch embedding layer"""
    patch_embed = PatchEmbedding(img_size=(224, 224), patch_size=16)
    
    # Create dummy image
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = patch_embed(x)
    
    # Check output shape
    expected_patches = (224 // 16) * (224 // 16)  # 196 patches
    assert output.shape == (2, expected_patches + 1, 768)  # +1 for CLS token


def test_multi_head_attention():
    """Test multi-head attention"""
    attn = MultiHeadAttention(embed_dim=768, num_heads=12)
    
    # Create dummy input
    x = torch.randn(2, 197, 768)  # batch_size, seq_len, embed_dim
    
    # Forward pass
    output = attn(x)
    
    # Check output shape
    assert output.shape == x.shape


def test_transformer_block():
    """Test transformer block"""
    block = TransformerBlock(embed_dim=768, num_heads=12)
    
    # Create dummy input
    x = torch.randn(2, 197, 768)
    
    # Forward pass
    output = block(x)
    
    # Check output shape
    assert output.shape == x.shape


def test_vision_transformer():
    """Test complete Vision Transformer"""
    vit = VisionTransformer(
        img_size=(224, 224),
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=6
    )
    
    # Create dummy diff image
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = vit(x)
    
    # Check output shape (should be CLS token representation)
    assert output.shape == (2, 768)


def test_vision_transformer_gradients():
    """Test that gradients flow properly"""
    vit = VisionTransformer()
    
    # Create dummy input
    x = torch.randn(1, 3, 224, 224)
    x.requires_grad_(True)
    
    # Forward pass
    output = vit(x)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
