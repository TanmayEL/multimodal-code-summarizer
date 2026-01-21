import pytest
import torch

from src.models.vision_transformer import VisionTransformer, PatchEmbedding, MultiHeadAttention, TransformerBlock


def test_patch_embedding():
    """Test patch embedding layer"""
    patch_embed = PatchEmbedding(img_size=(224, 224), patch_size=16)
    
    #dummy image
    x = torch.randn(2, 3, 224, 224)
    
    output = patch_embed(x)
    
    #sshape
    expected_patches = (224 // 16) * (224 // 16)  # 196 patches
    assert output.shape == (2, expected_patches + 1, 768)  # +1 for CLS token


def test_multi_head_attention():
    """Test multi-head attention"""
    attn = MultiHeadAttention(embed_dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)  #batch_size, seq_len, embed_dim
    
    output = attn(x)
    assert output.shape == x.shape


def test_transformer_block():
    """Test transformer block"""
    block = TransformerBlock(embed_dim=768, num_heads=12)
    
    x = torch.randn(2, 197, 768)
    
    output = block(x)
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
    
    x = torch.randn(2, 3, 224, 224)
    
    output = vit(x)
    
    assert output.shape == (2, 768)


def test_vision_transformer_gradients():
    """Test that gradients flow properly"""
    vit = VisionTransformer()
    
    x = torch.randn(1, 3, 224, 224)
    x.requires_grad_(True)
    output = vit(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
