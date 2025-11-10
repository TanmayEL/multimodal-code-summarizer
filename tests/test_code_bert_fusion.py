"""
Tests for CodeBERT and fusion components
"""

import pytest
import torch

from src.models.code_bert import CodeBERT, TokenEmbedding, CodeAttention, CodeTransformerBlock
from src.models.fusion import MultimodalFusionLayer, CrossModalAttention


def test_token_embedding():
    """Test token embedding layer"""
    embed = TokenEmbedding(vocab_size=1000, embed_dim=768, max_seq_len=512)

    token_ids = torch.randint(0, 1000, (2, 100))
    segment_ids = torch.randint(0, 2, (2, 100))
    
    # Forward pass
    output = embed(token_ids, segment_ids)
    
    assert output.shape == (2, 100, 768)


def test_code_attention():
    """Test code attention mechanism"""
    attn = CodeAttention(embed_dim=768, num_heads=12)
    
    x = torch.randn(2, 100, 768)
    mask = torch.ones(2, 100, 100)

    output = attn(x, mask)
    assert output.shape == x.shape


def test_code_transformer_block():
    """Test code transformer block"""
    block = CodeTransformerBlock(embed_dim=768, num_heads=12)
    x = torch.randn(2, 100, 768)
    output = block(x)
    assert output.shape == x.shape


def test_code_bert():
    """complete CodeBERT"""
    bert = CodeBERT(vocab_size=1000, embed_dim=768, num_heads=12, num_layers=6)
    diff_text = torch.randint(0, 1000, (2, 50))
    context = torch.randint(0, 1000, (2, 30))
    diff_features, context_features = bert(diff_text, context)
    assert diff_features.shape == (2, 768)
    assert context_features.shape == (2, 768)


def test_cross_modal_attention():
    attn = CrossModalAttention(embed_dim=768, num_heads=12)
    
    query = torch.randn(2, 10, 768)
    key = torch.randn(2, 20, 768)
    value = torch.randn(2, 20, 768)
    
    output = attn(query, key, value)
    assert output.shape == query.shape


def test_multimodal_fusion():
    """Test multimodal fusion layer"""
    fusion = MultimodalFusionLayer(embed_dim=768, num_heads=12)
    img_features = torch.randn(2, 768)
    diff_features = torch.randn(2, 768)
    context_features = torch.randn(2, 768)
    output = fusion(img_features, diff_features, context_features)
    
    assert output.shape == (2, 768)


def test_fusion_gradients():
    """Test that gradients flow through fusion layer"""
    fusion = MultimodalFusionLayer()
    
    img_features = torch.randn(1, 768, requires_grad=True)
    diff_features = torch.randn(1, 768, requires_grad=True)
    context_features = torch.randn(1, 768, requires_grad=True)
    
    # Forward pass
    output = fusion(img_features, diff_features, context_features)

    loss = output.sum()
    loss.backward()
    assert img_features.grad is not None
    assert diff_features.grad is not None
    assert context_features.grad is not None

