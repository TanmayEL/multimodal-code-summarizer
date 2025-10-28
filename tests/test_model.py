"""
Tests for the model architecture
"""

import pytest
import torch

from src.models.architecture import MultimodalCodeReviewModel


def test_model_initialization():
    """Test that model initializes correctly"""
    model = MultimodalCodeReviewModel()
    
    # Check that model has required attributes
    assert hasattr(model, 'vocab_size')
    assert hasattr(model, 'hidden_dim')
    assert hasattr(model, 'num_heads')
    assert hasattr(model, 'num_layers')
    
    # Check default values
    assert model.vocab_size == 10000
    assert model.hidden_dim == 768
    assert model.num_heads == 12
    assert model.num_layers == 6


def test_model_forward():
    """Test model forward pass"""
    model = MultimodalCodeReviewModel()
    
    # Create dummy inputs
    batch_size = 2
    diff_images = torch.randn(batch_size, 3, 224, 224)
    diff_text = torch.randint(0, 1000, (batch_size, 100))
    context = torch.randint(0, 1000, (batch_size, 50))
    
    # Forward pass
    output = model(diff_images, diff_text, context)
    
    # Check output shape
    assert output.shape == (batch_size, model.vocab_size)
    assert isinstance(output, torch.Tensor)


def test_model_generate_summary():
    """Test summary generation"""
    model = MultimodalCodeReviewModel()
    
    # Create dummy inputs
    diff_image = torch.randn(3, 224, 224)
    diff_text = torch.randint(0, 1000, (100,))
    context = torch.randint(0, 1000, (50,))
    
    # Generate summary
    summary = model.generate_summary(diff_image, diff_text, context)
    
    # Check output
    assert isinstance(summary, str)
    assert len(summary) > 0
