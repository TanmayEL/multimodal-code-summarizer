import pytest
import torch

from src.models.architecture import MultimodalCodeReviewModel


def test_model_initialization():
    model = MultimodalCodeReviewModel()
    
    assert hasattr(model, 'vocab_size')
    assert hasattr(model, 'hidden_dim')
    assert hasattr(model, 'num_heads')
    assert hasattr(model, 'num_layers')
    
    assert model.vocab_size == 10000
    assert model.hidden_dim == 768
    assert model.num_heads == 12
    assert model.num_layers == 6


def test_model_forward():
    """Test model forward pass"""
    model = MultimodalCodeReviewModel()
    
    batch_size = 2
    diff_images = torch.randn(batch_size, 3, 224, 224)
    diff_text = torch.randint(0, 1000, (batch_size, 100))
    context = torch.randint(0, 1000, (batch_size, 50))

    output = model(diff_images, diff_text, context)

    assert output.shape == (batch_size, model.vocab_size)
    assert isinstance(output, torch.Tensor)


def test_model_generate_summary():
    """Test summary generation"""
    model = MultimodalCodeReviewModel()

    diff_image = torch.randn(3, 224, 224)
    diff_text = torch.randint(0, 1000, (100,))
    context = torch.randint(0, 1000, (50,))

    summary = model.generate_summary(diff_image, diff_text, context)

    assert isinstance(summary, str)
    assert len(summary) > 0


def test_model_gradients():
    """Test that gradients flow through the model"""
    model = MultimodalCodeReviewModel()

    diff_images = torch.randn(1, 3, 224, 224, requires_grad=True)
    diff_text = torch.randint(0, 1000, (1, 100))
    context = torch.randint(0, 1000, (1, 50))

    output = model(diff_images, diff_text, context)
    
    loss = output.sum()
    loss.backward()
    
    assert diff_images.grad is not None
    assert diff_images.grad.shape == diff_images.shape
