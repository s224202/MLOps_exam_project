"""Tests for the evaluate module."""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from mlops_exam_project.model import WineQualityClassifier


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32],
        output_dim=6,
        dropout_rate=0.2,
    )


def test_model_can_load_and_predict(sample_model):
    """Test that a model can be loaded and make predictions."""
    sample_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 11)
    
    with torch.no_grad():
        output = sample_model(dummy_input)
    
    assert output.shape == torch.Size([1, 6])
    assert isinstance(output, torch.Tensor)


def test_model_state_dict_save_load(sample_model, tmp_path):
    """Test saving and loading model state."""
    model_path = tmp_path / "test_model.pth"
    
    # Save model
    torch.save(sample_model.state_dict(), model_path)
    assert model_path.exists()
    
    # Load model into a new instance
    new_model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32],
        output_dim=6,
        dropout_rate=0.2,
    )
    new_model.load_state_dict(torch.load(model_path))
    
    # Verify models produce same output
    sample_model.eval()
    new_model.eval()
    
    dummy_input = torch.randn(1, 11)
    with torch.no_grad():
        output1 = sample_model(dummy_input)
        output2 = new_model(dummy_input)
    
    torch.testing.assert_close(output1, output2)


def test_evaluation_metrics_calculation():
    """Test that evaluation metrics can be calculated."""
    from sklearn.metrics import accuracy_score, classification_report
    
    # Create dummy predictions and labels
    predictions = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 1, 0])
    
    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    assert 0 <= accuracy <= 1
    assert accuracy == 0.75  # 6 out of 8 correct
    
    # Verify classification report can be generated
    report = classification_report(
        labels.numpy(),
        predictions.numpy(),
        output_dict=True,
        zero_division=0,
    )
    assert "accuracy" in report


def test_confusion_matrix_calculation():
    """Test confusion matrix calculation."""
    from sklearn.metrics import confusion_matrix
    
    predictions = torch.tensor([0, 1, 2, 3, 4, 5])
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    
    cm = confusion_matrix(labels.numpy(), predictions.numpy())
    
    # Should be identity matrix for perfect predictions
    assert cm.shape == (6, 6)
    assert torch.all(torch.diag(torch.tensor(cm)) == 1)


def test_batch_evaluation():
    """Test model evaluation on batches."""
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32],
        output_dim=6,
        dropout_rate=0.2,
    )
    model.eval()
    
    # Create dummy batch
    batch_size = 32
    dummy_input = torch.randn(batch_size, 11)
    dummy_labels = torch.randint(0, 6, (batch_size,))
    
    with torch.no_grad():
        outputs = model(dummy_input)
        predictions = outputs.argmax(dim=1)
    
    assert predictions.shape == (batch_size,)
    assert torch.all(predictions >= 0) and torch.all(predictions < 6)


def test_device_compatibility():
    """Test model works on different devices."""
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32],
        output_dim=6,
        dropout_rate=0.2,
    )
    
    # Test on CPU
    model = model.to("cpu")
    dummy_input = torch.randn(1, 11, device="cpu")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.device.type == "cpu"
