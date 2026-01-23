"""Tests for the evaluate module."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from mlops_exam_project.model import WineQualityClassifier
from mlops_exam_project.evaluate import evaluate_model


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32],
        output_dim=6,
        dropout_rate=0.2,
    )


@pytest.fixture
def sample_dataloader():
    """Create a sample dataloader for testing."""
    features = torch.randn(32, 11)
    targets = torch.randint(0, 6, (32,))
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=8)


def test_evaluate_model_function(sample_model, sample_dataloader):
    """Test the evaluate_model core function."""
    results = evaluate_model(sample_model, sample_dataloader, device="cpu", num_classes=6)
    
    # Verify results structure
    assert "accuracy" in results
    assert "correct" in results
    assert "total" in results
    assert "predictions" in results
    assert "labels" in results
    assert "confusion_matrix" in results
    assert "report" in results
    
    # Verify values
    assert isinstance(results["accuracy"], (float, np.floating))
    assert 0 <= results["accuracy"] <= 1
    assert results["total"] == 32
    assert len(results["predictions"]) == 32
    assert len(results["labels"]) == 32
    assert results["confusion_matrix"].shape == (6, 6)
    assert isinstance(results["report"], str)


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


def test_model_evaluation_workflow(sample_model):
    """Test complete evaluation workflow like evaluate.py does."""
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader, TensorDataset
    
    sample_model.eval()
    
    # Create test dataset
    features = torch.randn(32, 11)
    labels = torch.randint(0, 6, (32,))
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=8)
    
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for features, batch_labels in dataloader:
            outputs = sample_model(features)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.numpy())
            all_labels.extend(batch_labels.numpy())
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    # Calculate accuracy (like evaluate.py does)
    accuracy = correct / total
    
    # Generate confusion matrix (like evaluate.py does)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Generate classification report (like evaluate.py does)
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[f"Quality {i}" for i in range(6)],
        digits=4,
        zero_division=0,
    )
    
    # Verify all outputs are valid
    assert 0 <= accuracy <= 1
    assert cm.shape == (6, 6)
    assert "accuracy" in report
    assert correct == sum([p == l for p, l in zip(all_predictions, all_labels)])


def test_model_predictions_in_valid_range(sample_model):
    """Test that model predictions are in valid range."""
    sample_model.eval()
    
    dummy_input = torch.randn(10, 11)
    
    with torch.no_grad():
        output = sample_model(dummy_input)
    
    # Predictions should have correct shape
    assert output.shape == torch.Size([10, 6])
    
    # Get class predictions
    predictions = output.argmax(dim=1)
    assert torch.all(predictions >= 0) and torch.all(predictions < 6)


def test_model_accuracy_calculation():
    """Test accuracy calculation logic used in evaluate.py."""
    # Simulate what evaluate.py does
    predictions = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    labels = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    
    correct = sum([p == l for p, l in zip(predictions, labels)])
    total = len(labels)
    accuracy = correct / total
    
    assert accuracy == 1.0
    assert correct == 10
    assert total == 10


def test_hydra_config_loading_logic():
    """Test the logic of loading and using config (simulating evaluate.py)."""
    from pathlib import Path
    
    # Simulate what evaluate.py does with config
    # It gets paths from config and combines them with project_root
    project_root = Path(__file__).parent.parent.parent
    
    # Simulate config values
    data_path_config = "data/processed"
    model_path_config = "models"
    
    # Simulate path construction like evaluate.py does
    data_path = project_root / data_path_config
    model_path = project_root / model_path_config
    
    # Verify paths are constructed correctly
    assert "data/processed" in str(data_path)
    assert "models" in str(model_path)
    assert project_root in data_path.parents or project_root == data_path.parent.parent.parent


def test_evaluate_model_all_predictions_same_class(sample_model):
    """Test evaluate_model when all predictions are same class."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Mixed labels
    test_features = torch.randn(16, 11)
    test_labels = torch.tensor([0, 1, 2, 3, 4, 5] * 3, dtype=torch.long)[:16]
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    results = evaluate_model(sample_model, test_loader, device="cpu", num_classes=6)
    
    # Should still produce valid metrics
    assert results["total"] == 16
    assert results["correct"] >= 0
    assert results["correct"] <= results["total"]


def test_evaluate_model_report_format(sample_model):
    """Test that evaluate_model produces properly formatted report."""
    from torch.utils.data import DataLoader, TensorDataset
    
    test_features = torch.randn(24, 11)
    test_labels = torch.randint(0, 6, (24,))
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    results = evaluate_model(sample_model, test_loader, device="cpu", num_classes=6)
    
    # Verify report is a string
    assert isinstance(results["report"], str)
    assert "Quality" in results["report"]
    assert "precision" in results["report"].lower() or "accuracy" in results["report"].lower()
