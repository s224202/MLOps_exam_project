from mlops_exam_project.model import WineQualityClassifier
import torch


def test_wine_quality_classifier():
    """Test that the model can handle a forward pass with dummy data."""
    model = WineQualityClassifier(
        input_dim=11, hidden_dims=[64, 32, 16], output_dim=10, dropout_rate=0.3
    )
    x = torch.randn(8, 11)  # Batch of 8 samples
    output = model(x)
    assert output.shape == (8, 10)
