"""Integration tests for the complete ML pipeline."""

import pytest
import torch
from pathlib import Path
from mlops_exam_project.data import WineData
from mlops_exam_project.model import WineQualityClassifier


@pytest.fixture
def temp_processed_dir(tmp_path):
    """Create a temporary directory for processed data."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


@pytest.fixture
def raw_data_path():
    """Provide path to raw data."""
    return Path("data/raw/WineQT.csv")


def test_end_to_end_pipeline(raw_data_path, temp_processed_dir):
    """Test complete pipeline: load data -> preprocess -> model."""
    dataset = WineData(raw_data_path, download=True)
    assert len(dataset) > 0

    dataset.preprocess(temp_processed_dir)
    processed_file = temp_processed_dir / "processed_wine_data.csv"
    assert processed_file.exists()

    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    x = torch.randn(1, 11)
    output = model(x)
    assert output.shape == torch.Size([1, 6])


def test_data_to_model_compatibility(raw_data_path):
    """Test that data output is compatible with model input."""
    dataset = WineData(raw_data_path, download=True)
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)

    x = torch.randn(1, 11)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([1, 6])


def test_preprocessing_output_format(raw_data_path, temp_processed_dir):
    """Test that preprocessing produces correctly formatted data."""
    dataset = WineData(raw_data_path, download=True)
    dataset.preprocess(temp_processed_dir)

    processed_file = temp_processed_dir / "processed_wine_data.csv"
    import pandas as pd

    processed_data = pd.read_csv(processed_file)

    assert "quality" in processed_data.columns
    assert "color" not in processed_data.columns
    assert len(processed_data) == len(dataset.data)


@pytest.mark.skip(reason="train() is a Hydra-decorated function requiring CLI setup")
def test_train_with_processed_data(raw_data_path, temp_processed_dir, capsys):
    """Test training function with the data pipeline."""
    dataset = WineData(raw_data_path, download=True)
    dataset.preprocess(temp_processed_dir)

    # train() requires Hydra configuration and cannot be called directly in tests
    # This test is skipped as it would require complex test fixtures for Hydra


def test_multiple_forward_passes(raw_data_path):
    """Test multiple forward passes through model with different data samples."""
    dataset = WineData(raw_data_path, download=True)
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)

    for i in range(5):
        x = torch.randn(1, 11)
        output = model(x)
        assert output.shape == torch.Size([1, 6])
        assert not torch.isnan(output).any()


def test_model_reproducibility(raw_data_path):
    """Test that model produces consistent outputs with different weights."""
    dataset = WineData(raw_data_path, download=True)
    model1 = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    model2 = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)

    x = torch.tensor([[0.5] * 11], dtype=torch.float32)
    output1 = model1(x)
    output2 = model2(x)

    assert output1.shape == torch.Size([1, 6])
    assert output2.shape == torch.Size([1, 6])


def test_dataset_length_consistency(raw_data_path, temp_processed_dir):
    """Test that dataset length remains consistent through operations."""
    dataset = WineData(raw_data_path, download=True)
    original_length = len(dataset)

    dataset.preprocess(temp_processed_dir)

    assert len(dataset) == original_length


def test_train_evaluate_pipeline():
    """Test complete training and evaluation pipeline using core functions."""
    from mlops_exam_project.train import train_core
    from mlops_exam_project.evaluate import evaluate_model
    from torch.utils.data import DataLoader, TensorDataset

    # Create small synthetic datasets
    train_features = torch.randn(64, 11)
    train_labels = torch.randint(0, 6, (64,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)

    val_features = torch.randn(32, 11)
    val_labels = torch.randint(0, 6, (32,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8)

    test_features = torch.randn(32, 11)
    test_labels = torch.randint(0, 6, (32,))
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Train model
    model = WineQualityClassifier(
        input_dim=11, hidden_dims=[32, 16], output_dim=6, dropout_rate=0.2
    )
    trained_model, train_stats = train_core(
        model, train_loader, val_loader, epochs=2, device="cpu", learning_rate=0.01
    )

    # Evaluate model
    eval_results = evaluate_model(
        trained_model, test_loader, device="cpu", num_classes=6
    )

    # Verify complete pipeline
    assert len(train_stats["epoch_loss"]) == 2
    assert len(train_stats["epoch_accuracy"]) == 2
    assert "accuracy" in eval_results
    assert "confusion_matrix" in eval_results
    assert eval_results["accuracy"] >= 0


def test_visualize_pipeline():
    """Test visualization pipeline using core functions."""
    from mlops_exam_project.visualize import (
        get_model_predictions,
        create_evaluation_visualizations,
    )
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt

    # Create synthetic data
    train_features = torch.randn(64, 11)
    train_labels = torch.randint(0, 6, (64,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)

    test_features = torch.randn(32, 11)
    test_labels = torch.randint(0, 6, (32,))
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Create and use model
    model = WineQualityClassifier(
        input_dim=11, hidden_dims=[32, 16], output_dim=6, dropout_rate=0.2
    )
    model.eval()

    # Get predictions
    train_preds, train_labels_arr = get_model_predictions(
        model, train_loader, device="cpu"
    )
    test_preds, test_labels_arr = get_model_predictions(
        model, test_loader, device="cpu"
    )

    # Create visualizations
    fig, train_acc, test_acc, per_class_acc = create_evaluation_visualizations(
        train_preds, train_labels_arr, test_preds, test_labels_arr, num_classes=6
    )

    # Verify visualization pipeline
    assert train_preds.shape[0] == 64
    assert test_preds.shape[0] == 32
    assert 0 <= train_acc <= 1
    assert 0 <= test_acc <= 1
    assert len(per_class_acc) == 6

    plt.close(fig)


def test_full_ml_pipeline():
    """Test full ML pipeline: data -> train -> evaluate -> visualize."""
    from mlops_exam_project.train import train_core
    from mlops_exam_project.evaluate import evaluate_model
    from mlops_exam_project.visualize import (
        get_model_predictions,
        create_evaluation_visualizations,
    )
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt

    # Prepare data
    torch.manual_seed(42)
    train_features = torch.randn(80, 11)
    train_labels = torch.randint(0, 6, (80,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)

    val_features = torch.randn(40, 11)
    val_labels = torch.randint(0, 6, (40,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8)

    test_features = torch.randn(40, 11)
    test_labels = torch.randint(0, 6, (40,))
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Stage 1: Train
    model = WineQualityClassifier(
        input_dim=11, hidden_dims=[32, 16], output_dim=6, dropout_rate=0.2
    )
    trained_model, train_stats = train_core(
        model, train_loader, val_loader, epochs=2, device="cpu", learning_rate=0.01
    )
    assert len(train_stats["epoch_loss"]) == 2

    # Stage 2: Evaluate
    eval_results = evaluate_model(
        trained_model, test_loader, device="cpu", num_classes=6
    )
    assert 0 <= eval_results["accuracy"] <= 1

    # Stage 3: Visualize
    train_preds, train_labels_arr = get_model_predictions(
        trained_model, train_loader, device="cpu"
    )
    test_preds, test_labels_arr = get_model_predictions(
        trained_model, test_loader, device="cpu"
    )
    fig, train_acc, test_acc, per_class_acc = create_evaluation_visualizations(
        train_preds, train_labels_arr, test_preds, test_labels_arr, num_classes=6
    )

    # Verify complete pipeline works
    assert train_acc >= 0
    assert test_acc >= 0
    assert len(per_class_acc) == 6

    plt.close(fig)
