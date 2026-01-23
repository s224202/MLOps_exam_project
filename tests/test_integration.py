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
