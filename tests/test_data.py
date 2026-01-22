from mlops_exam_project.data import WineData, preprocess
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import pytest
import tempfile
from unittest.mock import patch, MagicMock


def test_wine_data():
    """Test the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert isinstance(dataset.data, pd.DataFrame)


def test_wine_data_file_not_found():
    """Test that FileNotFoundError is raised when file doesn't exist and download=False."""
    with pytest.raises(FileNotFoundError):
        WineData(Path("nonexistent/path/file.csv"), download=False)


def test_preprocess_method(tmp_path=Path("data/processed")):
    """Test the preprocess method of the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    output_folder = tmp_path / "processed"
    dataset.preprocess(output_folder)
    processed_file = output_folder / "processed_wine_data.csv"
    assert processed_file.exists()
    processed_data = pd.read_csv(processed_file)
    assert not processed_data.empty
    for column in processed_data.columns:
        if column != "quality":
            mean = processed_data[column].mean()
            std = processed_data[column].std()
            print(f"Column: {column}, Mean: {mean}, Std: {std}")
            assert abs(mean) < 1e-6  # Mean should be approximately 0
            assert abs(std - 1) < 1e-6  # Std should be approximately 1


def test_getitem():
    """Test the __getitem__ method of the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    sample = dataset[0]
    assert isinstance(sample, pd.Series)
    assert len(sample) == len(dataset.data.columns)
    assert "quality" in sample.index


def test_wine_data_len():
    """Test the __len__ method of the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    assert len(dataset) == len(dataset.data)


def test_preprocess_cli(tmp_path=Path("data/processed")):
    """Test the preprocess CLI function."""
    data_path = Path("data/raw/WineQT.csv")
    output_folder = tmp_path / "processed_cli"
    preprocess(data_path, output_folder, download=True)
    processed_file = output_folder / "processed_wine_data.csv"
    assert processed_file.exists()


def test_wine_data_download_with_mock(tmp_path):
    """Test WineData download functionality with mocked fetch_ucirepo."""
    mock_data = MagicMock()
    mock_data.data.keys.return_value = ["features", "target", "original"]
    mock_data.data.original = pd.DataFrame({"color": ["red", "white", "red"]})
    mock_data.data.features = pd.DataFrame({"alcohol": [9.5, 11.2, 10.1]})
    mock_data.data.target = pd.Series([6, 5, 7])

    test_path = tmp_path / "test_data.csv"

    with patch("mlops_exam_project.data.fetch_ucirepo", return_value=mock_data):
        dataset = WineData(test_path, download=True)
        assert dataset.data_path == test_path
        assert test_path.exists()
