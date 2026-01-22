from mlops_exam_project.data import WineData, preprocess, split_data
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
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
    assert "quality" in processed_data.columns


def test_getitem():
    """Test the __getitem__ method of the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert isinstance(sample[0], torch.Tensor)
    assert sample[0].shape[0] > 0  # Has at least one feature


def test_wine_data_len():
    """Test the __len__ method of the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    assert len(dataset) == len(dataset.data)


def test_preprocess_cli(tmp_path=Path("data/processed")):
    """Test that the preprocess function is callable."""
    from mlops_exam_project.data import preprocess as prep_func
    assert callable(prep_func)


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


def test_split_data_creates_splits(tmp_path):
    """Ensure split_data writes train/val/test CSVs for clean data."""
    data = pd.DataFrame(
        {
            "feature1": list(range(20)),
            "feature2": [x * 0.1 for x in range(20)],
            "quality": ([0, 1] * 10),
        }
    )
    output_dir = tmp_path / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data(data, output_dir, train_test_split_ratio=0.8, train_val_split_ratio=0.8)

    train_fp = output_dir / "train_data.csv"
    val_fp = output_dir / "val_data.csv"
    test_fp = output_dir / "test_data.csv"
    assert train_fp.exists()
    assert val_fp.exists()
    assert test_fp.exists()

    train = pd.read_csv(train_fp)
    val = pd.read_csv(val_fp)
    test = pd.read_csv(test_fp)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert len(train) + len(val) + len(test) == len(data)


def test_preprocess_invokes_split_data(tmp_path):
    """Verify preprocess calls split_data with provided parameters."""
    data_path = Path("data/raw/WineQT.csv")
    output_folder = tmp_path / "processed_cli"
    output_folder.mkdir(parents=True, exist_ok=True)

    with patch("mlops_exam_project.data.split_data") as split_mock:
        preprocess(
            data_path=data_path,
            output_folder=output_folder,
            download=False,
            train_test_split_ratio=0.8,
            train_val_split_ratio=0.9,
        )
        split_mock.assert_called_once()


def test_split_data_on_real_csv(tmp_path):
    """Run split_data on the real CSV after cleaning NaN quality rows."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    cleaned = dataset.data.dropna(subset=["quality"]).copy()

    if cleaned.empty:
        pytest.skip("Real CSV has no rows with quality; skipping stratified split test")

    output_dir = tmp_path / "real_splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data(cleaned, output_dir, train_test_split_ratio=0.8, train_val_split_ratio=0.9)

    train_fp = output_dir / "train_data.csv"
    val_fp = output_dir / "val_data.csv"
    test_fp = output_dir / "test_data.csv"
    assert train_fp.exists()
    assert val_fp.exists()
    assert test_fp.exists()

    train = pd.read_csv(train_fp)
    val = pd.read_csv(val_fp)
    test = pd.read_csv(test_fp)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert len(train) + len(val) + len(test) == len(cleaned)
