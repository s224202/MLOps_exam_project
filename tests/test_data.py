from mlops_exam_project.data import WineData
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch


def test_wine_data():
    """Test the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert isinstance(dataset.data, pd.DataFrame)


def test_preprocess(tmp_path=Path("data/processed")):
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
            assert abs(min(processed_data[column]) - 0.0) < 1e-5
            assert abs(max(processed_data[column]) - 1.0) < 1e-5


def test_getitem():
    """Test the __getitem__ method of the WineData class."""
    dataset = WineData(Path("data/raw/WineQT.csv"), download=True)
    sample = dataset[0]
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], torch.Tensor)
    assert len(sample[0]) == len(dataset.data.columns) - 1
    assert isinstance(sample[1].item(), float)
    assert 0 <= sample[1] <= 5
