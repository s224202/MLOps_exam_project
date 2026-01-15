from mlops_exam_project.data import WineData
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


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
            mean = processed_data[column].mean()
            std = processed_data[column].std()
            print(f"Column: {column}, Mean: {mean}, Std: {std}")
            assert abs(mean) < 1e-6  # Mean should be approximately 0
            assert abs(std - 1) < 1e-6  # Std should be approximately 1
