from pathlib import Path
from torch.utils.data import Dataset
import torch

import pandas as pd
from ucimlrepo import fetch_ucirepo
import os
from sklearn.model_selection import train_test_split


class WineData(Dataset):
    """A custom dataset class for handling the wine quality data."""

    def __init__(self, data_path: Path, download: bool = False) -> None:
        self.data_path = data_path
        if not self.data_path.exists():
            if download:

                print(f"Data file not found at {self.data_path}. Downloading from UCI ML Repository...")

                wine_quality = fetch_ucirepo(id=186)
                print(wine_quality.data.keys())
                os.makedirs(self.data_path.parent, exist_ok=True)
                colors = wine_quality.data.original["color"].map({"red": 1, "white": 0})
                wine_quality.data.features["color"] = colors
                features = wine_quality.data.features.copy()
                print(wine_quality.data.targets)
                features["quality"] = wine_quality.data.targets
                df = pd.DataFrame(features)
                df = df[df["color"] == 1]
                df.to_csv(self.data_path, index=False)
                print(f"Data downloaded and saved to {self.data_path}")
            else:
                raise FileNotFoundError(
                    f"Data file not found at {self.data_path}. If you want to download it, run with '--download'"
                )
        self.data = pd.read_csv(self.data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        features = self.data.iloc[index].drop("quality").values.astype(float)
        target = self.data.iloc[index]["quality"]

        # Convert quality scores to classification labels (0-indexed)
        # Wine quality ranges from 3-8.  We convert to 0-5
        target = target - 3
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data to have zero mean and unit variance and save it to the output folder."""
        processed_data = self.data.copy()

        for column in processed_data.columns:
            if column != "quality":  # Since we don't want to scale the target variable
                mean = processed_data[column].mean()
                std = processed_data[column].std()
                processed_data[column] = (processed_data[column] - mean) / std

                # use min-max scaling instead
                min_val = processed_data[column].min()
                max_val = processed_data[column].max()
                processed_data[column] = (processed_data[column] - min_val) / (
                    max_val - min_val
                )

        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / "processed_wine_data.csv"
        processed_data.drop(columns=["color"], inplace=True)
        processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


def split_data(
    data,
    data_path,
    train_test_split_ratio: float = 0.8,
    train_val_split_ratio: float = 0.9,
) -> None:
    print("Splitting data into train, test, and validation sets...")
    print(f"Train/Test split ratio: {train_test_split_ratio}")
    print(f"Train/Validation split ratio: {train_val_split_ratio}")



    train_data, test_data = train_test_split(data, test_size=1-train_test_split_ratio, random_state=42, stratify=data["quality"])
    train_data, val_data = train_test_split(train_data, test_size=1-train_val_split_ratio, random_state=42, stratify=train_data["quality"]) 

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)


    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Validation data size: {len(val_data)}")

    print("Saving split data...")
    train_data.to_csv(data_path / "train_data.csv", index=False)
    test_data.to_csv(data_path / "test_data.csv", index=False) 
    val_data.to_csv(data_path / "val_data.csv", index=False)   

    print(f"Train, test, and validation data saved to {data_path}")
    print("Data splitting completed.")     



def preprocess(
    data_path: Path = typer.Option(Path("data/raw/Wqt.csv"), help="Path to raw data"),
    output_folder: Path = typer.Option(Path("data/processed/"), help="Output folder for processed data"),
    download: bool = typer.Option(True, help="Download data if not found"),
    train_test_split_ratio: float = typer.Option(0.8, help="Train/test split ratio"),
    train_val_split_ratio: float = typer.Option(0.9, help="Train/validation split ratio")
) -> None:
#def preprocess(data_path: Path = Path("data/raw/Wqt.csv"), output_folder: Path = Path("data/processed/"), download: bool = True, train_test_split_ratio: float = 0.8, train_val_split_ratio: float = 0.9) -> None:
# def preprocess(data_path: Path, output_folder: Path, download: bool = False, train_test_split_ratio: float = 0.8, train_val_split_ratio: float = 0.9) -> None:
    print("Preprocessing data...")
    dataset = WineData(data_path, download=download)
    dataset.preprocess(output_folder)
    print("Splitting data...")
    split_data(dataset.data, output_folder, train_test_split_ratio=train_test_split_ratio, train_val_split_ratio=train_val_split_ratio)


if __name__ == "__main__":
    print("Starting data preprocessing...")
    typer.run(preprocess)
    # preprocess(
    #     data_path=Path("data/raw/Wqt.csv"),
    #     output_folder=Path("data/processed/"),
    #     download=False,
    #     train_test_split_ratio=0.8,
    #     train_val_split_ratio=0.9
    # )       
    print("Data preprocessing completed.")


# Windows terminal command to run the script:
# ./src/mlops_exam_project/data.py ./data/raw/Wqt.csv ./data/processed/ --download
