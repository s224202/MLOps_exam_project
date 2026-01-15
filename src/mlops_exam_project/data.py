from pathlib import Path
import typer
from torch.utils.data import Dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

app = typer.Typer()


class WineData(Dataset):
    """A custom dataset class for handling the wine quality data."""

    def __init__(self, data_path: Path, download: bool = False) -> None:
        self.data_path = data_path
        if not self.data_path.exists():
            if download:
                wine_quality = fetch_ucirepo(id=186)
                print(wine_quality.data.keys())
                os.makedirs(self.data_path.parent, exist_ok=True)
                colors = wine_quality.data.original["color"].map({"red": 1, "white": 0})
                wine_quality.data.features["color"] = colors
                features = wine_quality.data.features.copy()
                features["quality"] = wine_quality.data.target
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
        return self.data.iloc[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data to have zero mean and unit variance and save it to the output folder."""
        processed_data = self.data.copy()
        for column in processed_data.columns:
            if column != "quality":  # Since we don't want to scale the target variable
                mean = processed_data[column].mean()
                std = processed_data[column].std()
                processed_data[column] = (processed_data[column] - mean) / std
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / "processed_wine_data.csv"
        processed_data.drop(columns=["color"], inplace=True)
        processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


@app.command()
def preprocess(data_path: Path, output_folder: Path, download: bool = False) -> None:
    print("Preprocessing data...")
    dataset = WineData(data_path, download=download)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    app()
