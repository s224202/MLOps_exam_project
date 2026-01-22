from pathlib import Path
from mlops_exam_project.data import WineData
from mlops_exam_project.model import WineQualityClassifier


def train(data_path: Path = Path("data/raw/WineQT.csv")) -> None:
    """Train the model on wine data."""
    dataset = WineData(data_path, download=True)
    print(f"Dataset size: {len(dataset)}")
    model = WineQualityClassifier()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Model state: {model.__getstate__()}")


if __name__ == "__main__":
    train()
