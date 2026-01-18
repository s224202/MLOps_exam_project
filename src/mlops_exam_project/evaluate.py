from pathlib import Path
import os
import torch
import typer
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from data import WineData 
from model import WineQualityClassifier

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)







# def evaluate(
#     model_checkpoint: Path = Path("models/wine_classifier.pth"),
#     data_path: Path = Path("data/processed"),
#     batch_size: int = 32,
#     hidden_dims: str = "64,32",
#     dropout_rate: float = 0.3,
# ) -> None:
"""
Evaluate a trained wine quality classifier.

Args:
    model_checkpoint: Path to the saved model checkpoint
    data_path: Path to preprocessed data
    batch_size: Batch size for evaluation
    hidden_dims: Comma-separated hidden layer dimensions (must match training)
    dropout_rate: Dropout rate (must match training)
"""
@hydra.main(version_base=None, config_path="../../configs", config_name="config") #, data_path: Path = "data/processed", model_path: Path = "models")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluate a trained wine quality classifier.
    Args:
        cfg: Configuration dictionary with evaluation parameters
    """



       #  Hydra changes the working directory to outputs/<date>/<time> for each run. Use an absolute path or make the path relative to the project root.
    #  The issue is that Hydra changes the working directory to outputs/<date>/<time> for each run.  Here we use an absolute path (or we ould make the path relative to the project root).
    project_root = Path(__file__).parent.parent.parent # Getting the project root directory
    #data_path = Path(cfg.data_path)
    data_path = project_root / cfg.data_path
    #train_data_name = cfg.train_data_filename
    #val_data_name = cfg.val_data_filename
    test_data_name = cfg.test_data_filename
    #model_path = Path(cfg.model_path)
    model_path = project_root / cfg.model_path
    model_name = cfg.model_name

    # print working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Using device: {DEVICE}")
    print(f"Test configuration: {cfg.training}")
    print(f"Test data path: {data_path / test_data_name}")



    print(f"Evaluating wine quality classifier on {DEVICE}")
    print(f"Model checkpoint: {model_path / model_name}")
    
    # Parse hidden dimensions
    #hidden_dims_list = [int(x) for x in cfg.training.hidden_dims.split(",")]
    hidden_dims_list = list(cfg.training.hidden_dims)
    
    # Load test dataset
    test_set = WineData(data_path / test_data_name, download=False)
    test_dataloader = DataLoader(test_set, batch_size=cfg.training.batch_size, shuffle=False)
    
    # Get number of classes
    num_classes =  6 #len(torch.unique(test_set.data["quality"]))
    print(f"Number of classes: {num_classes}")
    
    # Initialize model with same architecture as training
    model = WineQualityClassifier(
        input_dim= 11+1,
        hidden_dims=hidden_dims_list,
        output_dim=num_classes,
        dropout_rate=cfg.training.dropout_rate
    ).to(DEVICE)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path / model_name, map_location=DEVICE))
    print("Model loaded successfully\n")
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            # Forward pass
            outputs = model(features)
            predictions = outputs.argmax(dim=1)
            # Accumulate results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    # Calculate accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})\n")
    
    # Print detailed classification report
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=[f"Quality {i}" for i in range(num_classes)],
        digits=4
    ))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    print("\nRows represent true labels, columns represent predicted labels")


if __name__ == "__main__":
    #typer.run(evaluate)
    evaluate()