import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from data import WineData
from model import WineQualityClassifier

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def _find_checkpoint(artifact_dir: Path) -> Path:
    """Find the first .pth file in the artifact directory."""
    for ext in ("*.pth", "*.pt"):
        matches = list(artifact_dir.rglob(ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint file (*.pth or *.pt) found in {artifact_dir}")


def load_model_from_artifact(
    artifact_name: str,
    project: str,
    entity: str,
) -> tuple[WineQualityClassifier, dict]:
    """Load model from W&B artifact.
    
    Args:
        artifact_name: Full artifact name (e.g., 'red_wine_quality_model:latest')
        project: W&B project name
        entity: W&B entity name
        
    Returns:
        Tuple of (model, metadata)
    """
    run = wandb.init(project=project, entity=entity, job_type="evaluation")
    
    artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}", type="model")
    artifact_dir = Path(artifact.download())
    
    ckpt_path = _find_checkpoint(artifact_dir)
    metadata = artifact.metadata
    
    hidden_dims = metadata.get("hidden_dims", [16, 8])
    dropout_rate = metadata.get("dropout_rate", 0.1)
    input_dim = metadata.get("input_dim", 12)
    output_dim = metadata.get("output_dim", 6)
    
    print(f"Loading model from artifact: {artifact_name}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  dropout_rate: {dropout_rate}")
    print(f"  input_dim: {input_dim}")
    print(f"  output_dim: {output_dim}")
    
    model = WineQualityClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    ).to(DEVICE)
    
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    
    return model, metadata


def load_model_from_registry(
    model_name: str,
    version: str,
    entity: str,
) -> tuple[WineQualityClassifier, dict]:
    """Load model from W&B Model Registry.
    
    Args:
        model_name: Name of the model in registry
        version: Version or alias (e.g., 'latest', 'production', 'v0')
        entity: W&B entity name
        
    Returns:
        Tuple of (model, metadata)
    """
    run = wandb.init(project="mlops_exam_project", entity=entity, job_type="evaluation")
    
    registry_path = f"{entity}/model-registry/{model_name}:{version}"
    artifact = run.use_artifact(registry_path, type="model")
    artifact_dir = Path(artifact.download())
    
    ckpt_path = _find_checkpoint(artifact_dir)
    metadata = artifact.metadata
    
    hidden_dims = metadata.get("hidden_dims", [16, 8])
    dropout_rate = metadata.get("dropout_rate", 0.1)
    input_dim = metadata.get("input_dim", 12)
    output_dim = metadata.get("output_dim", 6)
    
    print(f"Loading model from registry: {registry_path}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  dropout_rate: {dropout_rate}")
    print(f"  input_dim: {input_dim}")
    print(f"  output_dim: {output_dim}")
    
    model = WineQualityClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    ).to(DEVICE)
    
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    
    return model, metadata


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained wine quality classifier.
    
    Args:
        cfg: Configuration dictionary with evaluation parameters
    """
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env", override=True)
    
    wandb_project = os.getenv("WANDB_PROJECT", "mlops_exam_project")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    
    data_path = project_root / cfg.data_path
    test_data_name = cfg.test_data_filename
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Using device: {DEVICE}")
    print(f"Test data path: {data_path / test_data_name}")
    
    use_wandb_artifact = cfg.get("use_wandb_artifact", False)
    use_wandb_registry = cfg.get("use_wandb_registry", False)
    
    if use_wandb_artifact:
        artifact_name = cfg.get("wandb_artifact_name", "red_wine_quality_model:latest")
        print(f"\nLoading model from W&B artifact: {artifact_name}")
        model, metadata = load_model_from_artifact(artifact_name, wandb_project, wandb_entity)
        num_classes = metadata.get("output_dim", 6)
    elif use_wandb_registry:
        registry_model_name = cfg.get("wandb_registry_model_name", "red_wine_quality_model")
        registry_version = cfg.get("wandb_registry_version", "latest")
        print(f"\nLoading model from W&B registry: {registry_model_name}:{registry_version}")
        model, metadata = load_model_from_registry(registry_model_name, registry_version, wandb_entity)
        num_classes = metadata.get("output_dim", 6)
    else:
        model_path = project_root / cfg.model_path
        model_name = cfg.model_name
        print(f"\nLoading model from local checkpoint: {model_path / model_name}")
        
        hidden_dims_list = list(cfg.training.hidden_dims)
        num_classes = 6
        
        model = WineQualityClassifier(
            input_dim=12,
            hidden_dims=hidden_dims_list,
            output_dim=num_classes,
            dropout_rate=cfg.training.dropout_rate,
        ).to(DEVICE)
        
        model.load_state_dict(torch.load(model_path / model_name, map_location=DEVICE, weights_only=True))
        metadata = {}
    
    print("Model loaded successfully\n")
    
    test_set = WineData(data_path / test_data_name, download=False)
    test_dataloader = DataLoader(test_set, batch_size=cfg.training.batch_size, shuffle=False)
    
    print(f"Number of classes: {num_classes}")
    
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})\n")
    
    if use_wandb_artifact or use_wandb_registry:
        wandb.log({
            "test_accuracy": accuracy,
            "test_correct": correct,
            "test_total": total,
        })
        
        if metadata:
            print(f"Training validation accuracy: {metadata.get('final_val_accuracy', 'N/A')}")
            print(f"Training validation loss: {metadata.get('final_val_loss', 'N/A')}\n")
    
    print("Classification Report:")
    print("=" * 60)
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[f"Quality {i}" for i in range(num_classes)],
            digits=4,
        )
    )
    
    print("\nConfusion Matrix:")
    print("=" * 60)
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    print("\nRows represent true labels, columns represent predicted labels")
    
    if use_wandb_artifact or use_wandb_registry:
        wandb.finish()


if __name__ == "__main__":
    evaluate()


# from pathlib import Path
# import os
# import torch
# from sklearn.metrics import classification_report, confusion_matrix
# from torch.utils.data import DataLoader
# from omegaconf import DictConfig
# import hydra
# from data import WineData
# from model import WineQualityClassifier

# DEVICE = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )


# # def evaluate(
# #     model_checkpoint: Path = Path("models/wine_classifier.pth"),
# #     data_path: Path = Path("data/processed"),
# #     batch_size: int = 32,
# #     hidden_dims: str = "64,32",
# #     dropout_rate: float = 0.3,
# # ) -> None:
# """
# Evaluate a trained wine quality classifier.

# Args:
#     model_checkpoint: Path to the saved model checkpoint
#     data_path: Path to preprocessed data
#     batch_size: Batch size for evaluation
#     hidden_dims: Comma-separated hidden layer dimensions (must match training)
#     dropout_rate: Dropout rate (must match training)
# """


# @hydra.main(
#     version_base=None, config_path="../../configs", config_name="config"
# )  # , data_path: Path = "data/processed", model_path: Path = "models")
# def evaluate(cfg: DictConfig) -> None:
#     """
#     Evaluate a trained wine quality classifier.
#     Args:
#         cfg: Configuration dictionary with evaluation parameters
#     """

#     #  Hydra changes the working directory to outputs/<date>/<time> for each run. Use an absolute path or make the path relative to the project root.
#     #  The issue is that Hydra changes the working directory to outputs/<date>/<time> for each run.  Here we use an absolute path (or we ould make the path relative to the project root).
#     project_root = Path(
#         __file__
#     ).parent.parent.parent  # Getting the project root directory
#     # data_path = Path(cfg.data_path)
#     data_path = project_root / cfg.data_path
#     # train_data_name = cfg.train_data_filename
#     # val_data_name = cfg.val_data_filename
#     test_data_name = cfg.test_data_filename
#     # model_path = Path(cfg.model_path)
#     model_path = project_root / cfg.model_path
#     model_name = cfg.model_name

#     # print working directory
#     print(f"Working directory: {os.getcwd()}")
#     print(f"Project root: {project_root}")
#     print(f"Using device: {DEVICE}")
#     print(f"Test configuration: {cfg.training}")
#     print(f"Test data path: {data_path / test_data_name}")

#     print(f"Evaluating wine quality classifier on {DEVICE}")
#     print(f"Model checkpoint: {model_path / model_name}")

#     # Parse hidden dimensions
#     # hidden_dims_list = [int(x) for x in cfg.training.hidden_dims.split(",")]
#     hidden_dims_list = list(cfg.training.hidden_dims)

#     # Load test dataset
#     test_set = WineData(data_path / test_data_name, download=False)
#     test_dataloader = DataLoader(
#         test_set, batch_size=cfg.training.batch_size, shuffle=False
#     )

#     # Get number of classes
#     num_classes = 6  # len(torch.unique(test_set.data["quality"]))
#     print(f"Number of classes: {num_classes}")

#     # Initialize model with same architecture as training
#     model = WineQualityClassifier(
#         input_dim=11 + 1,
#         hidden_dims=hidden_dims_list,
#         output_dim=num_classes,
#         dropout_rate=cfg.training.dropout_rate,
#     ).to(DEVICE)

#     # Load trained weights
#     model.load_state_dict(torch.load(model_path / model_name, map_location=DEVICE))
#     print("Model loaded successfully\n")

#     # Evaluate
#     model.eval()
#     all_predictions = []
#     all_labels = []
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for features, labels in test_dataloader:
#             features, labels = features.to(DEVICE), labels.to(DEVICE)
#             # Forward pass
#             outputs = model(features)
#             predictions = outputs.argmax(dim=1)
#             # Accumulate results
#             all_predictions.extend(predictions.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             correct += (predictions == labels).sum().item()
#             total += labels.size(0)

#     # Calculate accuracy
#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})\n")

#     # Print detailed classification report
#     print("Classification Report:")
#     print("=" * 60)
#     print(
#         classification_report(
#             all_labels,
#             all_predictions,
#             target_names=[f"Quality {i}" for i in range(num_classes)],
#             digits=4,
#         )
#     )

#     # Print confusion matrix
#     print("\nConfusion Matrix:")
#     print("=" * 60)
#     cm = confusion_matrix(all_labels, all_predictions)
#     print(cm)
#     print("\nRows represent true labels, columns represent predicted labels")


# if __name__ == "__main__":
#     # typer.run(evaluate)
#     evaluate()
