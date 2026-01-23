"""Load models from W&B Model Registry."""

from pathlib import Path
from typing import Literal, Optional

import os
import torch
import wandb
from dotenv import load_dotenv

from model import WineQualityClassifier as MyAwesomeModel

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def _find_checkpoint(artifact_dir: Path) -> Path:
    """Find the first checkpoint file in an artifact directory."""
    for pattern in ("*.pth", "*.pt"):
        matches = list(artifact_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint file (*.pth or *.pt) found in {artifact_dir}")


def load_model_from_registry(
    model_name: str = "red_wine_quality_model",
    version: str = "latest",
    alias: Optional[str] = None,
    entity: Optional[str] = None,
) -> MyAwesomeModel:
    """Load a model from W&B Model Registry.

    Args:
        model_name: Name of the model in the registry.
        version: Version to load (e.g., 'latest', 'v0', 'v1').
        alias: Alias to load (e.g., 'production', 'staging', 'best') - overrides version.
        entity: W&B entity name (loaded from .env if not provided).

    Returns:
        Loaded model instance configured with parameters from artifact metadata.
    """
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env", override=True)

    if entity is None:
        entity = os.getenv("WANDB_ENTITY")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    run = wandb.init(project="mlops_exam_project", entity=entity, job_type="inference")

    registry_path = f"{entity}/model-registry/{model_name}"
    artifact_path = f"{registry_path}:{alias}" if alias else f"{registry_path}:{version}"

    artifact = run.use_artifact(artifact_path, type="model")
    artifact_dir = Path(artifact.download())
    ckpt_path = _find_checkpoint(artifact_dir)

    metadata = artifact.metadata
    hidden_dims = metadata.get("hidden_dims", [16, 8])
    dropout_rate = metadata.get("dropout_rate", 0.1)
    input_dim = metadata.get("input_dim", 12)
    output_dim = metadata.get("output_dim", 6)

    print(f"Loading model from registry: {artifact_path}")
    print("Configuration:")
    print(f"  - Hidden dimensions: {hidden_dims}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Final validation accuracy: {metadata.get('final_val_accuracy', 'N/A')}")
    print(f"  - Final validation loss: {metadata.get('final_val_loss', 'N/A')}")
    print(f"  - Checkpoint file: {ckpt_path}")

    model = MyAwesomeModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    wandb.finish()

    return model

# import os
# from pathlib import Path
# from typing import Literal, Optional

# import torch
# import wandb
# from dotenv import load_dotenv

# from model import WineQualityClassifier as MyAwesomeModel

# DEVICE = torch.device(
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )


# def load_model_from_registry(
#     model_name: str = "red_wine_quality_model",
#     version: str = "latest",
#     alias: Optional[str] = None,
#     entity: Optional[str] = None,
# ) -> MyAwesomeModel:
#     """Load a model from W&B Model Registry.

#     Args:
#         model_name: Name of the model in the registry
#         version: Version to load (e.g., 'latest', 'v0', 'v1')
#         alias: Alias to load (e.g., 'production', 'staging', 'best') - overrides version
#         entity: W&B entity name (loaded from .env if not provided)

#     Returns:
#         Loaded model instance configured with parameters from artifact metadata
#     """
#     project_root = Path(__file__).parent.parent.parent
#     load_dotenv(project_root / ".env", override=True)

#     if entity is None:
#         entity = os.getenv("WANDB_ENTITY")

#     wandb_api_key = os.getenv("WANDB_API_KEY")
#     if wandb_api_key:
#         wandb.login(key=wandb_api_key)

#     run = wandb.init(project="mlops_exam_project", entity=entity, job_type="inference")

#     registry_path = f"{entity}/model-registry/{model_name}"
#     if alias:
#         artifact_path = f"{registry_path}:{alias}"
#     else:
#         artifact_path = f"{registry_path}:{version}"

#     artifact = run.use_artifact(artifact_path, type="model")
#     artifact_dir = artifact.download()

#     model_path = Path(artifact_dir) / "red_wine_quality_model.pth"

#     metadata = artifact.metadata
#     hidden_dims = metadata.get("hidden_dims", [16, 8])
#     dropout_rate = metadata.get("dropout_rate", 0.1)
#     input_dim = metadata.get("input_dim", 12)
#     output_dim = metadata.get("output_dim", 6)

#     print(f"Loading model from registry: {artifact_path}")
#     print(f"Configuration:")
#     print(f"  - Hidden dimensions: {hidden_dims}")
#     print(f"  - Dropout rate: {dropout_rate}")
#     print(f"  - Final validation accuracy: {metadata.get('final_val_accuracy', 'N/A')}")
#     print(f"  - Final validation loss: {metadata.get('final_val_loss', 'N/A')}")

#     model = MyAwesomeModel(
#         input_dim=input_dim,
#         hidden_dims=hidden_dims,
#         output_dim=output_dim,
#         dropout_rate=dropout_rate,
#     ).to(DEVICE)

#     model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
#     model.eval()

#     wandb.finish()

#     return model


def load_best_model_from_registry(
    model_name: str = "red_wine_quality_model",
    metric: Literal["final_val_accuracy", "final_val_loss"] = "final_val_accuracy",
    mode: Literal["min", "max"] = "max",
    entity: Optional[str] = None,
) -> MyAwesomeModel:
    """Load the best model from the registry based on a specific metric.

    Args:
        model_name: Name of the model in the registry
        metric: Metadata field to optimize
        mode: Whether to minimize or maximize the metric
        entity: W&B entity name (loaded from .env if not provided)

    Returns:
        Best model instance based on the specified metric
    """
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env", override=True)

    if entity is None:
        entity = os.getenv("WANDB_ENTITY")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    api = wandb.Api()
    registry_path = f"{entity}/model-registry/{model_name}"

    artifacts = api.artifact_versions("model", registry_path)

    best_artifact = None
    best_value = float("inf") if mode == "min" else float("-inf")

    for artifact in artifacts:
        metadata = artifact.metadata
        if metric not in metadata:
            continue

        value = metadata[metric]
        if (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
            best_value = value
            best_artifact = artifact

    if best_artifact is None:
        raise ValueError(f"No artifacts found in registry with metadata field '{metric}'")

    print(f"Found best model with {metric}={best_value} (mode={mode})")
    print(f"Version: {best_artifact.version}")

    return load_model_from_registry(
        model_name=model_name,
        version=best_artifact.version,
        entity=entity,
    )


if __name__ == "__main__":
    print("Loading latest model from registry:")
    model_latest = load_model_from_registry()

    print("\nLoading model with highest validation accuracy:")
    model_best = load_best_model_from_registry(metric="final_val_accuracy", mode="min")

    print(f"\nModels loaded successfully on {DEVICE}")