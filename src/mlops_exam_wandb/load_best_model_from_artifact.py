import os
from pathlib import Path
from typing import Literal, Optional
import typer
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


def load_model_from_artifact(
    artifact_name: str = "red_wine_quality_model:latest",
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> MyAwesomeModel:
    """Load a model from W&B artifact registry with its configuration.

    Args:
        artifact_name: Name of the artifact with optional version (e.g., 'model:latest' or 'model:v0')
        project: W&B project name (loaded from .env if not provided)
        entity: W&B entity name (loaded from .env if not provided)

    Returns:
        Loaded model instance configured with parameters from artifact metadata
    """
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env", override=True)

    if project is None:
        project = os.getenv("WANDB_PROJECT", "mlops_exam_project")
    if entity is None:
        entity = os.getenv("WANDB_ENTITY")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    run = wandb.init(project=project, entity=entity, job_type="inference")

    artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}", type="model")
    artifact_dir = artifact.download()

    model_path = Path(artifact_dir) / "red_wine_quality_model.pth"

    metadata = artifact.metadata
    hidden_dims = metadata.get("hidden_dims", [16, 8])
    dropout_rate = metadata.get("dropout_rate", 0.1)
    input_dim = metadata.get("input_dim", 12)
    output_dim = metadata.get("output_dim", 6)

    print(f"Loading model with configuration from artifact:")
    print(f"  - Hidden dimensions: {hidden_dims}")
    print(f"  - Dropout rate: {dropout_rate}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Output dimension: {output_dim}")

    model = MyAwesomeModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Model loaded from artifact: {artifact_name}")
    print(f"Final validation accuracy: {metadata.get('final_val_accuracy', 'N/A')}")
    print(f"Final validation loss: {metadata.get('final_val_loss', 'N/A')}")

    wandb.finish()

    return model


def load_best_model(
    metric: Literal["final_val_accuracy", "final_val_loss"] = "final_val_accuracy",
    mode: Literal["min", "max"] = "max",
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> MyAwesomeModel:
    """Load the best model based on a specific metric.

    Args:
        metric: Metadata field to optimize ('final_val_accuracy' or 'final_val_loss')
        mode: Whether to minimize or maximize the metric
        project: W&B project name (loaded from .env if not provided)
        entity: W&B entity name (loaded from .env if not provided)

    Returns:
        Best model instance based on the specified metric
    """
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env", override=True)

    if project is None:
        project = os.getenv("WANDB_PROJECT", "mlops_exam_project")
    if entity is None:
        entity = os.getenv("WANDB_ENTITY")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    api = wandb.Api()
    artifact_collection = f"{entity}/{project}/red_wine_quality_model"

    artifacts = api.artifact_versions("model", artifact_collection)

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
        raise ValueError(f"No artifacts found with metadata field '{metric}'")

    print(f"Found best model with {metric}={best_value} (mode={mode})")
    print(f"Artifact: {best_artifact.name}")

    return load_model_from_artifact(
        artifact_name=f"red_wine_quality_model:{best_artifact.version}",
        project=project,
        entity=entity,
    )


if __name__ == "__main__":
    typer.run(load_model_from_artifact)

    
    # print("Loading model with highest accuracy:")
    # model_best = load_best_model(metric="final_val_accuracy", mode="max")

    # print("\nLoading model with lowest loss:")
    # model_lowest_loss = load_best_model(metric="final_val_loss", mode="min")

    # print(f"\nModels loaded successfully on {DEVICE}")


        #example usage:
    # uv run src/mlops_exam_wandb/load_model_from_artifact.py

    # example with specified artifact, project, entity:
    # uv run src/mlops_exam_wandb/load_model_from_artifact.py --artifact-name "red_wine_quality_model:v17" 

    # uv run src/mlops_exam_wandb/load_model_from_artifact.py --artifact-name "red_wine_quality_model:v0" --project "mlops_exam_project" --entity "mr-mikael-sorensen"

    