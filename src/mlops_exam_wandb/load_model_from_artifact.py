


import os
from pathlib import Path
from typing import Optional

import torch
import typer
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
    """Return the first .pth file in the artifact directory."""
    for ext in ("*.pth", "*.pt"):
        matches = list(artifact_dir.rglob(ext))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No checkpoint file (*.pth or *.pt) found in {artifact_dir}")


def load_model_from_artifact(
    artifact_name: str = "red_wine_quality_model:latest",
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> MyAwesomeModel:
    """Load a model from W&B artifact registry with its configuration."""
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
    artifact_dir = Path(artifact.download())

    ckpt_path = _find_checkpoint(artifact_dir)

    metadata = artifact.metadata
    hidden_dims = metadata.get("hidden_dims", [16, 8])
    dropout_rate = metadata.get("dropout_rate", 0.1)
    input_dim = metadata.get("input_dim", 12)
    output_dim = metadata.get("output_dim", 6)

    print(f"Loading checkpoint: {ckpt_path}")
    print("Configuration from artifact metadata:")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  dropout_rate: {dropout_rate}")
    print(f"  input_dim: {input_dim}")
    print(f"  output_dim: {output_dim}")

    model = MyAwesomeModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print(f"Model loaded from artifact: {artifact_name}")
    print(f"Final validation accuracy: {metadata.get('final_val_accuracy', 'N/A')}")
    print(f"Final validation loss: {metadata.get('final_val_loss', 'N/A')}")

    wandb.finish()

    return model


if __name__ == "__main__":
    typer.run(load_model_from_artifact)
    #model = load_model_from_artifact()
    #print(f"Model loaded successfully on {DEVICE}")


    #example usage:
    # uv run src/mlops_exam_wandb/load_model_from_artifact.py

    # example with specified artifact, project, entity:
    # uv run src/mlops_exam_wandb/load_model_from_artifact.py --artifact-name "red_wine_quality_model:v17" 

    # uv run src/mlops_exam_wandb/load_model_from_artifact.py --artifact-name "red_wine_quality_model:v0" --project "mlops_exam_project" --entity "mr-mikael-sorensen"

    

# import os
# from pathlib import Path

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


# def load_model_from_artifact(
#     artifact_name: str = "red_wine_quality_model:latest",
#     project: str = None,
#     entity: str = None,
# ) -> MyAwesomeModel:
#     """Load a model from W&B artifact registry with its configuration.

#     Args:
#         artifact_name: Name of the artifact with optional version (e.g., 'model:latest' or 'model:v0')
#         project: W&B project name (loaded from .env if not provided)
#         entity: W&B entity name (loaded from .env if not provided)

#     Returns:
#         Loaded model instance configured with parameters from artifact metadata
#     """
#     project_root = Path(__file__).parent.parent.parent
#     load_dotenv(project_root / ".env", override=True)

#     if project is None:
#         project = os.getenv("WANDB_PROJECT", "mlops_exam_project")
#     if entity is None:
#         entity = os.getenv("WANDB_ENTITY")

#     wandb_api_key = os.getenv("WANDB_API_KEY")
#     if wandb_api_key:
#         wandb.login(key=wandb_api_key)

#     run = wandb.init(project=project, entity=entity, job_type="inference")

#     artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}", type="model")
#     artifact_dir = artifact.download()

#     model_path = Path(artifact_dir) / "red_wine_quality_model.pth"

#     metadata = artifact.metadata
#     hidden_dims = metadata.get("hidden_dims", [16, 8])
#     dropout_rate = metadata.get("dropout_rate", 0.1)
#     input_dim = metadata.get("input_dim", 12)
#     output_dim = metadata.get("output_dim", 6)

#     print(f"Loading model with configuration from artifact:")
#     print(f"  - Hidden dimensions: {hidden_dims}")
#     print(f"  - Dropout rate: {dropout_rate}")
#     print(f"  - Input dimension: {input_dim}")
#     print(f"  - Output dimension: {output_dim}")

#     model = MyAwesomeModel(
#         input_dim=input_dim,
#         hidden_dims=hidden_dims,
#         output_dim=output_dim,
#         dropout_rate=dropout_rate,
#     ).to(DEVICE)

#     model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
#     model.eval()

#     print(f"Model loaded from artifact: {artifact_name}")
#     print(f"Final validation accuracy: {metadata.get('final_val_accuracy', 'N/A')}")
#     print(f"Final validation loss: {metadata.get('final_val_loss', 'N/A')}")

#     wandb.finish()

#     return model


# if __name__ == "__main__":
#     model = load_model_from_artifact()
#     print(f"Model loaded successfully on {DEVICE}")