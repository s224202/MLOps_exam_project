import torch
import hydra
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path
from loguru import logger as llogger

# Handle imports for both Hydra context (relative) and test context (absolute)
try:
    from data import WineData
    from model import Model
except ImportError:
    from mlops_exam_project.data import WineData
    from mlops_exam_project.model import Model

llogger.remove()  # remove the default configuration
llogger.add("logs/Mlops-31.log", rotation="10 MB", level="DEBUG")  # log to file

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
llogger.debug("Using device: {}", DEVICE)


def train_core(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    device=None,
    learning_rate=0.001,
):
    """Core training logic - testable without Hydra.
    Args:
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        epochs: Number of epochs to train
        device: Device to train on
        learning_rate: Learning rate for optimizer
    Returns:
        Tuple of (trained model, statistics dict)
    """
    if device is None:
        device = DEVICE
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    statistics = {
        "train_loss": [],
        "train_accuracy": [],
        "epoch_loss": [],
        "epoch_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for features, target in train_dataloader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(features)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            epoch_loss += loss.item()
            epoch_correct += (y_pred.argmax(dim=1) == target).sum().item()
            epoch_total += target.size(0)
        avg_loss = epoch_loss / len(train_dataloader)
        avg_accuracy = epoch_correct / epoch_total
        statistics["epoch_loss"].append(avg_loss)
        statistics["epoch_accuracy"].append(avg_accuracy)
        model.eval()
        with torch.no_grad():
            val_accuracy = 0
            val_loss = 0
            val_steps = 0
            for features, target in val_dataloader:
                features, target = features.to(device), target.to(device)
                y_pred = model(features)
                loss = loss_fn(y_pred, target)
                val_loss += loss.item()
                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                val_accuracy += accuracy
                val_steps += 1
            val_accuracy /= val_steps
            val_loss /= val_steps
            statistics["val_loss"].append(val_loss)
            statistics["val_accuracy"].append(val_accuracy)
    return model, statistics


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print("Training day and night")
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / cfg.data_path
    llogger.debug("Data path: {}", data_path)
    train_data_name = cfg.train_data_filename
    llogger.debug("Train data name: {}", train_data_name)
    val_data_name = cfg.val_data_filename
    llogger.debug("Validation data name: {}", val_data_name)
    model_path = project_root / cfg.model_path
    model_name = "onnx_wine_model.pth"
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Using device: {DEVICE}")
    print(f"Training configuration: {cfg.training}")
    llogger.debug("Training configuration: {}", cfg.training)
    print(f"Training data path: {data_path / train_data_name}")
    print(f"Validation data path: {data_path / val_data_name}")
    print(f"Model will be saved to: {model_path / model_name}")
    model = Model(
        input_dim=12,
        hidden_dims=cfg.training.hidden_dims,
        output_dim=6,
        dropout_rate=cfg.training.dropout_rate,
    ).to(DEVICE)
    train_set = WineData(data_path / train_data_name, False)
    val_set = WineData(data_path / val_data_name, False)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.training.batch_size
    )
    model, statistics = train_core(
        model,
        train_dataloader,
        val_dataloader,
        cfg.training.epochs,
        device=DEVICE,
        learning_rate=cfg.training.lr,
    )
    print("Training complete")
    llogger.info("Training complete")
    torch.save(model.state_dict(), model_path / model_name)
    example_input = torch.randn(1, 12).to(DEVICE)
    onnx_file = model_path / "onnx_wine_model.onnx"
    torch.onnx.export(
        model,
        example_input,
        str(onnx_file),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model saved to {model_path / model_name}")
    llogger.info("Model saved to {}", model_path / model_name)
    print(f"ONNX model saved to {onnx_file}")
    llogger.info("ONNX model saved to {}", onnx_file)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(
        statistics["epoch_loss"],
        marker="o",
        color="blue",
        markersize=2,
        label="Train Loss",
    )
    axs[0].plot(
        statistics["val_loss"],
        marker="o",
        color="orange",
        markersize=2,
        label="Validation Loss",
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_title("Loss")
    axs[0].set_xticks(range(1, cfg.training.epochs))
    axs[0].set_xticklabels(range(1, cfg.training.epochs))
    axs[0].legend()
    axs[1].plot(
        statistics["epoch_accuracy"],
        marker="o",
        color="blue",
        markersize=2,
        label="Train Accuracy",
    )
    axs[1].plot(
        statistics["val_accuracy"],
        marker="o",
        color="orange",
        markersize=2,
        label="Validation Accuracy",
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Accuracy")
    axs[1].set_xticks(range(1, cfg.training.epochs))
    axs[1].set_xticklabels(range(1, cfg.training.epochs))
    axs[1].legend()
    fig_dir = Path(cfg.figure_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / cfg.figure_training_plot, bbox_inches="tight")


if __name__ == "__main__":
    train()
