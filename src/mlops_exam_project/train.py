# import matplotlib.pyplot as plt
import torch
import hydra
import os
import matplotlib.pyplot as plt
from data import WineData
from model import WineQualityClassifier as MyAwesomeModel
from omegaconf import DictConfig
from pathlib import Path
from loguru import logger as llogger

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


@hydra.main(
    version_base=None, config_path="../../configs", config_name="config"
)  # , data_path: Path = "data/processed", model_path: Path = "models")
def train(cfg: DictConfig) -> None:
    print("Training day and night")

    #  Hydra changes the working directory to outputs/<date>/<time> for each run. Use an absolute path or make the path relative to the project root.
    #  The issue is that Hydra changes the working directory to outputs/<date>/<time> for each run.  Here we use an absolute path (or we ould make the path relative to the project root).
    project_root = Path(
        __file__
    ).parent.parent.parent  # Getting the project root directory
    # data_path = Path(cfg.data_path)
    data_path = project_root / cfg.data_path
    llogger.debug("Data path: {}", data_path)
    train_data_name = cfg.train_data_filename
    llogger.debug("Train data name: {}", train_data_name)
    val_data_name = cfg.val_data_filename
    llogger.debug("Validation data name: {}", val_data_name)

    # test_data_name = cfg.test_data_filename
    # model_path = Path(cfg.model_path)
    model_path = project_root / cfg.model_path
    model_name = "onnx_wine_model.pth"

    # print working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Using device: {DEVICE}")
    print(f"Training configuration: {cfg.training}")
    llogger.debug("Training configuration: {}", cfg.training)
    print(f"Training data path: {data_path / train_data_name}")
    print(f"Validation data path: {data_path / val_data_name}")
    print(f"Model will be saved to: {model_path / model_name}")

    model = MyAwesomeModel(
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

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    statistics = {
        "train_loss": [],
        "train_accuracy": [],
        "epoch_loss": [],
        "epoch_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        llogger.info("Starting epoch {}", epoch)

        for i, (features, target) in enumerate(train_dataloader):
            features, target = features.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(features)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            # Record batch statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            # Accumulate epoch statistics
            epoch_loss += loss.item()
            epoch_correct += (y_pred.argmax(dim=1) == target).sum().item()
            epoch_total += target.size(0)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Calculate epoch averages
        avg_loss = epoch_loss / len(train_dataloader)
        llogger.info("Epoch {} average loss: {}", epoch, avg_loss)
        avg_accuracy = epoch_correct / epoch_total
        llogger.info("Epoch {} average accuracy: {}", epoch, avg_accuracy)
        statistics["epoch_loss"].append(avg_loss)
        statistics["epoch_accuracy"].append(avg_accuracy)

        # do validation every epoch
        llogger.info("Starting validation for epoch {}", epoch)
        model.eval()
        with torch.no_grad():
            val_accuracy = 0
            val_loss = 0
            val_steps = 0
            for features, target in val_dataloader:
                features, target = features.to(DEVICE), target.to(DEVICE)
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

            llogger.info(
                "Epoch {} validation loss: {}, validation accuracy: {}",
                epoch,
                val_loss,
                val_accuracy,
            )
            print(
                f"Epoch {epoch}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}"
            )

    print("Training complete")
    llogger.info("Training complete")
    torch.save(model.state_dict(), model_path / model_name)
    # Export to ONNX using the stable torch.onnx.export API
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
    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # axs[0].plot(statistics["train_loss"],  marker='o', color='blue', markersize=2, label='Train Loss')
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
    # axs[1].plot(statistics["train_accuracy"], marker='o', color='blue', markersize=2, label='Train Accuracy')
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

    # fig.savefig(project_root / cfg.figure_path / cfg.figure_training_plot, bbox_inches='tight') # dpi=150,

    # changes to save in reports/figures instead of figures/; needed for the docker + hydra setup
    fig_dir = Path(cfg.figure_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / cfg.figure_training_plot, bbox_inches="tight")


if __name__ == "__main__":
    train()
