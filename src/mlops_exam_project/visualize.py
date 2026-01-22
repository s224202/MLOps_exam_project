from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import hydra
from omegaconf import DictConfig
from data import WineData
from model import WineQualityClassifier as MyAwesomeModel


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def visualize(cfg: DictConfig) -> None:
    """
    Create visualizations for the wine quality classifier.

    Args:
        cfg: Configuration dictionary with visualization parameters
    """
    #  Hydra changes the working directory to outputs/<date>/<time> for each run. Use an absolute path or make the path relative to the project root.
    #  The issue is that Hydra changes the working directory to outputs/<date>/<time> for each run.  Here we use an absolute path (or we ould make the path relative to the project root).
    project_root = Path(
        __file__
    ).parent.parent.parent  # Getting the project root directory
    # data_path = Path(cfg.data_path)
    data_path = project_root / cfg.data_path
    train_data_name = cfg.train_data_filename
    test_data_name = cfg.test_data_filename
    # test_data_name = cfg.test_data_filename
    # model_path = Path(cfg.model_path)
    model_path = project_root / cfg.model_path
    model_name = cfg.model_name

    num_classes = 6

    # print working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Using device: {DEVICE}")
    print(f"Training configuration: {cfg.training}")
    print(f"Training data path: {data_path / train_data_name}")
    print(f"Test data path: {data_path / test_data_name}")
    print(f"Model will be saved to: {model_path / model_name}")
    #  Initialize and load model
    model = MyAwesomeModel(
        input_dim=12,
        hidden_dims=cfg.training.hidden_dims,
        output_dim=6,
        dropout_rate=cfg.training.dropout_rate,
    ).to(DEVICE)
    model.eval()
    print("Model loaded successfully\n")

    train_set = WineData(data_path / train_data_name, False)
    test_set = WineData(data_path / test_data_name, False)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.training.batch_size, shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.training.batch_size, shuffle=False
    )

    # Get predictions for both sets
    def get_predictions(dataloader):
        predictions = []
        labels = []
        with torch.no_grad():
            for features, label in dataloader:
                features = features.to(DEVICE)
                outputs = model(features)
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
                labels.extend(label.numpy())
        return np.array(predictions), np.array(labels)

    train_preds, train_labels = get_predictions(train_dataloader)
    test_preds, test_labels = get_predictions(test_dataloader)

    # Create  visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Confusion Matrix for  Training Set
    ax1 = plt.subplot(2, 3, 2)
    cm_train = confusion_matrix(train_labels, train_preds)
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix (Training Set)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # 2. Confusion Matrix for Test Set
    ax2 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(test_labels, test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    # # 2. Normalized Confusion Matrix
    # ax2 = plt.subplot(2, 3, 2)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2)
    # ax2.set_title('Normalized Confusion Matrix (Test)', fontsize=14, fontweight='bold')
    # ax2.set_xlabel('Predicted Label')
    # ax2.set_ylabel('True Label')

    # 3. Class Distribution
    ax3 = plt.subplot(2, 3, 3)
    train_dist = np.bincount(train_labels, minlength=num_classes)
    test_dist = np.bincount(test_labels, minlength=num_classes)
    x = np.arange(num_classes)
    width = 0.35
    ax3.bar(x - width / 2, train_dist, width, label="Train", alpha=0.8)
    ax3.bar(x + width / 2, test_dist, width, label="Test", alpha=0.8)
    ax3.set_xlabel("Quality Class")
    ax3.set_ylabel("Count")
    ax3.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # 4. Per-Class Accuracy
    ax4 = plt.subplot(2, 3, 4)
    per_class_acc = []
    for i in range(num_classes):
        mask = test_labels == i
        if mask.sum() > 0:
            acc = (test_preds[mask] == test_labels[mask]).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)

    bars = ax4.bar(range(num_classes), per_class_acc, color="steelblue", alpha=0.8)
    ax4.axhline(
        y=np.mean(per_class_acc),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(per_class_acc):.3f}",
    )
    ax4.set_xlabel("Quality Class")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Per-Class Accuracy (Test Set)", fontsize=14, fontweight="bold")
    ax4.set_xticks(range(num_classes))
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 5. Train vs Test Accuracy Comparison
    ax5 = plt.subplot(2, 3, 5)
    train_acc = (train_preds == train_labels).mean()
    test_acc = (test_preds == test_labels).mean()

    bars = ax5.bar(
        ["Train", "Test"],
        [train_acc, test_acc],
        color=["#2ecc71", "#3498db"],
        alpha=0.8,
    )
    ax5.set_ylabel("Accuracy")
    ax5.set_title("Train vs Test Accuracy", fontsize=14, fontweight="bold")
    ax5.set_ylim([0, 1])
    ax5.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Save figure
    fig.savefig(
        project_root / cfg.figure_path / cfg.figure_visualization, bbox_inches="tight"
    )  # dpi=150,
    print(
        f"Visualizations saved to {project_root / cfg.figure_path / cfg.figure_visualization}"
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Accuracy Gap:   {abs(train_acc - test_acc):.4f}")
    print("\nPer-Class Accuracy (Test):")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.4f}")


if __name__ == "__main__":
    visualize()
