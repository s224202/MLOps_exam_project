"""Tests for the visualize module."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from mlops_exam_project.model import WineQualityClassifier
from mlops_exam_project.visualize import get_model_predictions, create_evaluation_visualizations


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32],
        output_dim=6,
        dropout_rate=0.2,
    )


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing visualization."""
    np.random.seed(42)
    predictions = np.random.randint(0, 6, 100)
    labels = np.random.randint(0, 6, 100)
    return predictions, labels


@pytest.fixture
def sample_training_stats():
    """Create sample training statistics."""
    return {
        "train_loss": [0.5, 0.45, 0.4, 0.35, 0.3],
        "train_accuracy": [0.6, 0.65, 0.7, 0.75, 0.8],
        "val_loss": [0.55, 0.5, 0.45, 0.4, 0.35],
        "val_accuracy": [0.58, 0.63, 0.68, 0.73, 0.78],
    }


def test_confusion_matrix_visualization(sample_predictions):
    """Test that confusion matrix can be created."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    predictions, labels = sample_predictions
    cm = confusion_matrix(labels, predictions)
    
    assert cm.shape == (6, 6)
    assert np.all(cm >= 0)


def test_training_loss_plot_data(sample_training_stats):
    """Test that training loss data can be plotted."""
    stats = sample_training_stats
    
    train_loss = stats["train_loss"]
    val_loss = stats["val_loss"]
    
    assert len(train_loss) == len(val_loss)
    assert all(isinstance(x, float) for x in train_loss)


def test_training_accuracy_plot_data(sample_training_stats):
    """Test that training accuracy data can be plotted."""
    stats = sample_training_stats
    
    train_acc = stats["train_accuracy"]
    val_acc = stats["val_accuracy"]
    
    assert len(train_acc) == len(val_acc)
    assert all(0 <= x <= 1 for x in train_acc)
    assert all(0 <= x <= 1 for x in val_acc)


def test_plot_figure_creation():
    """Test creating a matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot dummy data
    epochs = list(range(5))
    loss = [0.5, 0.45, 0.4, 0.35, 0.3]
    
    ax.plot(epochs, loss, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    
    assert fig is not None
    assert len(fig.axes) == 1
    plt.close(fig)


def test_subplots_creation():
    """Test creating subplots for multiple metrics."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = list(range(5))
    loss = [0.5, 0.45, 0.4, 0.35, 0.3]
    accuracy = [0.6, 0.65, 0.7, 0.75, 0.8]
    
    axs[0].plot(epochs, loss, marker="o")
    axs[0].set_title("Loss")
    
    axs[1].plot(epochs, accuracy, marker="o")
    axs[1].set_title("Accuracy")
    
    assert len(fig.axes) == 2
    plt.close(fig)


def test_figure_save_to_file(tmp_path):
    """Test saving figure to file."""
    fig, ax = plt.subplots(1, 1)
    ax.plot([1, 2, 3], [1, 2, 3])
    
    save_path = tmp_path / "test_figure.png"
    fig.savefig(save_path)
    
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_multiple_figures():
    """Test creating multiple figures for different visualizations."""
    figs = []
    
    for i in range(3):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [i, i+1, i+2])
        figs.append(fig)
    
    assert len(figs) == 3
    
    for fig in figs:
        plt.close(fig)


def test_confusion_matrix_heatmap():
    """Test creating a heatmap from confusion matrix."""
    import seaborn as sns
    
    cm = np.array([
        [10, 2, 1, 0, 0, 0],
        [1, 9, 2, 0, 0, 0],
        [0, 1, 10, 1, 0, 0],
        [0, 0, 1, 9, 2, 0],
        [0, 0, 0, 2, 9, 1],
        [0, 0, 0, 0, 1, 10],
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    
    assert fig is not None
    plt.close(fig)


def test_scatter_plot_visualization():
    """Test creating scatter plots for feature visualization."""
    np.random.seed(42)
    
    # Create dummy 2D data
    x = np.random.randn(100)
    y = np.random.randn(100)
    colors = np.random.randint(0, 6, 100)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=colors, cmap="viridis")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax)
    
    assert fig is not None
    plt.close(fig)


def test_histogram_visualization():
    """Test creating histograms for data distribution."""
    data = np.random.normal(loc=0.5, scale=0.2, size=1000)
    
    fig, ax = plt.subplots()
    ax.hist(data, bins=30, edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Data Distribution")
    
    assert fig is not None
    plt.close(fig)


def test_model_predictions_for_visualization(sample_model):
    """Test getting model predictions for visualization (like visualize.py does)."""
    sample_model.eval()
    
    # Create test data
    test_features = torch.randn(20, 11)
    test_labels = torch.randint(0, 6, (20,))
    
    # Get predictions (like visualize.py does)
    with torch.no_grad():
        outputs = sample_model(test_features)
        predictions = outputs.argmax(dim=1).numpy()
    
    # Verify predictions are valid
    assert len(predictions) == 20
    assert all(0 <= p < 6 for p in predictions)


def test_training_statistics_plotting():
    """Test plotting training statistics (like visualize.py does)."""
    # Simulate training statistics like visualize.py would have
    stats = {
        "epoch_loss": [0.5, 0.45, 0.4, 0.35, 0.3],
        "epoch_accuracy": [0.6, 0.65, 0.7, 0.75, 0.8],
        "val_loss": [0.55, 0.5, 0.45, 0.4, 0.35],
        "val_accuracy": [0.58, 0.63, 0.68, 0.73, 0.78],
    }
    
    # Create figure like visualize.py does
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(len(stats["epoch_loss"]))
    
    # Plot loss
    axs[0].plot(epochs, stats["epoch_loss"], marker="o", label="Train Loss")
    axs[0].plot(epochs, stats["val_loss"], marker="o", label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    
    # Plot accuracy
    axs[1].plot(epochs, stats["epoch_accuracy"], marker="o", label="Train Accuracy")
    axs[1].plot(epochs, stats["val_accuracy"], marker="o", label="Val Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    
    assert len(fig.axes) == 2
    plt.close(fig)


def test_confusion_matrix_for_visualization():
    """Test creating confusion matrix visualization (like visualize.py does)."""
    import seaborn as sns
    
    # Simulate predictions and labels
    np.random.seed(42)
    true_labels = np.random.randint(0, 6, 100)
    predictions = np.random.randint(0, 6, 100)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Create visualization like visualize.py does
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    
    assert fig is not None
    assert cm.shape == (6, 6)
    plt.close(fig)


def test_figure_saving_workflow(tmp_path):
    """Test complete figure saving workflow (like visualize.py does)."""
    # Create figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 3, 5])
    ax.set_title("Test Plot")
    
    # Save like visualize.py does
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(exist_ok=True)
    save_path = fig_dir / "test_plot.png"
    
    fig.savefig(save_path, bbox_inches="tight")
    
    # Verify
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close(fig)


def test_hydra_config_paths_for_visualization():
    """Test path handling for visualization (like visualize.py does)."""
    from pathlib import Path
    
    # Simulate what visualize.py does with paths
    project_root = Path(__file__).parent.parent.parent
    
    # Simulate config values
    figure_path = "reports/figures"
    
    # Construct paths like visualize.py does
    fig_dir = Path(figure_path)
    
    # Verify path can be created
    assert isinstance(fig_dir, Path)
    assert "figures" in str(fig_dir)


def test_get_model_predictions_function(sample_model):
    """Test get_model_predictions helper function."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create test data
    test_features = torch.randn(32, 11)
    test_labels = torch.randint(0, 6, (32,))
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Get predictions from helper function
    predictions, labels = get_model_predictions(sample_model, test_loader, device="cpu")
    
    # Verify output
    assert len(predictions) == 32
    assert len(labels) == 32
    assert predictions.dtype == np.int64
    assert labels.dtype == np.int64
    assert all(0 <= p < 6 for p in predictions)
    assert all(0 <= l < 6 for l in labels)


def test_create_evaluation_visualizations_function():
    """Test create_evaluation_visualizations helper function."""
    # Create sample data
    np.random.seed(42)
    train_preds = np.random.randint(0, 6, 100)
    train_labels = np.random.randint(0, 6, 100)
    test_preds = np.random.randint(0, 6, 50)
    test_labels = np.random.randint(0, 6, 50)
    
    # Create visualizations
    fig, train_acc, test_acc, per_class_acc = create_evaluation_visualizations(
        train_preds, train_labels, test_preds, test_labels, num_classes=6
    )
    
    # Verify figure
    assert fig is not None
    assert isinstance(fig, plt.Figure)
    
    # Verify accuracy values
    assert 0 <= train_acc <= 1
    assert 0 <= test_acc <= 1
    assert len(per_class_acc) == 6
    assert all(0 <= acc <= 1 for acc in per_class_acc)
    
    # Verify subplots (includes colorbars, so 7 axes instead of 5)
    assert len(fig.axes) >= 5
    
    plt.close(fig)


def test_get_model_predictions_batch_consistency(sample_model):
    """Test that get_model_predictions handles different batch sizes correctly."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create test data
    test_features = torch.randn(16, 11)
    test_labels = torch.randint(0, 6, (16,))
    test_dataset = TensorDataset(test_features, test_labels)
    
    # Test with different batch sizes
    loader_batch4 = DataLoader(test_dataset, batch_size=4, shuffle=False)
    loader_batch8 = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    preds_b4, labels_b4 = get_model_predictions(sample_model, loader_batch4, device="cpu")
    preds_b8, labels_b8 = get_model_predictions(sample_model, loader_batch8, device="cpu")
    
    # Labels should be in same order
    assert np.array_equal(labels_b4, labels_b8)
    # Predictions should have same length
    assert len(preds_b4) == len(preds_b8) == 16


def test_evaluation_visualizations_accuracy_calculations():
    """Test accuracy calculations in visualizations."""
    # Create perfect predictions
    perfect_preds = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    perfect_labels = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    
    test_preds = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    test_labels = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    
    fig, train_acc, test_acc, per_class_acc = create_evaluation_visualizations(
        perfect_preds, perfect_labels, test_preds, test_labels, num_classes=6
    )
    
    # With perfect predictions, accuracy should be 1.0
    assert train_acc == 1.0
    assert test_acc == 1.0
    assert all(acc == 1.0 for acc in per_class_acc)
    
    plt.close(fig)


def test_evaluation_visualizations_with_imbalanced_data():
    """Test visualizations with imbalanced class distribution."""
    # Create imbalanced data (more class 0, less class 5)
    train_preds = np.concatenate([np.zeros(50, dtype=int), np.ones(30, dtype=int), np.full(20, 2, dtype=int)])
    train_labels = np.concatenate([np.zeros(50, dtype=int), np.ones(30, dtype=int), np.full(20, 2, dtype=int)])
    
    test_preds = np.concatenate([np.zeros(25, dtype=int), np.ones(15, dtype=int), np.full(10, 2, dtype=int)])
    test_labels = np.concatenate([np.zeros(25, dtype=int), np.ones(15, dtype=int), np.full(10, 2, dtype=int)])
    
    fig, train_acc, test_acc, per_class_acc = create_evaluation_visualizations(
        train_preds, train_labels, test_preds, test_labels, num_classes=6
    )
    
    # Verify it still works with imbalanced data
    assert 0 <= train_acc <= 1
    assert 0 <= test_acc <= 1
    assert len(fig.axes) >= 5
    
    plt.close(fig)
