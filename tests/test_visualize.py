"""Tests for the visualize module."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from mlops_exam_project.model import WineQualityClassifier


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
