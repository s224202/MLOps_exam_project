from mlops_exam_project.train import train
from pathlib import Path
import pytest


def test_train():
    """Test the train function."""
    train(Path("data/raw/WineQT.csv"))
    # If it runs without raising an exception, the test passes


def test_train_with_default_path(capsys):
    """Test the train function with default parameters."""
    train()
    captured = capsys.readouterr()
    assert "Dataset size:" in captured.out
    assert "Model parameters:" in captured.out
    assert "Model state:" in captured.out


def test_train_function_logic(capsys):
    """Test the train function produces expected output."""
    data_path = Path("data/raw/WineQT.csv")
    train(data_path)
    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    assert len(output_lines) >= 3
    assert any("Dataset size:" in line for line in output_lines)
