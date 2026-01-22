import torch
from mlops_exam_project.model import Model


def test_model_initialization():
    """Test Model initialization."""
    model = Model()
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "layer")
    assert isinstance(model.layer, torch.nn.Linear)


def test_model_forward_pass():
    """Test Model forward pass."""
    model = Model()
    x = torch.rand(1)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([1])


def test_model_forward_batch():
    """Test Model forward pass with batch."""
    model = Model()
    x = torch.rand(10, 1)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([10, 1])


def test_model_parameters():
    """Test that Model has trainable parameters."""
    model = Model()
    params = list(model.parameters())
    assert len(params) > 0
    total_params = sum(p.numel() for p in params)
    assert total_params > 0


def test_model_state_dict():
    """Test Model state dictionary."""
    model = Model()
    state = model.state_dict()
    assert "layer.weight" in state
    assert "layer.bias" in state


def test_model_main_execution(capsys):
    """Test the main execution block of model.py."""
    model = Model()
    x = torch.rand(1)
    result = model(x)
    print(f"Output shape of model: {result.shape}")
    captured = capsys.readouterr()
    assert "Output shape of model:" in captured.out

