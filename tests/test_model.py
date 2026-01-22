import torch
<<<<<<< HEAD
from mlops_exam_project.model import WineQualityClassifier


def test_model_initialization():
    """Test WineQualityClassifier initialization."""
    model = WineQualityClassifier()
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "network")
    assert isinstance(model.network, torch.nn.Sequential)


def test_model_forward_pass():
    """Test WineQualityClassifier forward pass."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    x = torch.randn(1, 11)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([1, 6])


def test_model_forward_batch():
    """Test WineQualityClassifier forward pass with batch."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    x = torch.randn(10, 11)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([10, 6])


def test_model_parameters():
    """Test that WineQualityClassifier has trainable parameters."""
    model = WineQualityClassifier()
=======
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
>>>>>>> 4b2eb16 (Code Coverage udvidet, integration test implementeret)
    params = list(model.parameters())
    assert len(params) > 0
    total_params = sum(p.numel() for p in params)
    assert total_params > 0


def test_model_state_dict():
<<<<<<< HEAD
    """Test WineQualityClassifier state dictionary."""
    model = WineQualityClassifier()
    state = model.state_dict()
    assert len(state) > 0
    assert any("network" in key for key in state.keys())
=======
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

>>>>>>> 4b2eb16 (Code Coverage udvidet, integration test implementeret)
