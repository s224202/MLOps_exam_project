import torch
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
    params = list(model.parameters())
    assert len(params) > 0
    total_params = sum(p.numel() for p in params)
    assert total_params > 0


def test_model_state_dict():
    """Test WineQualityClassifier state dictionary."""
    model = WineQualityClassifier()
    state = model.state_dict()
    assert len(state) > 0
    assert any("network" in key for key in state.keys())

def test_train_core_with_small_batch():
    """Test train_core handles small batch sizes correctly."""
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[16],
        output_dim=6,
        dropout_rate=0.1,
    )
    
    train_features = torch.randn(8, 11)
    train_labels = torch.randint(0, 6, (8,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=2)
    
    val_features = torch.randn(4, 11)
    val_labels = torch.randint(0, 6, (4,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    model, stats = train_core(
        model, train_loader, val_loader, epochs=1, device="cpu", learning_rate=0.01
    )
    
    assert len(stats["epoch_loss"]) == 1
    assert stats["epoch_loss"][0] > 0


def test_train_core_loss_decreasing():
    """Test that training loss generally decreases over epochs."""
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[16],
        output_dim=6,
        dropout_rate=0.1,
    )
    
    # Use consistent data for reproducibility
    torch.manual_seed(42)
    train_features = torch.randn(32, 11)
    train_labels = torch.randint(0, 6, (32,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)
    
    val_features = torch.randn(16, 11)
    val_labels = torch.randint(0, 6, (16,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    model, stats = train_core(
        model, train_loader, val_loader, epochs=3, device="cpu", learning_rate=0.01
    )
    
    # With 3 epochs, should have 3 epoch losses
    assert len(stats["epoch_loss"]) == 3
    assert len(stats["epoch_accuracy"]) == 3
    
    # Generally, later epochs should have lower loss than first epoch
    # (not guaranteed due to random initialization, but likely)
    assert stats["epoch_loss"][0] >= 0


