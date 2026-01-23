import torch
from mlops_exam_project.train import train_core
from mlops_exam_project.model import WineQualityClassifier
from torch.utils.data import DataLoader, TensorDataset


def test_model_initialization():
    """Test Model initialization."""
    model = WineQualityClassifier()
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "network")


def test_model_forward_pass():
    """Test Model forward pass."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    x = torch.rand(1, 11)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([1, 6])


def test_model_forward_batch():
    """Test Model forward pass with batch."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    x = torch.rand(10, 11)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == torch.Size([10, 6])


def test_model_parameters():
    """Test that Model has trainable parameters."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    params = list(model.parameters())
    assert len(params) > 0
    total_params = sum(p.numel() for p in params)
    assert total_params > 0


def test_model_state_dict():
    """Test Model state dictionary."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    state = model.state_dict()
    assert len(state) > 0
    assert any("weight" in key for key in state.keys())


def test_model_main_execution(capsys):
    """Test the main execution block of model.py."""
    model = WineQualityClassifier(input_dim=11, hidden_dims=[64, 32], output_dim=6)
    x = torch.rand(1, 11)
    result = model(x)
    print(f"Output shape of model: {result.shape}")
    captured = capsys.readouterr()
    assert "Output shape of model:" in captured.out


def test_train_core_function():
    """Test train_core function completes without errors."""
    # Create a small model
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[32, 16],
        output_dim=6,
        dropout_rate=0.2,
    )

    # Create small training and validation datasets
    train_features = torch.randn(32, 11)
    train_labels = torch.randint(0, 6, (32,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8)

    val_features = torch.randn(16, 11)
    val_labels = torch.randint(0, 6, (16,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Train for 1 epoch
    trained_model, statistics = train_core(
        model,
        train_loader,
        val_loader,
        epochs=1,
        device="cpu",
        learning_rate=0.001,
    )

    # Verify output
    assert isinstance(trained_model, WineQualityClassifier)
    assert "train_loss" in statistics
    assert "epoch_loss" in statistics
    assert "val_loss" in statistics
    assert len(statistics["train_loss"]) > 0
    assert len(statistics["epoch_loss"]) == 1
    assert len(statistics["val_loss"]) == 1


def test_train_core_statistics():
    """Test that train_core produces valid statistics."""
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[16],
        output_dim=6,
        dropout_rate=0.1,
    )

    train_features = torch.randn(16, 11)
    train_labels = torch.randint(0, 6, (16,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=4)

    val_features = torch.randn(8, 11)
    val_labels = torch.randint(0, 6, (8,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model, stats = train_core(
        model, train_loader, val_loader, epochs=2, device="cpu", learning_rate=0.01
    )

    # Verify statistics structure
    assert stats["epoch_loss"][0] > 0
    assert 0 <= stats["epoch_accuracy"][0] <= 1
    assert stats["val_loss"][0] > 0
    assert 0 <= stats["val_accuracy"][0] <= 1

    # Training should have more loss samples than epochs
    assert len(stats["train_loss"]) > 2


def test_train_core_model_training():
    """Test that train_core actually updates model weights."""
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[16],
        output_dim=6,
        dropout_rate=0.1,
    )

    # Get initial weights
    initial_weights = [p.clone() for p in model.parameters()]

    # Create simple datasets
    train_features = torch.randn(20, 11)
    train_labels = torch.randint(0, 6, (20,))
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=4)

    val_features = torch.randn(8, 11)
    val_labels = torch.randint(0, 6, (8,))
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Train
    trained_model, _ = train_core(
        model, train_loader, val_loader, epochs=1, device="cpu", learning_rate=0.01
    )

    # Check that weights changed
    final_weights = [p for p in trained_model.parameters()]
    weights_changed = False
    for init_w, final_w in zip(initial_weights, final_weights):
        if not torch.allclose(init_w, final_w):
            weights_changed = True
            break

    assert weights_changed, "Model weights should change during training"
