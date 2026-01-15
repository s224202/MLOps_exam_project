import torch, hydra, os
from torch import nn




@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg):
    print(cfg.hyperparameters.batch_size, cfg.hyperparameters.learning_rate)

    # Print kun det, I vil se (ingen config-dump)
"""
cfg.env.paths.data_root
cfg.env.runtime.device
cfg.model.name
cfg.training.epochs
cfg.training.hidden_dims
cfg.batch_size
cfg.learning_rate
"""
class WineQualityClassifier(nn.Module):
    """Feedforward neural network for wine quality classification."""
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: list[int] = [64, 32],
        output_dim: int = 6,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the wine quality classifier.
        Args:
            input_dim: Number of input features (11 for wine dataset)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        # Build the network layers
        layers = []
        prev_dim = input_dim
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        return self.network(x)
    
if __name__ == "__main__":
    # Test the model
    model = WineQualityClassifier(
        input_dim=11,
        hidden_dims=[64, 32, 16],
        output_dim=6,
        dropout_rate=0.3
    )
    # Create a random input
    x = torch.randn(8, 11)  # Batch of 8 samples
    # Forward pass
    output = model(x)
    print(f"Model architecture:\n{model}")
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
