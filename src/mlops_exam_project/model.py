import torch
from torch import nn
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
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
        # Add hidden layer
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
    print(output.shape)  # Should be (8, 6)

