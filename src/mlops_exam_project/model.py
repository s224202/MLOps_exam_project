import torch, hydra, os
from torch import nn
from omegaconf import DictConfig
from dataclasses import dataclass

@dataclass
class ModelConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    hidden_dims: int
    dropout_rate: float

    def summary(self):
        return (f"Learning rate: {self.learning_rate}, "
                f"Batch size: {self.batch_size}, "
                f"Epochs: {self.epochs}, "
                f"Hidden_dims: {self.hidden_dims}, "
                f"Dropout_rate: {self.dropout_rate}, "
                )


class WineQualityClassifier(nn.Module):
    """Feedforward neural network for wine quality classification."""
    @hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
    def __init__(
        self,input_dim: int,output_dim: int,    
        cfg:DictConfig,
        ):
        """
        Initialize the wine quality classifier.
        Args:
            input_dim: Number of input features (11 for wine dataset)
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes
            dropout_rate: Dropout probability
        """
        try: 
            model_cfg: ModelConfig = hydra.utils.instantiate(cfg.model)
            super().__init__()
            self.input_dim = input_dim  
            self.hidden_dims = cfg.training.hidden_dims
            self.output_dim = output_dim
            self.dropout_rate = cfg.training.dropout_rate
            # Build the network layers
            layers = []
            prev_dim = input_dim
            # Add hidden layers
            for hidden_dim in range(1,cfg.training.hidden_dims):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(cfg.training.dropout_rate))
                prev_dim = hidden_dim
            # Add output layer
            layers.append(nn.Linear(prev_dim, cfg.training.output_dim))
            self.network = nn.Sequential(*layers)
        except Exception as e:
            print(f"Error loading configuration with Hydra: {e}")

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
        input_dim=12,
        output_dim=10
        
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
