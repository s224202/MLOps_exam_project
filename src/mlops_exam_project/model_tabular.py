import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig

# Patch torch.load to work with PyTorch 2.6.0 and omegaconf
import torch.serialization as serialization
from omegaconf import DictConfig, ListConfig

# Add omegaconf classes to safe globals for PyTorch 2.6.0
serialization.add_safe_globals([DictConfig, ListConfig])

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Encode quality labels to 0-indexed classes for classification
le = LabelEncoder()
df["quality"] = le.fit_transform(df["quality"])

# Train/validation/test split
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["quality"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["quality"])

# Data configuration
data_config = DataConfig(
    target=["quality"],
    continuous_cols=[c for c in df.columns if c != "quality"],
    categorical_cols=[],
)

# Model configuration
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="128-64",
    activation="ReLU",
    dropout=0.1,
)

# Trainer and optimizer with checkpointing and early stopping
trainer_config = TrainerConfig(
    max_epochs=50,
    batch_size=64,
    checkpoints="valid_loss",  # Save best model based on validation loss
    early_stopping="valid_loss",  # Early stopping based on validation loss
    early_stopping_patience=5,  # Stop if no improvement for 5 epochs
    trainer_kwargs={"enable_progress_bar": False},
)
optimizer_config = OptimizerConfig()

# Create model
tab_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

# Train
tab_model.fit(
    train=train,
    validation=val,
)

# Evaluate
print("\nTest set evaluation:")
print(tab_model.evaluate(test))

# Make predictions
preds = tab_model.predict(test)
print("\nSample predictions:")
print(preds.head())