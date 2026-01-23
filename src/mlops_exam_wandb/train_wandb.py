import wandb
from dotenv import load_dotenv

import torch
import hydra
import os
import matplotlib.pyplot as plt
from data import WineData
from model import WineQualityClassifier as MyAwesomeModel
from omegaconf import DictConfig
from pathlib import Path


# # load the .env file to set environment variables for wandb
# load_dotenv()

# WANDB_API_KEY = os.getenv("WANDB_API_KEY")
# WANDB_PROJECT = os.getenv("WANDB_PROJECT", "corrupt_mnist")
# WANDB_ENTITY = os.getenv("WANDB_ENTITY")
# WANDB_JOB_TYPE = os.getenv("WANDB_JOB_TYPE", "training")




DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="config"
)  # , data_path: Path = "data/processed", model_path: Path = "models")
def train(cfg: DictConfig) -> None:
    print("Training day and night")

    #  Hydra changes the working directory to outputs/<date>/<time> for each run. Use an absolute path or make the path relative to the project root.
    #  The issue is that Hydra changes the working directory to outputs/<date>/<time> for each run.  Here we use an absolute path (or we ould make the path relative to the project root).
    project_root = Path(
        __file__
    ).parent.parent.parent  # Getting the project root directory
    
    
    
    # Ensure .env is loaded from project root (Hydra changes cwd)
    load_dotenv(project_root / ".env", override=True)

    wandb_project = os.getenv("WANDB_PROJECT", "mlops_exam_project")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_job_type = os.getenv("WANDB_JOB_TYPE", "training")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not wandb_project:
        raise ValueError("WANDB_PROJECT is not set. Please set it in .env or the environment.")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    
    
    
    # data_path = Path(cfg.data_path)
    data_path = project_root / cfg.data_path
    train_data_name = cfg.train_data_filename
    val_data_name = cfg.val_data_filename
    # test_data_name = cfg.test_data_filename
    # model_path = Path(cfg.model_path)
    model_path = project_root / cfg.model_path
    model_name = cfg.model_name

    # print working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Using device: {DEVICE}")
    print(f"Training configuration: {cfg.training}")
    print(f"Training data path: {data_path / train_data_name}")
    print(f"Validation data path: {data_path / val_data_name}")
    print(f"Model will be saved to: {model_path / model_name}")

    lr=cfg.training.lr
    batch_size=cfg.training.batch_size
    epochs=cfg.training.epochs
    #val_split=cfg.data.val_split
    hidden_dims=cfg.training.hidden_dims
    dropout_rate=cfg.training.dropout_rate



    print(f"{lr=}, {batch_size=}, {epochs=},   {hidden_dims=}, {dropout_rate=}") #  val_split
    wandb.init(
        project=wandb_project, # WANDB_PROJECT,
        entity= wandb_entity, #WANDB_ENTITY,
        job_type=wandb_job_type, # WANDB_JOB_TYPE,
        config={"lr": lr, 
                "batch_size": batch_size, 
                "epochs": epochs, 
                #"val_split": val_split,
                "hidden_dims": hidden_dims,
                "dropout_rate": dropout_rate
                },
    )

    model = MyAwesomeModel(
        input_dim=12,
        hidden_dims=hidden_dims,
        output_dim=6,
        dropout_rate= dropout_rate,
    ).to(DEVICE)
    train_set = WineData(data_path / train_data_name, False)
    val_set = WineData(data_path / val_data_name, False)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size= batch_size
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {
        "train_loss": [],
        "train_accuracy": [],
        "epoch_loss": [],
        "epoch_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for i, (features, target) in enumerate(train_dataloader):
            features, target = features.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(features)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            # Record batch statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            # Accumulate epoch statistics
            epoch_loss += loss.item()
            epoch_correct += (y_pred.argmax(dim=1) == target).sum().item()
            epoch_total += target.size(0)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Calculate epoch averages
        avg_loss = epoch_loss / len(train_dataloader)
        avg_accuracy = epoch_correct / epoch_total
        statistics["epoch_loss"].append(avg_loss)
        statistics["epoch_accuracy"].append(avg_accuracy)

        # do validation every epoch
        model.eval()
        with torch.no_grad():
            val_accuracy = 0
            val_loss = 0
            val_steps = 0
            for features, target in val_dataloader:
                features, target = features.to(DEVICE), target.to(DEVICE)
                y_pred = model(features)
                loss = loss_fn(y_pred, target)
                val_loss += loss.item()
                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                val_accuracy += accuracy
                val_steps += 1
            val_accuracy /= val_steps
            val_loss /= val_steps

            statistics["val_loss"].append(val_loss)
            statistics["val_accuracy"].append(val_accuracy)

            print(
                f"Epoch {epoch}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}"
            )


            # Log metrics to wandb  at the end of each epoch for display in the dashboard
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": avg_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })  




             # add a plot of histogram of the gradients
            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
            wandb.log({"gradients": wandb.Histogram(grads.cpu())})

    print("Training complete")

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), model_path / model_name)
    print(f"Model saved to {model_path / model_name}")




    # log the model as an artifact to wandb
    # final_accuracy = statistics["val_accuracy"][-1]
    # final_loss = statistics["val_loss"][-1]
    final_val_accuracy = statistics["val_accuracy"][-1]
    final_val_loss = statistics["val_loss"][-1]
    #torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="red_wine_quality_model",
        type="model",
        description="A model trained to classify red wine quality",
        metadata={
                "final_val_accuracy": final_val_accuracy, 
                "final_val_loss": final_val_loss, 
                "hidden_dims": hidden_dims,
                "dropout_rate": dropout_rate,
                "input_dim": 12,
                "output_dim": 6,
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                },
        )
    artifact.add_file(str(model_path / model_name))
    #artifact.tag("model")
    #artifact.tag("corruptmnist")
    #artifact.add_tags(["model", "corruptmnist"])
    wandb.log_artifact(artifact)



    # artifact.add_file(str(model_path / model_name))
    # wandb.log_artifact(artifact)
    
    artifact.wait()
    wandb.run.link_artifact(artifact, f"{wandb_entity}/model-registry/red_wine_quality_model")
    print(f"Model logged to registry with final_val_accuracy: {final_val_accuracy}")



    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # axs[0].plot(statistics["train_loss"],  marker='o', color='blue', markersize=2, label='Train Loss')
    axs[0].plot(
        statistics["epoch_loss"],
        marker="o",
        color="blue",
        markersize=2,
        label="Train Loss",
    )
    axs[0].plot(
        statistics["val_loss"],
        marker="o",
        color="orange",
        markersize=2,
        label="Validation Loss",
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_title("Loss")
    axs[0].set_xticks(range(1, cfg.training.epochs))
    axs[0].set_xticklabels(range(1, cfg.training.epochs))
    axs[0].legend()
    # axs[1].plot(statistics["train_accuracy"], marker='o', color='blue', markersize=2, label='Train Accuracy')
    axs[1].plot(
        statistics["epoch_accuracy"],
        marker="o",
        color="blue",
        markersize=2,
        label="Train Accuracy",
    )
    axs[1].plot(
        statistics["val_accuracy"],
        marker="o",
        color="orange",
        markersize=2,
        label="Validation Accuracy",
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Accuracy")
    axs[1].set_xticks(range(1, cfg.training.epochs))
    axs[1].set_xticklabels(range(1, cfg.training.epochs))
    axs[1].legend()

    # fig.savefig(project_root / cfg.figure_path / cfg.figure_training_plot, bbox_inches='tight') # dpi=150,

    # # changes to save in reports/figures instead of figures/; needed for the docker + hydra setup
    # fig_dir = Path(cfg.figure_path)
    # fig_dir.mkdir(parents=True, exist_ok=True)
    # fig.savefig(fig_dir / cfg.figure_training_plot, bbox_inches="tight")


    # img_display = cfg.figure_training_plot

    # # normalize images for better visualization so that they are in [0, 255]
    # #img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) * 255

    # images = [
    #     wandb.Image(img_display, caption="Training and Validation Loss and Accuracy")
    # ]
    # wandb.log({"images": images})

    fig_dir = Path(cfg.figure_path)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / cfg.figure_training_plot, bbox_inches="tight")

    img_path = fig_dir / cfg.figure_training_plot

    images = [
        wandb.Image(str(img_path), caption="Training and Validation Loss and Accuracy")
    ]
    wandb.log({"images": images})
    plt.close()  # close the plot to avoid memory leaks and overlapping figures


    

if __name__ == "__main__":
    train()
    #typer.run(train)
    wandb.finish()
