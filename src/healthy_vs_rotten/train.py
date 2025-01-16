from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import random
import numpy as np
import wandb
from dotenv import load_dotenv
import os

from data import FruitVegDataset
from model import FruitClassifier

def setup_wandb(cfg: DictConfig) -> None:
    """Setup Weights & Biases configuration from environment variables."""
    # Load environment variables
    if not load_dotenv():
        raise FileNotFoundError(
            ".env file not found. Please copy .env.template to .env and fill in your values."
        )
    
    # Check for required environment variables
    required_vars = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file."
        )
    
    # Login to wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))

project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
config_path = str(project_root / "configs")

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def train(cfg: DictConfig) -> None:
    """Train the model using Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup directories
    data_dir = Path(cfg.paths.data_dir)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Specify device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup W&B
    setup_wandb(cfg)
    
    # Initialize W&B run
    config = {
        "learning_rate": cfg.training.learning_rate,
        "batch_size": cfg.training.batch_size,
        "num_epochs": cfg.training.num_epochs,
        "small_dataset": cfg.training.small_dataset,
        "small_dataset_proportion": cfg.training.small_dataset_proportion,
        "device": device,
        "optimizer": "AdamW",
        "architecture": "FruitClassifier",
        "model": dict(cfg.model)
    }
    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config,
        mode=cfg.wandb.mode
    )
    
    # Load datasets
    train_dataset = FruitVegDataset(data_dir / "train")
    val_dataset = FruitVegDataset(data_dir / "val")
    
    if cfg.training.small_dataset:
        # Get indices for each class
        train_healthy_idx = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 1]
        train_rotten_idx = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 0]
        val_healthy_idx = [i for i, (_, label) in enumerate(val_dataset.samples) if label == 1]
        val_rotten_idx = [i for i, (_, label) in enumerate(val_dataset.samples) if label == 0]
        
        # Randomly select samples based on proportion
        random.seed(42)
        train_indices = (random.sample(train_healthy_idx, int(np.floor(cfg.training.small_dataset_proportion*len(train_healthy_idx)))) + 
                        random.sample(train_rotten_idx, int(np.floor(cfg.training.small_dataset_proportion*len(train_rotten_idx)))))
        val_indices = (random.sample(val_healthy_idx, int(np.floor(cfg.training.small_dataset_proportion*len(val_healthy_idx)))) + 
                      random.sample(val_rotten_idx, int(np.floor(cfg.training.small_dataset_proportion*len(val_rotten_idx)))))
        
        # Create subset datasets
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        print(f"Debug mode: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
        wandb.config.update({"train_samples": len(train_dataset), "val_samples": len(val_dataset)})
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    
    # Initialize model and watch gradients
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    wandb.watch(model, log="all", log_freq=10)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(cfg.training.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")):
            images = images.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Log training metrics every 10 steps
            if step % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_accuracy": (predictions == labels).float().mean().item(),
                    "epoch": epoch
                })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Log metrics to W&B
        wandb.log({
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_acc,
            "val/epoch_loss": val_loss,
            "val/epoch_accuracy": val_acc,
            "epoch": epoch
        })
        
        # Create and log confusion matrix
        wandb.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_predictions,
                class_names=["Rotten", "Healthy"]
            )
        })
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{cfg.training.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if cfg.training.save_model:
                model_path = output_dir / "best_model.pt"
                torch.save(model.state_dict(), model_path)
                print("Saved best model!")
                # Log best model to W&B
                artifact = wandb.Artifact(
                    name=f"model-{wandb.run.id}",
                    type="model",
                    description=f"Best model from run {wandb.run.name}"
                )
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    train()