from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import typer
from tqdm import tqdm
import random

from data import FruitVegDataset
from model import FruitClassifier


def train(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("models"),
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    small_dataset: bool = False,
    save_model: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Train the model.
    
    Args:
        data_dir: Directory containing the processed data
        output_dir: Directory to save the model
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        small_dataset: If True, use only 10 samples per class for quick testing
        save_model: Set to False if you do not wish to save the model
        device: Device to train on
    """
    # Setup
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    train_dataset = FruitVegDataset(data_dir / "train")
    val_dataset = FruitVegDataset(data_dir / "val")
    
    if small_dataset:
        # Get indices for each class
        train_healthy_idx = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 1]
        train_rotten_idx = [i for i, (_, label) in enumerate(train_dataset.samples) if label == 0]
        val_healthy_idx = [i for i, (_, label) in enumerate(val_dataset.samples) if label == 1]
        val_rotten_idx = [i for i, (_, label) in enumerate(val_dataset.samples) if label == 0]
        
        # Randomly select 10 samples per class
        random.seed(42)
        train_indices = (random.sample(train_healthy_idx, 10) + 
                        random.sample(train_rotten_idx, 10))
        val_indices = (random.sample(val_healthy_idx, 5) + 
                      random.sample(val_rotten_idx, 5))
        
        # Create subset datasets
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        print(f"Debug mode: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Rest of the training code remains the same...
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = FruitClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Train Accuracy: {100*train_correct/train_total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Accuracy: {100*val_correct/val_total:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_model:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                print("Saved best model!")

if __name__ == "__main__":
    """
    Example usage for small dataset and no save model:

    python src/healthy_vs_rotten/train.py --small-dataset --no-save-model 
    """
    typer.run(train)