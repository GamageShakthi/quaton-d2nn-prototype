"""
Training loop for D²NN optical classifiers.

Provides a Trainer class that handles:
    - Standard cross-entropy training for classification.
    - Automatic logging of loss and accuracy metrics.
    - Checkpoint saving and loading.
    - Support for both continuous and quantization-aware training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from typing import Dict, List, Optional, Tuple


class Trainer:
    """
    Training engine for D²NN classifiers.

    Args:
        model: The D²NN model to train.
        device: Torch device (cpu or cuda).
        lr: Learning rate.
        weight_decay: L2 regularization weight.
        checkpoint_dir: Directory for saving checkpoints.

    Attributes:
        history (dict): Training metrics keyed by epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        lr: float = 5e-3,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer — Adam works well for phase-mask optimization
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            (avg_loss, accuracy) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.

        Args:
            val_loader: Validation/test data loader.

        Returns:
            (avg_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 20,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation and optional checkpointing.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            n_epochs: Number of training epochs.
            save_best: If True, save the best model by val accuracy.
            verbose: Print per-epoch summaries.

        Returns:
            Training history dictionary.
        """
        best_val_acc = 0.0

        for epoch in range(1, n_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Step scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{n_epochs} │ "
                    f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
                    f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} │ "
                    f"LR: {current_lr:.2e}"
                )

            # Checkpoint
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint("best_model.pt", epoch, val_acc)

        # Save final model
        self.save_checkpoint("final_model.pt", n_epochs, val_acc)

        return self.history

    def save_checkpoint(
        self, filename: str, epoch: int, val_acc: float
    ) -> str:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename.
            epoch: Current epoch number.
            val_acc: Current validation accuracy.

        Returns:
            Full path to saved checkpoint.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "history": self.history,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        return checkpoint

    def save_history(self, path: str) -> None:
        """Save training history to JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
