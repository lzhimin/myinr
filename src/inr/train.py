import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from inr import config
from inr.logger import get_logger

logger = get_logger(__name__)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    epochs: int = None,
    lr: float = None,
    device: str = None,
    checkpoint_dir: str = None,
    save_every: int = 10,
):
    """
    Generic training loop for INR models.

    Args:
        model:          Any nn.Module that accepts coords and returns predictions.
        dataloader:     DataLoader yielding (coords, targets) batches.
        epochs:         Number of training epochs (defaults to config.EPOCHS).
        lr:             Learning rate (defaults to config.LEARNING_RATE).
        device:         Device string (defaults to config.DEVICE).
        checkpoint_dir: Directory to save checkpoints (defaults to config.OUTPUT_DIR).
        save_every:     Save a checkpoint every N epochs.
    """
    epochs = epochs or config.EPOCHS
    lr = lr or config.LEARNING_RATE
    device = device or config.DEVICE
    checkpoint_dir = checkpoint_dir or config.OUTPUT_DIR

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger.info(f"Starting training — epochs={epochs}, lr={lr}, device={device}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for coords, targets in dataloader:
            coords = coords.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(coords)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch}/{epochs}] loss={avg_loss:.6f}")

        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")
    return model
