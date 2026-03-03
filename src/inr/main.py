import torch
from torch.utils.data import DataLoader

from inr import config
from inr.logger import get_logger
from inr.models import SIREN
from inr.dataset import ScalarFieldDataset
from inr.train import train

logger = get_logger(__name__)


def main():
    logger.info(f"Data dir:   {config.DATA_DIR}")
    logger.info(f"Output dir: {config.OUTPUT_DIR}")
    logger.info(f"Device:     {config.DEVICE}")

    torch.manual_seed(config.SEED)

    # Example: fit a SIREN to a random scalar field
    coords = torch.rand(8192, 3)
    values = torch.rand(8192, 1)
    dataset = ScalarFieldDataset(coords, values)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    model = SIREN(in_features=3, out_features=1)
    train(model, dataloader)


if __name__ == "__main__":
    main()
