import os
from dotenv import load_dotenv

load_dotenv()

# Data
DATA_DIR = os.getenv("DATA_DIR", "data/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")

# Device
DEVICE = os.getenv("DEVICE", "cpu")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))

# Training
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 1e-4))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 100))
SEED = int(os.getenv("SEED", 42))

# Logging
LOG_DIR = os.getenv("LOG_DIR", "logs/")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
