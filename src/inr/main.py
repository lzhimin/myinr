from inr import config
from inr.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info(f"Data dir:   {config.DATA_DIR}")
    logger.info(f"Output dir: {config.OUTPUT_DIR}")
    logger.info(f"Device:     {config.DEVICE}")
    logger.info(f"Log level:  {config.LOG_LEVEL}")

if __name__ == "__main__":
    main()
