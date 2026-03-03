from inr import config

def main():
    print(f"Data dir:   {config.DATA_DIR}")
    print(f"Output dir: {config.OUTPUT_DIR}")
    print(f"Device:     {config.DEVICE}")
    print(f"Log level:  {config.LOG_LEVEL}")

if __name__ == "__main__":
    main()
