import os
from dotenv import load_dotenv

load_dotenv()

def main():
    data_dir = os.getenv("DATA_DIR", "data/")
    output_dir = os.getenv("OUTPUT_DIR", "outputs/")
    device = os.getenv("DEVICE", "cpu")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device:     {device}")
    print(f"Log level:  {log_level}")

if __name__ == "__main__":
    main()
