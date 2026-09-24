import sys
from src.config import load_config
from src.execution import execute


def main():
    config_path = sys.argv[1]
    config = load_config(config_path)
    execute(config)


if __name__ == "__main__":
    main()