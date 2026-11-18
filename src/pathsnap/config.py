from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


def load_config(config_path=None):
    """
    Load YAML project configuration.
    """

    if config_path is None:
        config_path = os.getenv(
            "PATHSNAP_CONFIG",
            "config/config.yaml"
        )

    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def get_data_root():
    """
    Return the configured raw dataset location.
    """

    root = os.getenv("PATHSNAP_DATA_ROOT", "data/raw")

    path = Path(root)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path
