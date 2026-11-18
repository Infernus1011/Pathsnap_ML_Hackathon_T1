import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import get_data_root
from pathsnap.data.inventory import (
    discover_dataset_files,
    summarize_inventory
)


def main():

    data_root = get_data_root()

    print("=" * 60)
    print("PATHSNAP DATASET INVENTORY")
    print("=" * 60)

    print(f"\nDataset root: {data_root}")

    inventory = discover_dataset_files(data_root)

    if inventory.empty:
        print("\nNo supported dataset files discovered.")
        print("Check the dataset directory and its structure.")
        return

    print("\nFile inventory:")
    print(
        inventory.to_string(index=False)
    )

    summary = summarize_inventory(inventory)

    print("\nSummary:")
    print(summary)

    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "dataset_inventory.csv"

    inventory.to_csv(output_file, index=False)

    print(f"\nInventory saved to: {output_file}")


if __name__ == "__main__":
    main()
