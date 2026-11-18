from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = [
    ".track",
    ".nodes",
    ".arcs",
    ".route"
]


def discover_dataset_files(root):
    """
    Recursively discover supported dataset files.
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}"
        )

    files = []

    for path in root.rglob("*"):

        if path.is_file() and path.suffix in SUPPORTED_EXTENSIONS:

            files.append({
                "partition_or_folder": path.parent,
                "relative_path": str(path.relative_to(root)),
                "file_name": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size
            })

    return pd.DataFrame(files)


def summarize_inventory(inventory):
    """
    Generate a compact summary of discovered files.
    """

    if inventory.empty:
        return {
            "total_files": 0,
            "file_types": {},
            "total_size_bytes": 0
        }

    return {
        "total_files": len(inventory),
        "file_types": inventory["extension"].value_counts().to_dict(),
        "total_size_bytes": int(
            inventory["size_bytes"].sum()
        )
    }
