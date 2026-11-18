from pathlib import Path

import pandas as pd


def read_track(file_path):
    """
    Read GPS observations.

    Expected columns:
    longitude, latitude, timestamp
    """

    columns = [
        "longitude",
        "latitude",
        "timestamp"
    ]

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns
    )


def read_nodes(file_path):
    """
    Read road-network nodes.

    Row index is the node ID.
    """

    columns = [
        "longitude",
        "latitude"
    ]

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns
    ).rename_axis("node_id")


def read_arcs(file_path):
    """
    Read road-network connections.

    Row index is the arc ID.
    """

    columns = [
        "from_node_id",
        "to_node_id"
    ]

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns
    ).rename_axis("arc_id")


def read_route(file_path):
    """
    Read reference route.

    Each row contains one reference arc ID.
    """

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["arc_id"]
    )

    return df
