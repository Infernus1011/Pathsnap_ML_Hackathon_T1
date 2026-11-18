import numpy as np


def validate_track(df):
    """
    Validate GPS observations.
    """

    required_columns = [
        "longitude",
        "latitude",
        "timestamp"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing track columns: {missing_columns}"
        )

    result = {
        "rows": len(df),
        "missing_values": df[required_columns].isna().sum().to_dict(),
        "longitude_finite": bool(
            np.isfinite(df["longitude"].dropna()).all()
        ),
        "latitude_finite": bool(
            np.isfinite(df["latitude"].dropna()).all()
        ),
        "timestamp_monotonic": bool(
            df["timestamp"].dropna().is_monotonic_increasing
        ),
        "duplicate_timestamps": int(
            df["timestamp"].duplicated().sum()
        ),
        "invalid_longitude": int(
            (~df["longitude"].between(-180, 180)).sum()
        ),
        "invalid_latitude": int(
            (~df["latitude"].between(-90, 90)).sum()
        )
    }

    return result


def validate_nodes(df):
    """
    Validate road-network nodes.
    """

    required_columns = [
        "longitude",
        "latitude"
    ]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("Invalid node schema.")

    return {
        "number_of_nodes": len(df),
        "missing_values": df.isna().sum().to_dict(),
        "longitude_range": [
            df["longitude"].min(),
            df["longitude"].max()
        ],
        "latitude_range": [
            df["latitude"].min(),
            df["latitude"].max()
        ]
    }


def validate_arcs(arcs, nodes):
    """
    Check whether arc endpoints refer to existing node IDs.
    """

    node_ids = set(nodes.index)

    from_ids = set(arcs["from_node_id"])
    to_ids = set(arcs["to_node_id"])

    invalid_from = from_ids - node_ids
    invalid_to = to_ids - node_ids

    return {
        "number_of_arcs": len(arcs),
        "invalid_from_node_ids": sorted(invalid_from),
        "invalid_to_node_ids": sorted(invalid_to),
        "valid": len(invalid_from) == 0 and len(invalid_to) == 0
    }


def validate_route(route, arcs):
    """
    Check whether reference route arc IDs exist.
    """

    valid_arc_ids = set(arcs.index)

    route_arc_ids = set(route["arc_id"].dropna())

    invalid_ids = route_arc_ids - valid_arc_ids

    return {
        "number_of_route_entries": len(route),
        "invalid_arc_ids": sorted(invalid_ids),
        "valid": len(invalid_ids) == 0
    }
