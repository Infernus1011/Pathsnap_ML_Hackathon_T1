
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pathsnap.data.legacy_loader import load_trip
from pathsnap.data.preprocessing import preprocess_trip

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "src" / "pathsnap" / "data" / "raw" / "train"
OUTPUT_DIR = ROOT / "reports" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_trip(trip_id):
    processed = preprocess_trip(load_trip(TRAIN_DIR, trip_id))

    gps = np.asarray(processed["gps_xy"])
    nodes = np.asarray(processed["nodes_xy"])

    arcs = processed["arcs"][
        ["source_node", "target_node"]
    ].to_numpy(dtype=int)

    route_ids = processed["route"]["arc_id"].to_numpy(dtype=int)
    route_arcs = arcs[route_ids]

    fig, ax = plt.subplots(figsize=(11, 11))

    # Draw reference-route road segments.
    for source, target in route_arcs:
        segment = nodes[[source, target]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color="blue",
            linewidth=1.5,
            alpha=0.8,
        )

    # Draw GPS trajectory.
    ax.plot(
        gps[:, 0],
        gps[:, 1],
        color="red",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.scatter(
        gps[0, 0],
        gps[0, 1],
        color="green",
        s=70,
        marker="o",
        label="GPS start",
        zorder=5,
    )

    ax.scatter(
        gps[-1, 0],
        gps[-1, 1],
        color="black",
        s=70,
        marker="x",
        label="GPS end",
        zorder=5,
    )

    ax.plot([], [], color="blue", label="Reference route")
    ax.plot([], [], color="red", label="GPS trajectory")

    ax.set_title(f"GPS vs Reference Route — Trip {trip_id}")
    ax.set_xlabel("Local X coordinate (m)")
    ax.set_ylabel("Local Y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()

    output = OUTPUT_DIR / f"trip_{trip_id}_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


for trip_id in ["00000054", "00000033"]:
    plot_trip(trip_id)
