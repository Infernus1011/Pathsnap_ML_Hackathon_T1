
import numpy as np
from scipy.spatial import cKDTree

from pathsnap.data.legacy_loader import load_trip
from pathsnap.data.preprocessing import preprocess_trip
from pathsnap.route.route_alignment import align_gps_to_route

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "src" / "pathsnap" / "data" / "raw" / "train"


def diagnose(trip_id):
    trip = load_trip(TRAIN_DIR, trip_id)
    processed = preprocess_trip(trip)

    gps = np.asarray(processed["gps_xy"])
    nodes = np.asarray(processed["nodes_xy"])

    arcs = processed["arcs"][
        ["source_node", "target_node"]
    ].to_numpy(dtype=int)

    route_ids = processed["route"]["arc_id"].to_numpy(dtype=int)
    route_arcs = arcs[route_ids]

    starts = nodes[route_arcs[:, 0]]
    ends = nodes[route_arcs[:, 1]]

    # Sample points along reference-route segments to
    # efficiently estimate proximity to the route.
    lengths = np.linalg.norm(ends - starts, axis=1)
    samples = []

    for start, end, length in zip(starts, ends, lengths):
        n = max(2, int(np.ceil(length / 5)) + 1)
        ratios = np.linspace(0, 1, n)
        samples.append(
            start + ratios[:, None] * (end - start)
        )

    sampled_route = np.vstack(samples)
    tree = cKDTree(sampled_route)

    nearest_distances, _ = tree.query(gps)
    alignment = align_gps_to_route(processed)
    aligned_distances = alignment["match_distances"]

    print(f"\nTrip {trip_id}")
    print("-" * 45)
    print("GPS observations:", len(gps))
    print("Reference-route arcs:", len(route_ids))

    print(
        "Median nearest-route distance:",
        round(np.median(nearest_distances), 2), "m"
    )
    print(
        "Median aligned distance:",
        round(np.median(aligned_distances), 2), "m"
    )

    print(
        "Nearest-route distance > 50 m:",
        int(np.sum(nearest_distances > 50))
    )
    print(
        "Aligned distance > 50 m:",
        int(np.sum(aligned_distances > 50))
    )


for trip_id in ["00000054", "00000033"]:
    diagnose(trip_id)
