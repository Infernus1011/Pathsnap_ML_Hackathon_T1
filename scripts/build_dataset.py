import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.data.legacy_loader import list_trips, load_trip
from pathsnap.features.feature_engineering import RoadCandidateGenerator
from pathsnap.data.preprocessing import preprocess_trip
from pathsnap.route.route_alignment import align_gps_to_route


TRAIN_FEATURE_COLUMNS = [
    "trip_id",
    "obs_id",
    "candidate_arc_id",
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio",
    "is_connected_to_prev_nearest",
]


def build_feature_schema():
    return {
        "columns": {
            "trip_id": {"type": "string", "description": "Eight-digit trip identifier"},
            "obs_id": {"type": "int32", "description": "Zero-based GPS observation index"},
            "candidate_arc_id": {"type": "int32", "description": "Zero-based road arc index within the trip"},
            "dist_perp": {"type": "float64", "unit": "metres", "description": "Shortest point-to-finite-segment distance"},
            "dist_to_start": {"type": "float64", "unit": "metres", "description": "Distance to the directed arc start node"},
            "dist_to_end": {"type": "float64", "unit": "metres", "description": "Distance to the directed arc end node"},
            "arc_length": {"type": "float64", "unit": "metres", "description": "Euclidean road-segment length"},
            "heading_diff": {"type": "float64", "unit": "degrees", "range": [0, 180], "description": "Smallest circular difference from vehicle heading; stationary and zero-length arcs use 0"},
            "speed": {"type": "float64", "unit": "metres per second", "description": "GPS-derived speed; first observation is 0"},
            "proj_ratio": {"type": "float64", "range": [0, 1], "description": "Clamped projection ratio onto the arc"},
            "is_connected_to_prev_nearest": {"type": "int8", "range": [0, 1], "description": "1 only when candidate source is the previous nearest candidate's target; first observation is 0"},
            "is_match": {"type": "int8", "range": [0, 1], "description": "Training label for the aligned reference-route arc"},
        },
        "candidate_count": 10,
        "heading_convention": "0=east, 90=north, 180=west, 270=south",
        "timestamp_units": "track timestamp differences are interpreted as seconds",
        "confidence_policy": {
            "alignment_max_distance_metres": 50.0,
            "alignment_max_forward_jump_positions": 10,
            "missing_candidate_policy": "exclude observation",
            "large_distance_policy": "exclude observation",
            "implausible_jump_policy": "exclude observation",
        },
        "label_definition": "is_match=1 only for the candidate equal to the GPS observation's monotonically aligned route occurrence arc",
    }


def _trip_alignment_metrics(processed):
    result = align_gps_to_route(processed)
    positions = result["route_positions"]
    jumps = np.abs(np.diff(positions))
    return result, jumps


def build_one_training_trip(processed, k=10, search_k=128):
    generator = RoadCandidateGenerator(
        processed["nodes_xy"],
        processed["arcs"],
        spacing=25.0,
    )
    frame = generator.build_feature_frame(
        gps_xy=processed["gps_xy"],
        speed=processed["speed"],
        heading=processed["heading"],
        trip_id=processed["trip_id"],
        k=k,
        search_k=search_k,
    )

    alignment, jumps = _trip_alignment_metrics(processed)
    route_positions = alignment["route_positions"]
    matched_arc_ids = alignment["matched_arc_ids"]
    distances = alignment["match_distances"]

    uncertainty = np.ones(len(processed["gps_xy"]), dtype=bool)
    uncertain_reason = np.full(len(processed["gps_xy"]), "", dtype=object)
    observation_stats = []

    candidate_sets = (
        frame.groupby("obs_id", sort=False)["candidate_arc_id"]
        .apply(lambda values: set(values.astype(int)))
    )

    for obs_id in range(len(processed["gps_xy"])):
        forward_jump = 0 if obs_id == 0 else int(jumps[obs_id - 1])
        candidate_ids = candidate_sets.loc[obs_id]
        matched_arc_id = int(matched_arc_ids[obs_id])
        distance = float(distances[obs_id])
        covered = matched_arc_id in candidate_ids

        reasons = []
        if not covered:
            reasons.append("matched_arc_missing_from_candidates")
        if distance > 50.0:
            reasons.append("large_alignment_distance")
        if forward_jump > 10:
            reasons.append("implausible_route_jump")

        uncertainty[obs_id] = not reasons
        uncertain_reason[obs_id] = ";".join(reasons)
        observation_stats.append(
            {
                "trip_id": processed["trip_id"],
                "obs_id": obs_id,
                "alignment_route_position": int(route_positions[obs_id]),
                "matched_arc_id": matched_arc_id,
                "alignment_distance": distance,
                "route_position_jump": forward_jump,
                "matched_arc_in_candidates": covered,
                "excluded": bool(reasons),
                "reason": ";".join(reasons),
            }
        )

    labels = np.zeros(len(frame), dtype=np.int8)
    uncertain_obs = set(np.flatnonzero(~uncertainty))
    kept_mask = ~frame["obs_id"].isin(uncertain_obs)
    frame = frame.loc[kept_mask].copy()

    positive_mask = np.zeros(len(frame), dtype=bool)
    obs_ids = frame["obs_id"].to_numpy(dtype=np.int32)
    candidate_ids = frame["candidate_arc_id"].to_numpy(dtype=np.int32)
    positive_mask = np.array(
        [
            candidate_id == int(matched_arc_ids[obs_id])
            for obs_id, candidate_id in zip(obs_ids, candidate_ids)
        ],
        dtype=bool,
    )
    labels = positive_mask.astype(np.int8)
    frame["is_match"] = labels

    diagnostics = pd.DataFrame(observation_stats)
    return frame, diagnostics


def build_one_student_trip(processed, k=10, search_k=128):
    generator = RoadCandidateGenerator(
        processed["nodes_xy"],
        processed["arcs"],
        spacing=25.0,
    )
    return generator.build_feature_frame(
        gps_xy=processed["gps_xy"],
        speed=processed["speed"],
        heading=processed["heading"],
        trip_id=processed["trip_id"],
        k=k,
        search_k=search_k,
    )


def measure_candidate_recall(processed, k=10, sample_size=20):
    generator = RoadCandidateGenerator(
        processed["nodes_xy"],
        processed["arcs"],
        spacing=25.0,
    )
    rng = np.random.default_rng(2026)
    indices = rng.choice(
        len(processed["gps_xy"]),
        size=min(sample_size, len(processed["gps_xy"])),
        replace=False,
    )
    exact_top10 = set()
    recall_count = 0
    for obs_id in indices:
        point = processed["gps_xy"][obs_id]
        candidates = generator.get_candidates(point, k=k)
        exact_distances, _ = generator.point_to_segments(
            point, np.arange(len(processed["arcs"]), dtype=np.int64)
        )
        exact_ids = np.lexsort(
            (np.arange(len(exact_distances)), exact_distances)
        )[:k]
        recall_count += len(set(exact_ids).intersection(candidates["arc_ids"]))
        exact_top10.update(exact_ids)
    return recall_count / (len(indices) * k)


def build_datasets(root, train_limit=None, test_limit=None, k=10):
    root = Path(root)
    data_dir = root / "src" / "pathsnap" / "data"
    train_dir = data_dir / "raw" / "train"
    test_dir = data_dir / "raw" / "student_test"
    output_dir = data_dir / "processed"
    reports_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_frames = []
    train_diagnostics = []
    test_frames = []
    recall_scores = []
    alignment_distances = []
    trip_counts = {"train": 0, "student_test": 0}
    observation_counts = {"train": 0, "student_test": 0}
    started = time.perf_counter()

    train_ids = list_trips(train_dir)
    if train_limit is not None:
        train_ids = train_ids[:train_limit]

    for trip_id in train_ids:
        trip = load_trip(train_dir, trip_id)
        processed = preprocess_trip(trip)
        trip_counts["train"] += 1
        observation_counts["train"] += len(processed["gps_xy"])

        frame, diagnostics = build_one_training_trip(processed, k=k)
        train_frames.append(frame)
        train_diagnostics.append(diagnostics)
        alignment_distances.extend(
            diagnostics["alignment_distance"].tolist()
        )

        if len(recall_scores) < 5:
            recall_scores.append(measure_candidate_recall(processed, k=k))

    test_ids = list_trips(test_dir)
    if test_limit is not None:
        test_ids = test_ids[:test_limit]

    for trip_id in test_ids:
        trip = load_trip(test_dir, trip_id)
        processed = preprocess_trip(trip)
        trip_counts["student_test"] += 1
        observation_counts["student_test"] += len(processed["gps_xy"])
        test_frames.append(build_one_student_trip(processed, k=k))

    train = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame(
        columns=TRAIN_FEATURE_COLUMNS + ["is_match"]
    )
    test = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame(
        columns=TRAIN_FEATURE_COLUMNS
    )
    diagnostics = (
        pd.concat(train_diagnostics, ignore_index=True)
        if train_diagnostics
        else pd.DataFrame()
    )

    train["trip_id"] = train["trip_id"].astype("string")
    test["trip_id"] = test["trip_id"].astype("string")
    train["obs_id"] = train["obs_id"].astype("int32")
    test["obs_id"] = test["obs_id"].astype("int32")
    train["candidate_arc_id"] = train["candidate_arc_id"].astype("int32")
    test["candidate_arc_id"] = test["candidate_arc_id"].astype("int32")
    train["is_match"] = train["is_match"].astype("int8")
    for column in [
        "dist_perp", "dist_to_start", "dist_to_end",
        "arc_length", "heading_diff", "speed", "proj_ratio",
    ]:
        train[column] = train[column].astype("float64")
        test[column] = test[column].astype("float64")
    train["is_connected_to_prev_nearest"] = train[
        "is_connected_to_prev_nearest"
    ].astype("int8")
    test["is_connected_to_prev_nearest"] = test[
        "is_connected_to_prev_nearest"
    ].astype("int8")

    train.to_parquet(output_dir / "train_candidates.parquet", index=False)
    test.to_parquet(output_dir / "student_test_candidates.parquet", index=False)
    if not diagnostics.empty:
        diagnostics.to_csv(
            output_dir / "training_alignment_diagnostics.csv",
            index=False,
        )

    (output_dir / "feature_schema.json").write_text(
        json.dumps(build_feature_schema(), indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    alignment = np.asarray(alignment_distances, dtype=float)
    excluded = int(diagnostics["excluded"].sum()) if not diagnostics.empty else 0
    candidate_rows = len(train) + len(test)

    report = [
        "# Member 1 Validation Report",
        "",
        "## Processing summary",
        f"- Training trips processed: {trip_counts['train']} of {len(train_ids)} selected",
        f"- Student-test trips processed: {trip_counts['student_test']} of {len(test_ids)} selected",
        f"- Training GPS observations: {observation_counts['train']}",
        f"- Student-test GPS observations: {observation_counts['student_test']}",
        f"- Exported candidate rows: {candidate_rows}",
        f"- Training candidate rows: {len(train)}",
        f"- Student-test candidate rows: {len(test)}",
        f"- Runtime: {elapsed:.2f} seconds",
    ]

    if not diagnostics.empty:
        report.extend(
            [
                "",
                "## Alignment and labels",
                f"- Mean alignment distance: {np.mean(alignment):.2f} m",
                f"- Median alignment distance: {np.median(alignment):.2f} m",
                f"- 95th percentile alignment distance: {np.percentile(alignment, 95):.2f} m",
                f"- Maximum alignment distance: {np.max(alignment):.2f} m",
                f"- Observations excluded/uncertain: {excluded}",
                f"- Positive labels: {int(train['is_match'].sum())}",
                f"- Negative labels: {int((train['is_match'] == 0).sum())}",
                f"- Missing matched arc in candidate set: {int((~diagnostics['matched_arc_in_candidates']).sum())}",
                f"- Alignment distance > 50 m: {int((diagnostics['alignment_distance'] > 50).sum())}",
                f"- Route-position jump > 10: {int((diagnostics['route_position_jump'] > 10).sum())}",
            ]
        )

    report.extend(
        [
            "",
            "## Candidate retrieval",
            f"- Exact-brute-force top-10 recall on sampled observations: {recall * 100:.2f}%",
            "- Candidate recall was sampled across up to five training trips; the KDTree can miss exact global neighbours.",
            "",
            "## Data assumptions",
            "- Timestamp differences are interpreted as seconds.",
            "- Stationary observations have heading_diff=0.",
            "- Zero-length arcs have heading_diff=0.",
            "- Student-test route files are never used to construct inference features.",
            "- The reference-route alignment is inferred supervision, not independently verified ground truth.",
            "- Directed connectivity uses candidate.source == previous nearest candidate.target.",
            "- Observations missing the aligned arc or exceeding the confidence policy are excluded.",
        ]
    )

    (reports_dir / "member1_validation.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    return {
        "train": train,
        "student_test": test,
        "diagnostics": diagnostics,
        "candidate_recall": recall,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build PathSnap Member 1 candidate datasets"
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    result = build_datasets(
        args.root,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        k=args.k,
    )
    print(
        f"Built {len(result['train'])} training rows and "
        f"{len(result['student_test'])} student-test rows"
    )


if __name__ == "__main__":
    main()
