"""
Train the best-performing classifier and save it to artifacts/final_model.joblib.

Run from the repo root:
    python scripts/train_and_save_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import joblib

from pathsnap.data.readers import read_track, read_nodes, read_arcs
from pathsnap.data.preprocessing import preprocess_trip
from pathsnap.features.feature_engineering import RoadCandidateGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_ROOT   = Path(__file__).resolve().parents[1] / "src" / "pathsnap" / "data"
RAW_ROOT    = DATA_ROOT / "raw" / "student_test"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "final_model.joblib"

FEATURE_COLS = [
    "dist_perp", "dist_to_start", "dist_to_end",
    "arc_length", "heading_diff", "speed", "proj_ratio",
]

K_CANDIDATES = 10


def build_features_for_trip(trip_dir: Path) -> pd.DataFrame:
    tid = trip_dir.name
    track_df = read_track(trip_dir / f"{tid}.track")
    nodes_df = read_nodes(trip_dir / f"{tid}.nodes")
    arcs_df  = read_arcs(trip_dir  / f"{tid}.arcs")

    trip_raw = {"trip_id": tid, "track": track_df, "nodes": nodes_df,
                "arcs": arcs_df, "route": None}
    prep = preprocess_trip(trip_raw)

    arcs_for_gen = arcs_df[["from_node_id", "to_node_id"]].rename(
        columns={"from_node_id": "source_node", "to_node_id": "target_node"}
    )

    gen = RoadCandidateGenerator(nodes_xy=prep["nodes_xy"], arcs=arcs_for_gen)
    df  = gen.build_feature_frame(
        gps_xy=prep["gps_xy"], speed=prep["speed"], heading=prep["heading"],
        trip_id=tid, k=K_CANDIDATES,
    )

    route_path = trip_dir / f"{tid}.route"
    if route_path.exists():
        route_arcs = set(
            pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()
        )
        df["label"] = df["candidate_arc_id"].isin(route_arcs).astype(int)
    else:
        nearest = df.groupby("obs_id")["dist_perp"].idxmin()
        df["label"] = 0
        df.loc[nearest, "label"] = 1

    return df


def main():
    trip_dirs = sorted(RAW_ROOT.iterdir())
    if not trip_dirs:
        print(f"No trip directories found under {RAW_ROOT}")
        sys.exit(1)

    print(f"Found {len(trip_dirs)} trips. Building features ...")
    frames = []
    for td in trip_dirs:
        print(f"  {td.name} ...", end=" ", flush=True)
        try:
            df = build_features_for_trip(td)
            frames.append(df)
            print(f"{len(df):,} rows, {df['label'].sum()} positives")
        except Exception as exc:
            print(f"SKIP ({exc})")

    if not frames:
        print("No data built. Aborting.")
        sys.exit(1)

    full_df = pd.concat(frames, ignore_index=True)
    X = full_df[FEATURE_COLS].values
    y = full_df["label"].values
    print(f"\nTotal: {len(full_df):,} rows | Positives: {y.sum()}")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )

    scores = cross_val_score(model, X, y, cv=3, scoring="f1", n_jobs=-1)
    print(f"CV F1 (3-fold): {scores.mean():.4f} +/- {scores.std():.4f}")

    model.fit(X, y)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"\n[OK] Model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
