"""
PathSnap end-to-end prediction pipeline.

Loads the trained ML model and HMM/Viterbi route decoder at startup,
then processes any GPS track on demand.
"""

from pathlib import Path
import os
import joblib
import pandas as pd

from pathsnap.data.readers import read_track, read_nodes, read_arcs
from pathsnap.data.preprocessing import preprocess_trip
from pathsnap.features.feature_engineering import RoadCandidateGenerator
from pathsnap.route.hmm_decoder import HMMRouteDecoder

# Repo root is 3 levels above this file: src/pathsnap/api/pipeline.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "artifacts" / "final_model.joblib"

FEATURE_COLS = [
    "dist_perp",
    "dist_to_start",
    "dist_to_end",
    "arc_length",
    "heading_diff",
    "speed",
    "proj_ratio",
]

K_CANDIDATES = 10


class PathSnapPipeline:
    """
    End-to-end GPS map-matching pipeline.

    Call load() once at startup, then predict() per request.
    """

    def __init__(self):
        self.is_ready = False
        self.model = None

    # ------------------------------------------------------------------
    def load(self, model_path: str | Path | None = None):
        """
        Load the trained classifier from disk.

        The pipeline is marked ready after a successful load.
        """
        path = Path(
            model_path
            or os.environ.get("PATHSNAP_MODEL_PATH", "")
            or _DEFAULT_MODEL_PATH
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                "Run  python scripts/train_and_save_model.py  to generate it."
            )

        self.model = joblib.load(path)
        self.is_ready = True
        print(f"[PathSnapPipeline] Model loaded from {path}")

    # ------------------------------------------------------------------
    def _resolve_trip_files(self, track_path: Path):
        """
        Given the .track file, locate companion .nodes and .arcs files.
        """
        stem = track_path.stem
        trip_dir = track_path.parent
        nodes_path = trip_dir / f"{stem}.nodes"
        arcs_path = trip_dir / f"{stem}.arcs"

        if not nodes_path.is_file():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
        if not arcs_path.is_file():
            raise FileNotFoundError(f"Arcs file not found: {arcs_path}")

        return nodes_path, arcs_path

    # ------------------------------------------------------------------
    def predict(self, trip_data: Path) -> dict:
        """
        Run the complete prediction pipeline on a GPS track.

        Parameters
        ----------
        trip_data : Path
            Absolute path to the .track file.

        Returns
        -------
        dict with keys:
            predicted_arc_ids : list[int]
            compressed_route  : list[int]
            route_continuity  : float
            metrics           : dict | None
        """
        if not self.is_ready:
            raise RuntimeError(
                "PathSnap prediction pipeline is not ready. Call load() first."
            )

        track_path = Path(trip_data)
        nodes_path, arcs_path = self._resolve_trip_files(track_path)

        # 1. Load raw data
        track_df = read_track(track_path)
        nodes_df = read_nodes(nodes_path)
        arcs_df = read_arcs(arcs_path)  # columns: from_node_id, to_node_id
        trip_id = track_path.stem

        # 2. Preprocess GPS + road nodes into local XY space
        trip_raw = {
            "trip_id": trip_id,
            "track": track_df,
            "nodes": nodes_df,
            "arcs": arcs_df,
            "route": None,
        }
        prep = preprocess_trip(trip_raw)

        # Rename to match RoadCandidateGenerator interface
        arcs_for_gen = arcs_df[["from_node_id", "to_node_id"]].rename(
            columns={"from_node_id": "source_node", "to_node_id": "target_node"}
        )

        # 3. Candidate generation + feature extraction
        generator = RoadCandidateGenerator(
            nodes_xy=prep["nodes_xy"],
            arcs=arcs_for_gen,
        )

        candidate_df = generator.build_feature_frame(
            gps_xy=prep["gps_xy"],
            speed=prep["speed"],
            heading=prep["heading"],
            trip_id=trip_id,
            k=K_CANDIDATES,
        )

        if candidate_df.empty:
            return {
                "predicted_arc_ids": [],
                "compressed_route": [],
                "route_continuity": 0.0,
                "metrics": None,
            }

        # 4. HMM / Viterbi decoding
        decoder = HMMRouteDecoder(arcs_df=arcs_for_gen)

        predicted_arcs, compressed_route, route_continuity = decoder.decode(
            candidate_df=candidate_df,
            model=self.model,
        )

        return {
            "predicted_arc_ids": [int(a) for a in predicted_arcs],
            "compressed_route": compressed_route,
            "route_continuity": float(route_continuity),
            "metrics": {
                "route_continuity": float(route_continuity),
            },
        }
