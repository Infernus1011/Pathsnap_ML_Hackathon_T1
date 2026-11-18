
import numpy as np
from scipy.spatial import cKDTree


class RoadCandidateGenerator:
    """
    Generate nearby road-segment candidates for GPS points.

    Uses sampled points along road segments for KDTree
    lookup, followed by exact point-to-segment distances.
    """

    def __init__(self, nodes_xy, arcs, spacing=25.0):
        self.nodes_xy = np.asarray(nodes_xy, dtype=float)

        if hasattr(arcs, "to_numpy"):
            self.arcs = arcs[
                ["source_node", "target_node"]
            ].to_numpy(dtype=np.int64)
        else:
            self.arcs = np.asarray(arcs, dtype=np.int64)

        self.spacing = float(spacing)

        if self.spacing <= 0:
            raise ValueError("Spacing must be positive")

        self.starts = self.nodes_xy[self.arcs[:, 0]]
        self.ends = self.nodes_xy[self.arcs[:, 1]]

        self.vectors = self.ends - self.starts
        self.length_sq = np.sum(self.vectors ** 2, axis=1)
        self.lengths = np.sqrt(self.length_sq)

        self._build_spatial_index()

    def _build_spatial_index(self):
        """
        Sample points along road segments.

        Every sample retains the ID of its parent arc.
        """
        sample_counts = (
            np.ceil(self.lengths / self.spacing).astype(int) + 1
        )

        sample_counts = np.maximum(sample_counts, 2)

        total_samples = int(sample_counts.sum())

        sample_xy = np.empty(
            (total_samples, 2), dtype=np.float64
        )

        sample_arc_ids = np.empty(
            total_samples, dtype=np.int32
        )

        offset = 0

        for arc_id, count in enumerate(sample_counts):
            fractions = np.linspace(0.0, 1.0, count)

            points = (
                self.starts[arc_id]
                + fractions[:, None] * self.vectors[arc_id]
            )

            sample_xy[offset:offset + count] = points
            sample_arc_ids[offset:offset + count] = arc_id

            offset += count

        self.sample_arc_ids = sample_arc_ids
        self.tree = cKDTree(sample_xy)

        print(
            f"Spatial index built: "
            f"{len(self.arcs):,} arcs, "
            f"{total_samples:,} sampled points"
        )

    def point_to_segments(self, point, arc_ids):
        """
        Compute closest-point distances to selected segments.

        Returns:
            distances: shortest point-to-segment distance
            ratios: clamped projection ratio in [0, 1]
        """
        point = np.asarray(point, dtype=float)
        arc_ids = np.asarray(arc_ids, dtype=np.int64)

        starts = self.starts[arc_ids]
        vectors = self.vectors[arc_ids]
        length_sq = self.length_sq[arc_ids]

        relative = point - starts

        ratios = np.divide(
            np.sum(relative * vectors, axis=1),
            length_sq,
            out=np.zeros(len(arc_ids), dtype=float),
            where=length_sq > 0,
        )

        ratios = np.clip(ratios, 0.0, 1.0)

        projected = starts + ratios[:, None] * vectors

        distances = np.linalg.norm(
            point - projected, axis=1
        )

        return distances, ratios

    def get_candidates(self, point, k=10, search_k=128):
        """
        Retrieve nearby candidate arcs and rank them by
        exact point-to-segment distance.

        Note: sampled KDTree search is approximate.
        """
        if k < 1:
            raise ValueError("k must be at least 1")

        search_k = min(
            max(search_k, k),
            self.tree.n
        )

        _, sample_indices = self.tree.query(
            point, k=search_k
        )

        sample_indices = np.atleast_1d(sample_indices)

        candidate_arc_ids = np.unique(
            self.sample_arc_ids[sample_indices]
        )

        # Expand the query if too few distinct arcs are found.
        while (
            len(candidate_arc_ids) < k
            and search_k < self.tree.n
        ):
            search_k = min(search_k * 2, self.tree.n)

            _, sample_indices = self.tree.query(
                point, k=search_k
            )

            candidate_arc_ids = np.unique(
                self.sample_arc_ids[
                    np.atleast_1d(sample_indices)
                ]
            )

        distances, ratios = self.point_to_segments(
            point, candidate_arc_ids
        )

        # Deterministic ranking: distance first, arc ID second.
        order = np.lexsort(
            (candidate_arc_ids, distances)
        )[:k]

        return {
            "arc_ids": candidate_arc_ids[order],
            "distances": distances[order],
            "proj_ratios": ratios[order],
        }

    def build_feature_frame(
        self,
        gps_xy,
        speed,
        heading,
        trip_id,
        k=10,
        search_k=128,
    ):
        """
        Build candidate-row features for one trip.

        Directed connectivity for observation i is true when the candidate
        source node is the target node of the nearest candidate from i-1.
        The first observation has an explicit value of 0.
        """
        import pandas as pd

        gps_xy = np.asarray(gps_xy, dtype=float)
        speed = np.asarray(speed, dtype=float)
        heading = np.asarray(heading, dtype=float)

        rows = []
        previous_nearest = None

        for obs_id, point in enumerate(gps_xy):
            candidates = self.get_candidates(
                point, k=k, search_k=search_k
            )
            arc_ids = candidates["arc_ids"]

            candidate_starts = self.starts[arc_ids]
            candidate_ends = self.ends[arc_ids]
            candidate_vectors = candidate_ends - candidate_starts
            candidate_lengths = np.sqrt(
                np.sum(candidate_vectors * candidate_vectors, axis=1)
            )

            dist_to_start = np.linalg.norm(
                point - candidate_starts, axis=1
            )
            dist_to_end = np.linalg.norm(
                point - candidate_ends, axis=1
            )

            arc_angles = np.degrees(
                np.arctan2(
                    candidate_vectors[:, 1],
                    candidate_vectors[:, 0],
                )
            )
            vehicle_heading = float(heading[obs_id])
            vehicle_speed = float(speed[obs_id])

            heading_diff = np.empty(len(arc_ids), dtype=float)
            if vehicle_speed <= 1e-6:
                heading_diff[:] = 0.0
            else:
                difference = np.abs(arc_angles - vehicle_heading)
                heading_diff[:] = np.minimum(difference, 360.0 - difference)

            zero_length = candidate_lengths <= 1e-9
            heading_diff[zero_length] = 0.0

            if previous_nearest is None:
                connected = np.zeros(len(arc_ids), dtype=np.int8)
            else:
                prev_end = int(self.arcs[previous_nearest, 1])
                current_sources = self.arcs[arc_ids, 0]
                connected = (
                    current_sources == prev_end
                ).astype(np.int8)

            for rank, arc_id in enumerate(arc_ids):
                rows.append(
                    {
                        "trip_id": str(trip_id),
                        "obs_id": int(obs_id),
                        "candidate_arc_id": int(arc_id),
                        "dist_perp": float(candidates["distances"][rank]),
                        "dist_to_start": float(dist_to_start[rank]),
                        "dist_to_end": float(dist_to_end[rank]),
                        "arc_length": float(candidate_lengths[rank]),
                        "heading_diff": float(heading_diff[rank]),
                        "speed": vehicle_speed,
                        "proj_ratio": float(
                            candidates["proj_ratios"][rank]
                        ),
                        "is_connected_to_prev_nearest": int(
                            connected[rank]
                        ),
                    }
                )

            previous_nearest = int(arc_ids[0])

        return pd.DataFrame(rows)
