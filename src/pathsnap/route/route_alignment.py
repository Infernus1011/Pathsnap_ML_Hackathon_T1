
import numpy as np


def align_gps_to_route(processed, jump_penalty=2.0):
    """
    Align GPS observations to ordered reference-route arcs.

    Uses monotonic dynamic programming:
      - route positions may stay the same or advance
      - backward movement is prohibited
      - forward jumps are allowed, with a penalty

    Returns one route occurrence for each GPS observation.

    This is an initial alignment, not yet validated ground truth.
    """

    gps_xy = np.asarray(processed["gps_xy"], dtype=float)
    nodes_xy = np.asarray(processed["nodes_xy"], dtype=float)

    arcs = processed["arcs"][
        ["source_node", "target_node"]
    ].to_numpy(dtype=np.int64)

    route_ids = processed["route"][
        "arc_id"
    ].to_numpy(dtype=np.int64)

    route_arcs = arcs[route_ids]

    starts = nodes_xy[route_arcs[:, 0]]
    ends = nodes_xy[route_arcs[:, 1]]

    vectors = ends - starts
    lengths_sq = np.sum(vectors * vectors, axis=1)

    n_obs = len(gps_xy)
    n_route = len(route_ids)

    if n_obs == 0 or n_route == 0:
        raise ValueError("GPS and route must be nonempty")

    # int32 backpointers: approximately 4 bytes per DP cell.
    backpointers = np.zeros(
        (n_obs, n_route), dtype=np.int32
    )

    previous_cost = np.full(
        n_route, np.inf, dtype=np.float64
    )

    # Allow alignment to begin anywhere on the reference route.
    previous_cost[:] = 0.0

    route_positions = np.arange(n_route, dtype=np.float64)

    for obs_id, point in enumerate(gps_xy):

        # Exact GPS-to-route-segment distances.
        relative = point - starts

        ratios = np.divide(
            np.sum(relative * vectors, axis=1),
            lengths_sq,
            out=np.zeros(n_route, dtype=float),
            where=lengths_sq > 0,
        )

        ratios = np.clip(ratios, 0.0, 1.0)

        projected = starts + ratios[:, None] * vectors

        distances = np.linalg.norm(
            point - projected, axis=1
        )

        # Transition cost from previous route position i
        # to current route position j:
        #
        # previous_cost[i] + jump_penalty * (j - i)
        #
        # Valid only when i <= j.
        adjusted = (
            previous_cost - jump_penalty * route_positions
        )

        prefix_best_cost = np.minimum.accumulate(adjusted)

        # Track the best previous route position for each j.
        prefix_best_index = np.empty(n_route, dtype=np.int32)

        best_index = 0
        best_value = adjusted[0]

        for j in range(n_route):
            if adjusted[j] < best_value:
                best_value = adjusted[j]
                best_index = j

            prefix_best_index[j] = best_index

        current_cost = (
            distances
            + prefix_best_cost
            + jump_penalty * route_positions
        )

        backpointers[obs_id] = prefix_best_index
        previous_cost = current_cost

    # Backtrack from the minimum-cost final route position.
    alignment = np.empty(n_obs, dtype=np.int32)

    alignment[-1] = int(np.argmin(previous_cost))

    for obs_id in range(n_obs - 1, 0, -1):
        alignment[obs_id - 1] = backpointers[
            obs_id, alignment[obs_id]
        ]

    # Calculate distances for the chosen alignment.
    chosen_starts = starts[alignment]
    chosen_vectors = vectors[alignment]
    chosen_lengths_sq = lengths_sq[alignment]

    relative = gps_xy - chosen_starts

    ratios = np.divide(
        np.sum(relative * chosen_vectors, axis=1),
        chosen_lengths_sq,
        out=np.zeros(n_obs, dtype=float),
        where=chosen_lengths_sq > 0,
    )

    ratios = np.clip(ratios, 0.0, 1.0)

    projected = (
        chosen_starts + ratios[:, None] * chosen_vectors
    )

    match_distances = np.linalg.norm(
        gps_xy - projected, axis=1
    )

    return {
        "route_positions": alignment,
        "matched_arc_ids": route_ids[alignment],
        "match_distances": match_distances,
        "projection_ratios": ratios,
        "total_cost": float(np.min(previous_cost)),
    }
