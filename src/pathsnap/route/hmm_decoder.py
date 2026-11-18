import math
import numpy as np

class HMMRouteDecoder:
    """
    Hidden Markov Model (HMM) Viterbi Decoder for map matching vehicle trajectories.
    
    Combines:
    - Emission probabilities from the ML classifier's candidate scores
    - Transition probabilities from the road network topology (.arcs)
    """

    def __init__(self, arcs_df, p_same=0.50, p_connected=0.45, p_disconnected=1e-5):
        """
        arcs_df: DataFrame with columns ['source_node', 'target_node'] indexed by arc_id.
        """
        self.arcs = arcs_df
        # Normalize column names if needed
        if "from_node_id" in arcs_df.columns:
            self.sources = arcs_df["from_node_id"].to_numpy()
            self.targets = arcs_df["to_node_id"].to_numpy()
        elif "source_node" in arcs_df.columns:
            self.sources = arcs_df["source_node"].to_numpy()
            self.targets = arcs_df["target_node"].to_numpy()
        else:
            self.sources = arcs_df.iloc[:, 0].to_numpy()
            self.targets = arcs_df.iloc[:, 1].to_numpy()

        self.log_p_same = math.log(max(p_same, 1e-12))
        self.log_p_connected = math.log(max(p_connected, 1e-12))
        self.log_p_disconnected = math.log(max(p_disconnected, 1e-12))

    def compute_log_transition(self, prev_arc, curr_arc):
        """
        Calculates log transition probability between two candidate road segments.
        """
        if prev_arc == curr_arc:
            return self.log_p_same
        # Check if prev_arc connects directly to curr_arc in road graph
        if self.targets[prev_arc] == self.sources[curr_arc]:
            return self.log_p_connected
        return self.log_p_disconnected

    def decode(self, candidate_df, model):
        """
        Runs Viterbi decoding over candidate_df ordered by obs_id.
        
        Returns:
            predicted_arcs_per_obs: list of arc_id for each observation
            compressed_route: list of unique sequential arc_ids (route sequence)
            route_continuity: percentage of valid topological transitions
        """
        feature_cols = [
            "dist_perp",
            "dist_to_start",
            "dist_to_end",
            "arc_length",
            "heading_diff",
            "speed",
            "proj_ratio"
        ]

        # Predict match probabilities using the trained classifier
        X = candidate_df[feature_cols].values
        probs = model.predict_proba(X)[:, 1]
        candidate_df = candidate_df.copy()
        candidate_df["model_score"] = np.clip(probs, 1e-6, 1.0)

        # Group candidates by observation
        obs_groups = candidate_df.groupby("obs_id")
        obs_ids = sorted(obs_groups.groups.keys())
        T = len(obs_ids)

        if T == 0:
            return [], [], 0.0

        # Candidate arcs and emission scores per step
        step_arcs = []
        step_emissions = []
        for obs_id in obs_ids:
            group = obs_groups.get_group(obs_id)
            c_arcs = group["candidate_arc_id"].values
            scores = group["model_score"].values
            # Softmax / normalize emissions across candidates at this step
            sum_s = np.sum(scores)
            norm_scores = scores / sum_s if sum_s > 0 else np.ones(len(scores)) / len(scores)
            step_arcs.append(c_arcs)
            step_emissions.append(np.log(np.maximum(norm_scores, 1e-12)))

        # Viterbi DP
        K0 = len(step_arcs[0])
        V = [step_emissions[0].copy()]
        backpointers = []

        for t in range(1, T):
            prev_arcs = step_arcs[t - 1]
            curr_arcs = step_arcs[t]
            K_prev = len(prev_arcs)
            K_curr = len(curr_arcs)

            v_curr = np.zeros(K_curr)
            bp_curr = np.zeros(K_curr, dtype=int)

            for j in range(K_curr):
                curr_a = curr_arcs[j]
                # Vectorized or computed transitions from all prev states
                log_trans = np.array([
                    self.compute_log_transition(prev_arcs[i], curr_a)
                    for i in range(K_prev)
                ])
                total_probs = V[t - 1] + log_trans
                best_prev_idx = np.argmax(total_probs)
                bp_curr[j] = best_prev_idx
                v_curr[j] = total_probs[best_prev_idx] + step_emissions[t][j]

            V.append(v_curr)
            backpointers.append(bp_curr)

        # Backtracking
        best_last_idx = np.argmax(V[-1])
        optimal_indices = [best_last_idx]

        for t in range(T - 2, -1, -1):
            prev_best = backpointers[t][optimal_indices[-1]]
            optimal_indices.append(prev_best)

        optimal_indices.reverse()
        predicted_arcs_per_obs = [step_arcs[t][optimal_indices[t]] for t in range(T)]

        # Calculate route continuity
        connected_transitions = 0
        total_transitions = max(1, T - 1)
        for t in range(T - 1):
            a1 = predicted_arcs_per_obs[t]
            a2 = predicted_arcs_per_obs[t + 1]
            if a1 == a2 or self.targets[a1] == self.sources[a2]:
                connected_transitions += 1

        route_continuity = connected_transitions / total_transitions

        # Compress to traversed path sequence (remove consecutive duplicates)
        compressed_route = []
        for a in predicted_arcs_per_obs:
            if not compressed_route or compressed_route[-1] != a:
                compressed_route.append(int(a))

        return predicted_arcs_per_obs, compressed_route, route_continuity
