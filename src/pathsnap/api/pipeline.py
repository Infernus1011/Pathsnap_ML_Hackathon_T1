class PathSnapPipeline:

    def __init__(self):

        self.is_ready = False

    def load(self):

        """
        Load the trained model, preprocessing components,
        road network and route decoder.

        Implement after the ML and route reconstruction
        modules are ready.
        """

        raise NotImplementedError(
            "Prediction pipeline has not been integrated yet."
        )

    def predict(self, trip_data):

        """
        Run the complete prediction pipeline.

        Expected future sequence:

        1. GPS preprocessing
        2. Candidate generation
        3. Feature extraction
        4. ML prediction
        5. HMM/Viterbi decoding
        6. Return predicted arc IDs
        """

        if not self.is_ready:
            raise RuntimeError(
                "PathSnap prediction pipeline is not ready."
            )

        raise NotImplementedError(
            "Connect the trained model and route decoder."
        )
