from pathlib import Path
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.api.schemas import (
    HealthResponse,
    TripListResponse,
    PredictionRequest,
    PredictionResponse
)

from pathsnap.api.trip_registry import TripRegistry
from pathsnap.api.pipeline import PathSnapPipeline


# Repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Dataset directory
DATA_ROOT = PROJECT_ROOT / "src" / "pathsnap" / "data"

# Evaluator trip manifest
MANIFEST_PATH = DATA_ROOT / "evaluator_trip_manifest.csv"

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction",
    version="0.1.0"
)

# Configure the frontend origin in deployment.
allowed_origins = os.getenv(
    "PATHSNAP_ALLOWED_ORIGINS",
    "http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

registry = TripRegistry(
    manifest_path=MANIFEST_PATH,
    data_root=DATA_ROOT,
)
pipeline = PathSnapPipeline()


@app.get("/", tags=["General"])
def root():

    return {
        "message": "Welcome to PathSnap API",
        "documentation": "/docs"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"]
)
def health():

    return HealthResponse(
        status="running",
        service="PathSnap API",
        version="0.1.0"
    )


@app.get(
    "/api/v1/trips",
    response_model=TripListResponse,
    tags=["Trips"]
)
def get_trips():

    try:
        trips = registry.list_trip_ids()

        return TripListResponse(
            trips=trips,
            total=len(trips)
        )

    except (FileNotFoundError, ValueError) as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(request: PredictionRequest):

    try:
        trip = registry.get_trip(request.trip_id)

    except KeyError:

        raise HTTPException(
            status_code=404,
            detail="Trip not found."
        )

    except (FileNotFoundError, ValueError) as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    track_path = Path(trip["track_path"])

    if not track_path.is_absolute():
        track_path = PROJECT_ROOT / track_path

    if not track_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="GPS track file not found."
        )

    if not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction pipeline is not integrated yet."
        )

    start_time = time.perf_counter()

    try:

        result = pipeline.predict(
            trip_data=track_path
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )

    elapsed = time.perf_counter() - start_time

    return PredictionResponse(
        trip_id=request.trip_id,
        predicted_arc_ids=result["predicted_arc_ids"],
        metrics=result.get("metrics"),
        processing_time_seconds=elapsed,
        message="Prediction completed successfully."
    )
