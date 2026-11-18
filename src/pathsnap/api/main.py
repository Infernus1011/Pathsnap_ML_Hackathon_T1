from pathlib import Path
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathsnap.api.schemas import (
    HealthResponse,
    TripListResponse,
    PredictionRequest,
    PredictionResponse,
    EvaluationMetrics,
)

from pathsnap.api.trip_registry import TripRegistry
from pathsnap.api.pipeline import PathSnapPipeline
from dotenv import load_dotenv

# Repository root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(PROJECT_ROOT / ".env")

# Dataset directory
DATA_ROOT = PROJECT_ROOT / "src" / "pathsnap" / "data"

# Evaluator trip manifest
MANIFEST_PATH = DATA_ROOT / "evaluator_trip_manifest.csv"

registry = TripRegistry(
    manifest_path=MANIFEST_PATH,
    data_root=DATA_ROOT,
)
pipeline = PathSnapPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once when the server starts."""
    try:
        pipeline.load()
        print("[startup] PathSnap pipeline ready.")
    except FileNotFoundError as exc:
        print(f"[startup] WARNING: {exc}")
        print("[startup] API will return 503 for /predict until the model is generated.")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction",
    version="0.1.0",
    lifespan=lifespan,
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
        # get_trip returns an absolute Path to the .track file
        track_path = registry.get_trip(request.trip_id)

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

    if not track_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="GPS track file not found."
        )

    if not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction pipeline is not ready. Run train_and_save_model.py first."
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

    # Build EvaluationMetrics from result dict if available
    metrics = None
    if result.get("metrics"):
        metrics = EvaluationMetrics(
            route_continuity=result["metrics"].get("route_continuity")
        )

    return PredictionResponse(
        trip_id=request.trip_id,
        predicted_arc_ids=result["predicted_arc_ids"],
        compressed_route=result.get("compressed_route", []),
        route_continuity=result.get("route_continuity", 0.0),
        metrics=metrics,
        processing_time_seconds=elapsed,
        message="Prediction completed successfully."
    )
