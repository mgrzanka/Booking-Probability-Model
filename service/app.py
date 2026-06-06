import csv
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

from model_utils import MODEL_A_NAME, MODEL_B_NAME, ModelBundle, extract_feature_dict
from schemas import PredictRequest, PredictResponse

app = FastAPI(title="Nocarz Reservation Prediction")
models = ModelBundle()

LOG_DIR = Path(__file__).resolve().parent / "ab_logs"
LOG_PATH = LOG_DIR / "predictions_log.csv"

LOG_FIELDS = [
    "timestamp",
    "request_id",
    "user_id",
    "listing_id",
    "model_variant",
    "model_name",
    "booking_probability",
    "days_since_last_review",
    "alltime_review_count",
    "reviews_in_last_90_days",
    "price",
    "property_type",
    "room_type",
    "Y",
]


def assign_variant(user_id: int) -> str:
    """przydział A/B po user_id (50/50)."""
    digest = hashlib.md5(str(user_id).encode()).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "B"


def ensure_log_file() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()


def log_prediction(
    request_id: str,
    payload: PredictRequest,
    model_variant: str,
    model_name: str,
    booking_probability: float,
) -> None:
    ensure_log_file()
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "user_id": payload.user_id,
        "listing_id": payload.listing_id,
        "model_variant": model_variant,
        "model_name": model_name,
        "booking_probability": booking_probability,
        "days_since_last_review": payload.days_since_last_review,
        "alltime_review_count": payload.alltime_review_count,
        "reviews_in_last_90_days": payload.reviews_in_last_90_days,
        "price": payload.price,
        "property_type": payload.property_type,
        "room_type": payload.room_type,
        "Y": payload.Y,
    }
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    request_id = str(uuid.uuid4())
    model_variant = assign_variant(payload.user_id)
    features = extract_feature_dict(payload.model_dump())

    if model_variant == "A":
        probability = models.predict_variant_a(features)
        model_name = MODEL_A_NAME
    else:
        probability = models.predict_variant_b(features)
        model_name = MODEL_B_NAME

    log_prediction(request_id, payload, model_variant, model_name, probability)

    return PredictResponse(
        request_id=request_id,
        model_variant=model_variant,
        model_name=model_name,
        booking_probability=probability,
        user_id=payload.user_id,
        listing_id=payload.listing_id,
    )
