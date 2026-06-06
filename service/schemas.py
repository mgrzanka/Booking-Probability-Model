from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    user_id: int
    listing_id: int
    timestamp: Optional[str] = None

    days_since_last_review: float = -1
    alltime_review_scores_cleanliness: float = 0.0
    alltime_review_scores_location: float = 0.0
    alltime_review_scores_communication: float = 0.0
    alltime_review_scores_checkin: float = 0.0
    alltime_review_scores_value: float = 0.0
    alltime_review_scores_accuracy: float = 0.0
    alltime_review_scores_rating: float = 0.0
    alltime_review_count: int = 0
    recent_review_scores_cleanliness: float = 0.0
    recent_review_scores_location: float = 0.0
    recent_review_scores_communication: float = 0.0
    recent_review_scores_checkin: float = 0.0
    recent_review_scores_value: float = 0.0
    recent_review_scores_accuracy: float = 0.0
    recent_review_scores_rating: float = 0.0
    reviews_in_last_90_days: int = 0
    property_type: str = "Unknown"
    room_type: str = "Unknown"
    accommodates: float = 2.0
    bathrooms: float = 1.0
    bedrooms: float = 1.0
    price: float = 0.0
    neighbourhood_cleansed: str = "Unknown"

    Y: Optional[int] = Field(
        default=None,
        description="Opcjonalna etykieta historyczna do ewaluacji A/B w logu.",
    )


class PredictResponse(BaseModel):
    request_id: str
    model_variant: str
    model_name: str
    booking_probability: float
    user_id: int
    listing_id: int
