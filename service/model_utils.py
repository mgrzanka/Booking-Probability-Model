from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Kolumny cech, których oczekują zapisane pipeline'y (impute + encode + scale + model).
NUM_FEATURES = [
    "days_since_last_review",
    "alltime_review_scores_cleanliness",
    "alltime_review_scores_location",
    "alltime_review_scores_communication",
    "alltime_review_scores_checkin",
    "alltime_review_scores_value",
    "alltime_review_scores_accuracy",
    "alltime_review_scores_rating",
    "alltime_review_count",
    "recent_review_scores_cleanliness",
    "recent_review_scores_location",
    "recent_review_scores_communication",
    "recent_review_scores_checkin",
    "recent_review_scores_value",
    "recent_review_scores_accuracy",
    "recent_review_scores_rating",
    "reviews_in_last_90_days",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "price",
    "host_acceptance_rate",
    "minimum_nights",
    "minimum_maximum_nights",
]
CAT_FEATURES = ["property_type", "room_type", "neighbourhood_cleansed", "host_is_superhost"]


def extract_feature_dict(request_data: dict) -> dict:
    return {key: request_data.get(key) for key in NUM_FEATURES + CAT_FEATURES}


class ModelBundle:
    """Ładuje dwa samowystarczalne pipeline'y (wariant A i B testu A/B)."""

    def __init__(self):
        self.model_a = joblib.load(DATA_DIR / "model_a_pipeline.joblib")
        self.model_b = joblib.load(DATA_DIR / "model_b_pipeline.joblib")
        self.model_a_name = type(self.model_a.named_steps["clf"]).__name__
        self.model_b_name = type(self.model_b.named_steps["clf"]).__name__

    def _to_frame(self, features: dict) -> pd.DataFrame:
        row = {col: features.get(col) for col in NUM_FEATURES + CAT_FEATURES}
        return pd.DataFrame([row])

    def predict_variant_a(self, features: dict) -> float:
        return float(self.model_a.predict_proba(self._to_frame(features))[:, 1][0])

    def predict_variant_b(self, features: dict) -> float:
        return float(self.model_b.predict_proba(self._to_frame(features))[:, 1][0])
