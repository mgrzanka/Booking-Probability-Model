from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MODEL_A_NAME = "LogisticRegression"
MODEL_B_NAME = "RandomForest"


class ModelBundle:
    def __init__(self):
        self.baseline_model = joblib.load(DATA_DIR / "baseline_logreg_model.joblib")
        self.target_model = joblib.load(DATA_DIR / "target_randomforest_model.joblib")
        self.encoder = joblib.load(DATA_DIR / "onehot_encoder.joblib")
        self.scaler = joblib.load(DATA_DIR / "standard_scaler.joblib")
        self.feature_names = joblib.load(DATA_DIR / "feature_columns.joblib")
        column_config = joblib.load(DATA_DIR / "column_config.joblib")
        self.num_cols = column_config["num_cols"]
        self.cat_cols = column_config["cat_cols"]

    def _prepare_features(self, features: dict) -> pd.DataFrame:
        row = {col: features.get(col) for col in self.num_cols + self.cat_cols}
        df = pd.DataFrame([row])
        df[self.num_cols] = df[self.num_cols].astype(float)
        df[self.num_cols] = df[self.num_cols].fillna(df[self.num_cols].median())
        df[self.cat_cols] = df[self.cat_cols].fillna("Unknown").astype(str)
        return df

    def _encode(self, df: pd.DataFrame, scale: bool = False) -> np.ndarray:
        x_num = df[self.num_cols].copy()
        if scale:
            x_num = self.scaler.transform(x_num)
        else:
            x_num = x_num.values
        x_cat = self.encoder.transform(df[self.cat_cols])
        return np.hstack([x_num, x_cat])

    def predict_variant_a(self, features: dict) -> float:
        df = self._prepare_features(features)
        encoded = self._encode(df, scale=True)
        return float(self.baseline_model.predict_proba(encoded)[:, 1][0])

    def predict_variant_b(self, features: dict) -> float:
        df = self._prepare_features(features)
        encoded = self._encode(df, scale=False)
        return float(self.target_model.predict_proba(encoded)[:, 1][0])


def extract_feature_dict(request_data: dict) -> dict:
    feature_keys = [
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
        "property_type",
        "room_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "price",
        "neighbourhood_cleansed",
    ]
    return {key: request_data.get(key) for key in feature_keys}
